"""Step 2: build boy.png from the two model masks: gate ISNet with the filled human silhouette (drops the wave),
refine the edge with a guided filter, decontaminate edge colours against an inpainted background estimate.
Writes assets/layers/boy-full.png (the complete cut-out, before hair splitting)."""
import numpy as np, cv2
from PIL import Image
ROOT = __file__.rsplit('/tools/', 1)[0]
SP = '/tmp/claude-0/-home-claude/16f7c1b2-c4b4-5aea-9d02-af57212e0f40/scratchpad'
src = np.asarray(Image.open(f'{ROOT}/assets/hero-source.png').convert('RGBA')).astype(np.float32)
H, W = src.shape[:2]
I = src[..., :3] / 255
mi = np.load(f'{SP}/m_isnet.npy'); mh = np.load(f'{SP}/m_human.npy')

# gate: human silhouette with holes filled (the books sit in a hole of the human model), slightly grown
hm = (mh > 0.5).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(hm); hm = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
inv = (1 - hm).copy(); ff = np.zeros((H + 2, W + 2), np.uint8); cv2.floodFill(inv, ff, (0, 0), 2)
filled = ((inv != 2) | (hm == 1)).astype(np.uint8)
gate = cv2.GaussianBlur(cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))).astype(np.float32), (0, 0), 6)
a = np.clip(mi * gate, 0, 1)
# keep the main body only
n, lab, st, _ = cv2.connectedComponentsWithStats((a > 0.5).astype(np.uint8))
main = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
a *= cv2.GaussianBlur(cv2.dilate(main, np.ones((15, 15), np.uint8)).astype(np.float32), (0, 0), 3)

# guided filter on the edge band, using the photo as guide, so hair strands follow the real image
def guided(Ig, p, r, eps):
    """He et al. guided filter with a grey guide (cv2.ximgproc has no guidedFilter in this build)."""
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mI, mp = box(Ig), box(p); cov = box(Ig * p) - mI * mp; var = box(Ig * Ig) - mI * mI
    A = cov / (var + eps); b = mp - A * mI
    return box(A) * Ig + box(b)
gf = guided(cv2.cvtColor(I.astype(np.float32), cv2.COLOR_RGB2GRAY), a.astype(np.float32), 3, 1e-3)
band = cv2.dilate(((a > 0.02) & (a < 0.98)).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
a = np.where(band, np.clip(gf, 0, 1), a)
a[a < 0.015] = 0; a[a > 0.985] = 1
a *= (src[..., 3] / 255)                      # respect the image's own rounded-corner clip

# background estimate behind the edge: inpaint the boy region of the original
hole = cv2.dilate((a > 0.01).astype(np.uint8), np.ones((9, 9), np.uint8))
small = cv2.resize((I * 255).astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
bg = cv2.inpaint(small, cv2.resize(hole, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST), 6, cv2.INPAINT_TELEA)
B = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
np.save(f'{SP}/bg_estimate.npy', B)

# colour-based alpha on the soft edge: a pixel that looks like the background cannot be very opaque.
# a_c = |I - B| / |Fsolid - B| along the line between the nearest solid foreground colour and the background
solid0 = (a >= 0.98).astype(np.uint8)
_, idx0 = cv2.distanceTransformWithLabels(1 - solid0, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
ys0, xs0 = np.where(solid0); lut0 = np.zeros(idx0.max() + 1, int); lut0[idx0[solid0.astype(bool)]] = np.arange(len(ys0))
Fs = I[ys0[lut0[idx0]], xs0[lut0[idx0]]]
d = Fs - B; ac = np.clip(((I - B) * d).sum(2) / ((d * d).sum(2) + 1e-4), 0, 1)
contrast = np.sqrt((d * d).sum(2)) > 0.25          # only where foreground and background differ clearly (dark hair on light blue)
edge = (a > 0) & (a < 0.98) & contrast
a = np.where(edge, np.minimum(a, np.clip(ac * 1.15, 0, 1)), a)
a[a < 0.03] = 0

# decontaminate: solve I = a F + (1-a) B for F on the soft edge, then fill from the nearest solid pixel where a is tiny
F = I.copy()
soft = (a > 0.04) & (a < 0.98)
F[soft] = np.clip((I[soft] - (1 - a[soft, None]) * B[soft]) / a[soft, None], 0, 1)
solid = (a >= 0.98).astype(np.uint8)
_, idx = cv2.distanceTransformWithLabels(1 - solid, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
ys, xs = np.where(solid); lut = np.zeros(idx.max() + 1, int); lut[idx[solid.astype(bool)]] = np.arange(len(ys))
near = lut[idx]
weak = (a > 0) & (a <= 0.04)
F[weak] = I[ys[near[weak]], xs[near[weak]]]
# blend the solved colour towards the nearest solid colour a little to tame noise in very soft strands
w = np.clip((0.35 - a) / 0.35, 0, 1)[..., None] * soft[..., None]
F = F * (1 - 0.5 * w) + I[ys[near], xs[near]] * 0.5 * w

out = np.dstack([F * 255, a * 255])
Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)).save(f'{ROOT}/assets/layers/boy-full.png')
np.save(f'{SP}/alpha_boy.npy', a)
print('boy alpha coverage', (a > 0.5).mean().round(3))
