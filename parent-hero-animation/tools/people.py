"""Step 2: people.png (mother + daughter, the moving layer) and foreground.png (laptop, desk, books, pencil cup:
original pixels, drawn still in front of the people so the laptop stays planted while they lean and nod).
Alpha: ISNet (fine hair) gated by the human-segmentation silhouette, refined with a guided filter and a colour test
against an estimate of the background, then edge colours decontaminated. The right edge is padded 24px by
replicating the last column, so the small lean never opens a gap at the border."""
import json, numpy as np, cv2
from PIL import Image
ROOT = __file__.rsplit('/tools/', 1)[0]
SP = '/tmp/claude-0/-home-claude/16f7c1b2-c4b4-5aea-9d02-af57212e0f40/scratchpad/parentani'
L = f'{ROOT}/assets/layers'
src = np.asarray(Image.open(f'{ROOT}/assets/hero-source.png').convert('RGBA')).astype(np.float32)
H, W = src.shape[:2]
clip = src[..., 3] / 255
# the transparent corners hold no colour: fill them from the nearest opaque pixel in the same row
I = src[..., :3].copy()
for y in range(H):
    xs = np.where(clip[y] > 0.99)[0]
    if len(xs) == 0: continue
    I[y, :xs[0]] = I[y, xs[0]]; I[y, xs[-1] + 1:] = I[y, xs[-1]]
I /= 255
mi = np.load(f'{SP}/m_isnet.npy'); mh = np.load(f'{SP}/m_human.npy')

# laptop = what ISNet sees that the human model does not, inside the laptop area
lap = (mi > 0.5) & (mh < 0.35)
box = np.zeros((H, W), bool); box[568:792, 560:1105] = True; lap &= box
lap = cv2.morphologyEx(lap.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(lap); lap = (lab == 1 + np.argmax(st[1:, 4]))
inv = (~lap).astype(np.uint8); ff = np.zeros((H + 2, W + 2), np.uint8); cv2.floodFill(inv, ff, (0, 0), 0); lap |= inv.astype(bool)

# people: human silhouette (filled, grown) gates ISNet, minus the laptop
hm = (mh > 0.5).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(hm); hm = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
gate = cv2.GaussianBlur(cv2.dilate(hm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))).astype(np.float32), (0, 0), 5)
a = np.clip(np.maximum(mi, mh * 0.0) * gate, 0, 1)
a[lap] = np.minimum(a[lap], mh[lap])                       # under the laptop keep only what the human model says
def guided(Ig, p, r, eps):
    box_ = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mI, mp = box_(Ig), box_(p); cov = box_(Ig * p) - mI * mp; var = box_(Ig * Ig) - mI * mI
    A = cov / (var + eps); b = mp - A * mI
    return box_(A) * Ig + box_(b)
# the pink sweater sleeve left of the laptop: both models miss it, the colour finds it
hsv = cv2.cvtColor((I * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
pinkz = np.zeros((H, W), bool); pinkz[380:770, 500:720] = True
pink = pinkz & ((hsv[..., 0] > 155) | (hsv[..., 0] < 10)) & (hsv[..., 1] > 28) & (hsv[..., 2] > 100) & ~lap
pink = cv2.morphologyEx(pink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
n_, lab_, st_, _ = cv2.connectedComponentsWithStats(pink); pink = np.isin(lab_, [i for i in range(1, n_) if st_[i, 4] > 300])
inv_ = (~pink).astype(np.uint8); ff_ = np.zeros((H + 2, W + 2), np.uint8); cv2.floodFill(inv_, ff_, (0, 0), 0); pink |= inv_.astype(bool)
pink = cv2.morphologyEx(pink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
a = np.maximum(a, cv2.GaussianBlur(pink.astype(np.float32), (0, 0), 0.8))
# the people continue a little behind the laptop's top and left edges, so a small lean never shows a gap there
lapd = cv2.dilate(lap.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
# EXT_IN: how far they continue behind the lid; EDGE: the lid's own soft edge rows/cols, kept still in the foreground
EXT_IN, EDGE, SRC = 44, 4, 7
ext = np.zeros((H, W), np.float32); edge_band = np.zeros((H, W), bool)
for x in range(560, 1105):
    col = np.where(lap[:, x])[0]
    if len(col) == 0: continue
    y0 = col.min()
    if a[max(y0 - SRC, 0), x] > 0.9: ext[y0 - EDGE:min(y0 + EXT_IN, H), x] = 1; edge_band[y0 - EDGE:y0, x] = True
for y in range(570, 790):
    row = np.where(lap[y, :])[0]
    if len(row) == 0: continue
    x0_ = row.min()
    if a[y, max(x0_ - SRC, 0)] > 0.9: ext[y, x0_ - EDGE:x0_ + EXT_IN] = 1; edge_band[y, x0_ - EDGE:x0_] = True
a_body = a.copy()                                  # alpha before the hidden extension (used by the foreground)
a = np.maximum(a, ext)
gf = guided(cv2.cvtColor(I.astype(np.float32), cv2.COLOR_RGB2GRAY), a.astype(np.float32), 3, 1e-3)
band = cv2.dilate(((a > 0.02) & (a < 0.98)).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
a = np.where(band, np.clip(gf, 0, 1), a); a[a < 0.015] = 0; a[a > 0.985] = 1
n, lab, st, _ = cv2.connectedComponentsWithStats((a > 0.3).astype(np.uint8))
main = cv2.dilate(np.isin(lab, [i for i in range(1, n) if st[i, 4] > 2000]).astype(np.uint8), np.ones((9, 9), np.uint8))
a *= cv2.GaussianBlur(main.astype(np.float32), (0, 0), 2)

# background estimate behind the edge (for the colour test and decontamination)
hole = cv2.dilate((a > 0.01).astype(np.uint8), np.ones((11, 11), np.uint8))
small = cv2.resize((I * 255).astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
B = cv2.resize(cv2.inpaint(small, cv2.resize(hole, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST), 8, cv2.INPAINT_TELEA), (W, H)).astype(np.float32) / 255
solid0 = (a >= 0.98).astype(np.uint8)
_, idx0 = cv2.distanceTransformWithLabels(1 - solid0, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
ys0, xs0 = np.where(solid0); lut0 = np.zeros(idx0.max() + 1, int); lut0[idx0[solid0.astype(bool)]] = np.arange(len(ys0))
Fs = I[ys0[lut0[idx0]], xs0[lut0[idx0]]]
d = Fs - B; ac = np.clip(((I - B) * d).sum(2) / ((d * d).sum(2) + 1e-4), 0, 1)
contrast = np.sqrt((d * d).sum(2)) > 0.25
edge = (a > 0) & (a < 0.98) & contrast
a = np.where(edge, np.minimum(a, np.clip(ac * 1.15, 0, 1)), a); a[a < 0.03] = 0
F = I.copy(); soft = (a > 0.04) & (a < 0.98)
F[soft] = np.clip((I[soft] - (1 - a[soft, None]) * B[soft]) / a[soft, None], 0, 1)
solid = (a >= 0.98).astype(np.uint8)
_, idx = cv2.distanceTransformWithLabels(1 - solid, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
ys, xs = np.where(solid); lut = np.zeros(idx.max() + 1, int); lut[idx[solid.astype(bool)]] = np.arange(len(ys))
near = lut[idx]; weak = (a > 0) & (a <= 0.04); F[weak] = I[ys[near[weak]], xs[near[weak]]]
w = np.clip((0.35 - a) / 0.35, 0, 1)[..., None] * soft[..., None]
F = F * (1 - 0.5 * w) + I[ys[near], xs[near]] * 0.5 * w

# hidden strip under the lid (and the lid's soft edge) takes the colour just outside the edge; visible people inside
# the laptop outline (hands) are left alone
hid = edge_band | (lap & (a_body < 0.5))
ytop = np.full(W, H)
for x in range(560, 1105):
    top = np.where(ext[:, x] > 0)[0]
    if len(top) == 0: continue
    ytop[x] = top.min() + EDGE; col = top[hid[top, x]]
    F[col, x] = F[max(top.min() - 3, 0), x]
for y in range(570, 790):                     # pixels nearer the left edge than the top edge take the sleeve colour
    rr = np.where(lap[y, :])[0]
    if len(rr) == 0 or ext[y, rr.min()] == 0: continue
    x0_ = rr.min(); span = np.arange(x0_ - EDGE, min(x0_ + EXT_IN, W))
    span = span[hid[y, span] & (ext[y, span] > 0) & ((span - x0_) < (y - ytop[span]))]
    if len(span): F[y, span] = F[y, max(x0_ - EDGE - 3, 0)]
# holes fully enclosed by the people and the laptop (the blouse just above the lid) are people too
Mfull = ((a > 0.5) | lap).astype(np.uint8)
inv_h = (1 - Mfull).copy(); ffh = np.zeros((H + 2, W + 2), np.uint8); cv2.floodFill(inv_h, ffh, (0, 0), 0)
holes = inv_h.astype(bool) & ~lap
holes = cv2.dilate(holes.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~lap & (a < 0.98)
a[holes] = 1.0; F[holes] = I[holes]; a_body[holes] = 1.0
print('filled enclosed holes', int(holes.sum()))
PAD = 24
people = np.dstack([F * 255, a * 255])
people = np.concatenate([people, np.repeat(people[:, -1:], PAD, 1)], 1)       # replicate the right edge
people = np.concatenate([people, np.repeat(people[-1:], PAD, 0)], 0)          # and the bottom edge
Image.fromarray(np.clip(people + .5, 0, 255).astype(np.uint8)).save(f'{L}/people.png')
np.save(f'{SP}/alpha_people.npy', a)

# foreground (static): the laptop, plus the desk zone (cup, books, desk, wave) wherever the people are not.
# The background in the desk zone is static too (the drift layer fades out there), so a rectangle is fine,
# but it keeps 6px away from the people so they move freely over the still background.
fg = ((lap & (a_body < 0.5)) | edge_band).astype(np.float32)
zone = np.zeros((H, W), np.float32); zone[596:H, 0:1105] = 1
near_people = cv2.dilate((a_body > 0.05).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))).astype(bool)
zone[near_people & ~lap] = 0
fg = np.maximum(fg, cv2.GaussianBlur(zone, (0, 0), 1.2) * (1 - np.clip(a_body * 1.6, 0, 1)))
fg = cv2.GaussianBlur(fg, (0, 0), 0.6) * clip
ys_, xs_ = np.where(fg > 0.004); x0, y0 = xs_.min(), ys_.min()
Image.fromarray(np.clip(np.dstack([src[..., :3], fg * 255])[y0:, x0:] + .5, 0, 255).astype(np.uint8)).save(f'{L}/foreground.png')
json.dump(dict(people=dict(x=0, y=0, w=W + PAD, h=H + PAD), foreground=dict(x=int(x0), y=int(y0), w=int(W - x0), h=int(H - y0))), open(f'{ROOT}/tools/people_layers.json', 'w'))
print('people coverage', (a > 0.5).mean().round(3), 'foreground at', x0, y0)
# previews
pp = Image.open(f'{L}/people.png')
tiles = []
for col in [(255, 255, 255), (253, 226, 196), (249, 166, 90)]:
    c = Image.new('RGBA', pp.size, col + (255,)); c.alpha_composite(pp); tiles.append(c.convert('RGB'))
o = Image.new('RGB', (1800, 400)); [o.paste(t.resize((600, 400)), (i * 600, 0)) for i, t in enumerate(tiles)]; o.save(f'{SP}/people_prev.png')
z = Image.new('RGB', (3 * 300, 300)); [z.paste(t.crop((640, 160, 940, 460)), (i * 300, 0)) for i, t in enumerate(tiles)]; z.resize((1350, 450)).save(f'{SP}/people_hair.png')
