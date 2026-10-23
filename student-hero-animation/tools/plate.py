"""Step 4: clean background plate (boy and floating elements removed, filled smoothly from their surroundings,
edges extended so a gentle drift never shows a gap) and one transparent layer per floating element, unmixed
against the plate so each element reproduces the original pixels exactly when it sits in place.
Two passes: pass 1 clears whole boxes just to find each element's real shape; pass 2 clears only that shape,
so the pale blob edges around the elements survive in the plate."""
import json, numpy as np, cv2
from PIL import Image
ROOT = __file__.rsplit('/tools/', 1)[0]
SP = '/tmp/claude-0/-home-claude/16f7c1b2-c4b4-5aea-9d02-af57212e0f40/scratchpad'
src = np.asarray(Image.open(f'{ROOT}/assets/hero-source.png').convert('RGBA')).astype(np.float32)
H, W = src.shape[:2]
I = src[..., :3]; clip = src[..., 3] / 255
boy_a = np.load(f'{SP}/alpha_boy.npy')
PAD = 24

# element boxes on the 1198 x 800 image (x0, y0, x1, y1), with room for soft shadows
ELEMENTS = {
    'bulb':   (640, 34, 760, 150),
    'chart':  (944, 2, 1138, 186),
    'list':   (1004, 204, 1188, 388),
    'plane':  (1076, 376, 1198, 482),
    'dots':   (1112, 8, 1190, 54),
    'circle': (1096, 96, 1198, 252),
}
boy_hole = cv2.dilate((boy_a > 0.01).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) | (clip < 0.99)

def make_plate(unknown):
    """multi-scale normalised convolution fill: smooth, follows the surrounding gradients"""
    K = (~unknown).astype(np.float32)
    fill = np.zeros_like(I); have = np.zeros((H, W), np.float32)
    for sigma, thr in [(4, .15), (8, .15), (16, .12), (32, .08), (64, .04), (128, .01), (256, .001), (400, 1e-6)]:
        num = cv2.GaussianBlur(I * K[..., None], (0, 0), sigma); den = cv2.GaussianBlur(K, (0, 0), sigma)
        est = num / np.maximum(den[..., None], 1e-9)
        take = (den > thr) & (have == 0) & unknown
        fill[take] = est[take]; have[take] = 1
    assert (have[unknown] > 0).all(), 'unfilled pixels left'
    plate = np.where(unknown[..., None], fill, I)
    seam = cv2.GaussianBlur(unknown.astype(np.float32), (0, 0), 2)[..., None]
    plate = I * (1 - seam) + cv2.GaussianBlur(plate, (0, 0), 1.2) * seam
    return np.where(unknown[..., None], plate, I)

def unmix(Ib, Bb):
    """minimal alpha so that I = a F + (1 - a) B with F in [0, 255]"""
    up = np.where(Ib > Bb, (Ib - Bb) / np.maximum(255 - Bb, 1), 0)
    dn = np.where(Ib < Bb, (Bb - Ib) / np.maximum(Bb, 1), 0)
    return np.clip(np.maximum(up, dn).max(2), 0, 1)
def largest(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    return (lab == 1 + np.argmax(st[1:, 4])) if n > 1 else m.astype(bool)
def fill_holes(m):
    m = m.astype(np.uint8); inv = (1 - m).copy(); h, w = m.shape; ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inv, ff, (0, 0), 2); return (inv != 2) | (m == 1)

def extract(plate, write):
    info = {'pad': PAD}; masks = np.zeros((H, W), bool)
    for name, (x0, y0, x1, y1) in ELEMENTS.items():
        Ib = I[y0:y1, x0:x1]; Bb = plate[y0:y1, x0:x1]
        hsv = cv2.cvtColor(Ib.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        notboy = boy_a[y0:y1, x0:x1] < 0.02
        a_raw = unmix(Ib, Bb)
        if name in ('bulb', 'dots', 'plane'):
            # blue line art: saturated blue strokes, grown 2px for anti-aliasing
            stroke = (Ib[..., 2] > Ib[..., 0] + 45) & (hsv[..., 1] > 90) & notboy
            if name == 'plane': stroke &= (Ib[..., 2] > 150) & (hsv[..., 2] > 110) & (hsv[..., 1] > 150)
            m = cv2.dilate(stroke.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            a = a_raw * m
        elif name in ('chart', 'list'):
            # card: the frosted panel is lighter than what is behind it; then its soft shadow around it
            light = cv2.GaussianBlur((Ib.mean(2) - Bb.mean(2)), (0, 0), 2) > 6
            card = fill_holes(largest(cv2.morphologyEx(light.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))))
            card = largest(cv2.morphologyEx(card.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))))
            body = cv2.dilate(card.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            halo = cv2.dilate(card.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))).astype(bool)
            ring = halo & ~body
            lowsat = hsv[..., 1] < 90
            # soft shadow and frosted halo around the card, smoothed so no speckle travels with it
            shadow = np.minimum(cv2.GaussianBlur(a_raw * lowsat, (0, 0), 2.5), 0.6) * ring
            masks[y0:y1, x0:x1] |= halo                     # the plate is rebuilt under the whole halo
            interior = cv2.erode(card.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            a = np.where(body, a_raw, shadow)
            a[interior] = 1.0                      # solid card body: the original frosted pixels, no speckle
            a *= notboy
        else:
            # circle: the pale disc, unmixed from a lightly smoothed image so it stays clean
            dif = cv2.GaussianBlur(np.abs(Ib - Bb).mean(2), (0, 0), 4)
            disc = fill_holes(largest(dif > 2.5))
            disc = cv2.dilate(disc.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            Ib = cv2.GaussianBlur(Ib, (0, 0), 1.5)
            a = cv2.GaussianBlur(unmix(Ib, Bb), (0, 0), 1.5) * disc * notboy
        # fade the layer out towards its box edges so nothing shows a straight cut while it moves
        # (edges on the image's own border stay: the container clips them there anyway)
        hh, ww = a.shape; yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        dl = xx if x0 > 0 else np.full_like(xx, 99); dr = (ww - 1 - xx) if x1 < W else np.full_like(xx, 99)
        dt = yy if y0 > 0 else np.full_like(yy, 99); db = (hh - 1 - yy) if y1 < H else np.full_like(yy, 99)
        if name in ('chart', 'list', 'circle'):   # line-art elements are already cut to their strokes
            a *= np.clip(np.minimum(np.minimum(dl, dr), np.minimum(dt, db)) / 12.0, 0, 1)
        a[a < 0.02] = 0
        masks[y0:y1, x0:x1] |= cv2.dilate((a > 0).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        if not write: continue
        F = np.where(a[..., None] > 0, Bb + (Ib - Bb) / np.maximum(a[..., None], 1e-3), 0)
        out = np.dstack([np.clip(F, 0, 255), a * 255])
        ys, xs = np.where(a > 0); bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        Image.fromarray(np.clip(out[by0:by1, bx0:bx1] + 0.5, 0, 255).astype(np.uint8)).save(f'{ROOT}/assets/layers/el-{name}.png')
        info[name] = dict(x=int(x0 + bx0), y=int(y0 + by0), w=int(bx1 - bx0), h=int(by1 - by0))
    return info, masks

# pass 1: clear whole boxes, only to find each element's real shape
unknown = boy_hole.copy()
for (x0, y0, x1, y1) in ELEMENTS.values(): unknown[y0:y1, x0:x1] = True
plate = make_plate(unknown)
_, masks = extract(plate, False)
# pass 2: clear only where an element really is
plate = make_plate(boy_hole | masks)
info, _ = extract(plate, True)
json.dump(info, open(f'{ROOT}/tools/element_layers.json', 'w'), indent=1)
plate_pad = cv2.copyMakeBorder(plate, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
Image.fromarray(np.clip(plate_pad + 0.5, 0, 255).astype(np.uint8)).save(f'{ROOT}/assets/layers/background.jpg', quality=92, subsampling=0)

# check: plate + elements + boy + hair over each other reproduces the source
def over(dst, layer, x, y):
    h, w = layer.shape[:2]; l = layer.astype(np.float32); al = l[..., 3:] / 255
    dst[y:y + h, x:x + w] = l[..., :3] * al + dst[y:y + h, x:x + w] * (1 - al)
comp = plate.copy()
for name, v in info.items():
    if name == 'pad': continue
    over(comp, np.asarray(Image.open(f'{ROOT}/assets/layers/el-{name}.png')), v['x'], v['y'])
over(comp, np.asarray(Image.open(f'{ROOT}/assets/layers/boy.png')), 0, 0)
hl = json.load(open(f'{ROOT}/tools/hair_layers.json'))
for n in ['hair_top', 'hair_left', 'hair_right']:
    over(comp, np.asarray(Image.open(f'{ROOT}/assets/layers/{n}.png')), hl[n]['x'], hl[n]['y'])
inside = clip > 0.99
d = np.abs(comp - I).mean(2)[inside]
print('composite vs source: mean', d.mean().round(3), 'p99', np.percentile(d, 99).round(2), 'max', d.max().round(1))
Image.fromarray(np.clip(np.concatenate([I, comp], 1), 0, 255).astype(np.uint8)).resize((W, H // 2)).save(f'{SP}/plate_check.png')
Image.fromarray(np.clip(plate, 0, 255).astype(np.uint8)).resize((W // 2, H // 2)).save(f'{SP}/plate_only.png')
