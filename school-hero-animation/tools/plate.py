"""Step 2: clean plate (cards, card shadows and clouds removed, sky only), cloud sprites and card layers.
The building, trees and ground keep their original pixels. Clouds that touch the roofline get their hidden
lower part extended with a soft fade, so a drifting cloud never shows the straight cut of the roof."""
import json, numpy as np, cv2
from PIL import Image
ROOT = __file__.rsplit('/tools/', 1)[0]
SP = '/tmp/claude-0/-home-claude/16f7c1b2-c4b4-5aea-9d02-af57212e0f40/scratchpad/schoolani'
L = f'{ROOT}/assets/layers'
src = np.asarray(Image.open(f'{ROOT}/assets/hero-source.png').convert('RGBA')).astype(np.float32)
H, W = src.shape[:2]
I = src[..., :3]; clipA = src[..., 3] / 255
sky_soft = np.asarray(Image.open(f'{L}/skyline_mask.png')).astype(np.float32) / 255
sky = sky_soft > 0.5
top = np.array(json.load(open(f'{ROOT}/tools/skyline.json'))['top'])
CARDS = {'chart': (996, 58, 1154, 240), 'people': (864, 222, 1012, 366), 'cap': (1044, 248, 1182, 376)}
cardbox = np.zeros((H, W), bool)
for x0, y0, x1, y1 in CARDS.values(): cardbox[y0:y1, x0:x1] = True

def normconv(K, unknown):
    fill = np.zeros_like(I); have = np.zeros((H, W), bool)
    for sigma, thr in [(3, .2), (6, .15), (12, .1), (24, .06), (48, .03), (96, .01), (200, 1e-3), (400, 1e-7)]:
        num = cv2.GaussianBlur(I * K[..., None], (0, 0), sigma); den = cv2.GaussianBlur(K, (0, 0), sigma)
        est = num / np.maximum(den[..., None], 1e-12)
        take = (den > thr) & ~have & unknown
        fill[take] = est[take]; have |= take
    return fill

# local sky background: extend the sky smoothly over the building, then a large grey-scale opening removes
# bright blobs (clouds, card glow) smaller than the structuring element
E = np.where(sky[..., None], I, normconv(sky.astype(np.float32), ~sky))
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (141, 141))
BG = np.stack([cv2.dilate(cv2.erode(E[..., c], k), k) for c in range(3)], -1)
BG = cv2.GaussianBlur(BG, (0, 0), 12)
whiter = cv2.GaussianBlur((I - BG).mean(2), (0, 0), 2)
# clouds only around the cloud areas (from the prompt's hints, doubled to this image size, with margin)
CLOUD_ZONES = [(740, 330, 960, 510), (670, 410, 780, 500), (550, 470, 740, 620), (1140, 340, 1198, 450)]
zone = np.zeros((H, W), bool)
for x0, y0, x1, y1 in CLOUD_ZONES: zone[y0:y1, x0:x1] = True
cloud = sky & zone & (whiter > 5.0)
cloud = cv2.morphologyEx(cloud.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
n, lab, st, _ = cv2.connectedComponentsWithStats(cloud.astype(np.uint8))
cloud = np.isin(lab, [i for i in range(1, n) if st[i, 4] > 200])
cloud = cv2.dilate(cloud.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(bool) & sky
glow = sky & (whiter > 2.5) & cv2.dilate(cardbox.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))).astype(bool)
cardbox_x = cv2.dilate((cardbox | glow).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(bool)
cloud &= ~cv2.dilate(cardbox.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)   # never a piece of a card
print('cloud pixels', int(cloud.sum()), 'card area', int(cardbox_x.sum()))
unknown = sky & (cloud | cardbox_x)
K = (sky & ~unknown).astype(np.float32)
fill = normconv(K, unknown)
fill = np.where(unknown[..., None], cv2.GaussianBlur(np.where(unknown[..., None], fill, I), (0, 0), 3), I)
seam = cv2.GaussianBlur(unknown.astype(np.float32), (0, 0), 2.5)[..., None] * sky_soft[..., None]
plate = I * (1 - seam) + fill * seam
plate = np.where(sky[..., None], plate, I)                       # building, trees and ground: exact original
out = np.dstack([plate, src[..., 3:]])
Image.fromarray(np.clip(out + .5, 0, 255).astype(np.uint8)).save(f'{L}/base_clean.png')

# ---- cloud sprites: alpha against the plate, white-ish colour, extended under the roofline with a fade
info = {}
a_cl = np.clip(((I - plate) / np.maximum(255 - plate, 1)).mean(2), 0, 1)
a_cl = cv2.GaussianBlur(a_cl, (0, 0), 1.2) * cv2.GaussianBlur(cloud.astype(np.float32), (0, 0), 3)
a_cl[a_cl < 0.02] = 0
F_cl = np.where(a_cl[..., None] > 0.02, plate + (I - plate) / np.maximum(a_cl[..., None], 0.02), 255)
F_cl = np.clip(F_cl, 0, 255)
EXT = 60                                          # hidden lower part: a faded mirror of the cloud above the roof
a_ext = a_cl.copy(); F_ext = F_cl.copy()
for x in range(W):
    t = int(top[x]) - 2
    if t >= H - 1 or t < 4 or a_cl[max(t - 3, 0), x] < 0.03: continue
    for y in range(t, min(t + EXT, H)):
        src_y = max(t - (y - t) - 1, 0); k = (y - t) / EXT; f = (1 - k) ** 2
        a_ext[y, x] = max(a_ext[y, x] if sky[y, x] else 0, a_cl[src_y, x] * f); F_ext[y, x] = F_cl[src_y, x]
a_vis = a_cl                                       # visible clouds (before the hidden extension) define the groups
a_cl, F_cl = a_ext, F_ext
a_cl = cv2.GaussianBlur(a_cl, (0, 0), 1.0)
grp = cv2.morphologyEx((cv2.GaussianBlur(a_vis, (0, 0), 3) > 0.22).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(grp)
seeds = np.where(np.isin(lab, [i for i in range(1, n) if st[i, 4] >= 250]), lab, 0)
_, near = cv2.distanceTransformWithLabels((seeds == 0).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
sy, sx = np.where(seeds > 0); lut = np.zeros(near.max() + 1, np.int32); lut[near[seeds > 0]] = seeds[sy, sx]
owner = lut[near]
clouds = []
for i in sorted(set(seeds[seeds > 0].tolist())):
    m = cv2.GaussianBlur((owner == i).astype(np.float32), (0, 0), 1.5)
    al = a_cl * m
    ys, xs = np.where(al > 0.01)
    x0, x1, y0, y1 = max(xs.min() - 4, 0), min(xs.max() + 5, W), max(ys.min() - 4, 0), min(ys.max() + 5, H)
    k = len(clouds) + 1
    sprite = np.dstack([F_cl[y0:y1, x0:x1], al[y0:y1, x0:x1] * 255])
    Image.fromarray(np.clip(sprite + .5, 0, 255).astype(np.uint8)).save(f'{L}/cloud_{k}.png')
    clouds.append(dict(file=f'assets/layers/cloud_{k}.png', x=int(x0), y=int(y0), w=int(x1 - x0), h=int(y1 - y0)))
info['clouds'] = clouds
print('clouds', [(c['x'], c['y'], c['w'], c['h']) for c in clouds])

# ---- cards: body (frosted panel + soft shadow) and the blue icon pieces
def largest(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    return (lab == 1 + np.argmax(st[1:, 4])) if n > 1 else m.astype(bool)
def fill_holes(m):
    m = m.astype(np.uint8); inv = (1 - m).copy(); h, w = m.shape; ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inv, ff, (0, 0), 2); return (inv != 2) | (m == 1)
def save(arr, name, ox, oy, mask=None):
    a = arr.copy()
    if mask is not None: a[..., 3] *= mask
    ys, xs = np.where(a[..., 3] > 1); x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    Image.fromarray(np.clip(a[y0:y1, x0:x1] + .5, 0, 255).astype(np.uint8)).save(f'{L}/{name}.png')
    return dict(file=f'assets/layers/{name}.png', x=int(ox + x0), y=int(oy + y0), w=int(x1 - x0), h=int(y1 - y0))
cards = {}
# one alpha for everything that was cleared around the cards, then split between the three cards by nearest panel
up = np.where(I > plate, (I - plate) / np.maximum(255 - plate, 1), 0); dn = np.where(I < plate, (plate - I) / np.maximum(plate, 1), 0)
A_raw = np.clip(np.maximum(up, dn).max(2), 0, 1)
panels = np.zeros((H, W), np.int32)
for idx, (name, (x0, y0, x1, y1)) in enumerate(CARDS.items(), start=1):
    Ib = I[y0:y1, x0:x1]; Bb = plate[y0:y1, x0:x1]
    light = cv2.GaussianBlur(Ib.mean(2) - Bb.mean(2), (0, 0), 2) > 8
    pn = fill_holes(largest(cv2.morphologyEx(light.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))))
    pn = largest(cv2.morphologyEx(pn.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))))
    panels[y0:y1, x0:x1][pn] = idx
interior_all = np.zeros((H, W), bool)
for idx in range(1, 4): interior_all |= cv2.erode((panels == idx).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
dist_panel = cv2.distanceTransform((panels == 0).astype(np.uint8), cv2.DIST_L2, 5)
region = cardbox_x & sky & ~(cv2.dilate(cloud.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & (dist_panel > 2))
falloff = np.clip(1 - dist_panel / 30.0, 0, 1) ** 1.5           # the glow fades out within 30px of the card
A_all = np.where(cv2.dilate((panels > 0).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool), A_raw, cv2.GaussianBlur(A_raw, (0, 0), 2.0))
A_all = A_all * cv2.GaussianBlur(region.astype(np.float32), (0, 0), 5) * falloff ; A_all[interior_all] = 1
F_all = np.clip(np.where(A_all[..., None] > 0, plate + (I - plate) / np.maximum(A_all[..., None], 1e-3), 0), 0, 255)
_, near = cv2.distanceTransformWithLabels((panels == 0).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
ys_, xs_ = np.where(panels > 0); lut = np.zeros(near.max() + 1, np.int32); lut[near[panels > 0]] = panels[ys_, xs_]
owner = lut[near]
for idx, (name, (bx0, by0, bx1, by1)) in enumerate(CARDS.items(), start=1):
    wgt = cv2.GaussianBlur((owner == idx).astype(np.float32), (0, 0), 1.5)
    a_full = A_all * wgt
    keep = cv2.dilate((owner == idx).astype(np.uint8) * (dist_panel < 32).astype(np.uint8), np.ones((3, 3), np.uint8))
    n_, lab_, st_, _ = cv2.connectedComponentsWithStats(((a_full > 0.02) & keep.astype(bool)).astype(np.uint8))
    a_full *= np.isin(lab_, [j for j in range(1, n_) if st_[j, 4] > 400])
    ys, xs = np.where(a_full > 0.004)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    a = a_full[y0:y1, x0:x1]; Ib = I[y0:y1, x0:x1]
    hsv = cv2.cvtColor(Ib.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    interior = (panels[y0:y1, x0:x1] == idx) & interior_all[y0:y1, x0:x1]
    full = np.dstack([F_all[y0:y1, x0:x1], a * 255])
    # icon: saturated blue inside the panel; the card under it is inpainted
    icon = (hsv[..., 1] > 90) & (hsv[..., 0] > 95) & (hsv[..., 0] < 130) & interior
    icon_g = cv2.dilate(icon.astype(np.uint8), np.ones((3, 3), np.uint8))
    base = full.copy()
    base[..., :3] = cv2.inpaint(np.clip(full[..., :3], 0, 255).astype(np.uint8), cv2.dilate(icon_g, np.ones((5, 5), np.uint8)), 5, cv2.INPAINT_TELEA)
    c = {'card': save(base, f'card-{name}', x0, y0)}
    iconA = cv2.dilate(icon_g, np.ones((3, 3), np.uint8)).astype(np.float32)
    n, lab, st, cen = cv2.connectedComponentsWithStats(icon.astype(np.uint8), connectivity=8)
    comps = [(lab == i, st[i], cen[i]) for i in range(1, n) if st[i, 4] > 12]
    if name == 'chart':                                  # bars, left to right
        comps.sort(key=lambda q: q[2][0])
        c['parts'] = [save(full, f'chart-bar-{j + 1}', x0, y0, cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)) for j, (m, _, _) in enumerate(comps)]
    elif name == 'people':                               # left / centre / right person (head + body each)
        comps = [q for q in comps if q[1][4] > 40 and hsv[..., 1][q[0]].mean() > 150]      # solid blue only
        allm = np.zeros(icon.shape, bool)
        for m, _, _ in comps: allm |= m
        ix = np.where(allm.any(0))[0]; ix0, iw = ix.min(), ix.max() - ix.min()
        groups = {'left': [], 'centre': [], 'right': []}
        for m, s_, ce in comps:
            f = (ce[0] - ix0) / iw
            groups['left' if f < 0.36 else 'right' if f > 0.64 else 'centre'].append(m)
        c['parts'] = {}
        for k, ms in groups.items():
            mm = np.zeros(icon.shape, bool)
            for m in ms: mm |= m
            c['parts'][k] = save(full, f'people-{k}', x0, y0, cv2.dilate(mm.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32))
        c['ncomp'] = len(comps)
    else:                                                # cap + tassel (the thin piece hanging on the right)
        comps.sort(key=lambda q: -q[1][4])
        capm = comps[0][0].copy(); tas = np.zeros(icon.shape, bool)
        for m, s_, ce in comps[1:]:
            (tas if ce[0] > comps[0][2][0] + 10 else capm).__ior__(m)
        if not tas.any():                                # tassel attached: take the narrow column at the right end
            ys, xs = np.where(capm); xr = xs.max()
            col = np.zeros_like(capm); col[:, xr - 7:xr + 1] = True
            low = np.zeros_like(capm); low[int(np.percentile(ys, 45)):, :] = True
            tas = capm & col & low; capm &= ~tas
        c['parts'] = {'cap': save(full, 'cap-hat', x0, y0, cv2.dilate(capm.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)),
                      'tassel': save(full, 'cap-tassel', x0, y0, cv2.dilate(tas.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32))}
    cards[name] = c
info['cards'] = cards
json.dump(info, open(f'{ROOT}/tools/sky_layers.json', 'w'), indent=1)

# ---- check: plate + clouds + cards == original (sky)
comp = plate.copy()
def over(dst, l, x, y):
    h, w = l.shape[:2]; l = l.astype(np.float32); al = l[..., 3:] / 255
    h = min(h, H - y); w = min(w, W - x); l = l[:h, :w]; al = al[:h, :w]
    dst[y:y + h, x:x + w] = l[..., :3] * al + dst[y:y + h, x:x + w] * (1 - al)
for cl in clouds:
    sp = np.asarray(Image.open(f"{ROOT}/{cl['file']}")).astype(np.float32).copy()
    sp[..., 3] *= sky_soft[cl['y']:cl['y'] + sp.shape[0], cl['x']:cl['x'] + sp.shape[1]]
    over(comp, sp, cl['x'], cl['y'])
for name, c in cards.items():
    over(comp, np.asarray(Image.open(f"{ROOT}/{c['card']['file']}")), c['card']['x'], c['card']['y'])
    parts = c['parts'].values() if isinstance(c['parts'], dict) else c['parts']
    for p in parts: over(comp, np.asarray(Image.open(f"{ROOT}/{p['file']}")), p['x'], p['y'])
d = np.abs(comp - I).mean(2)[clipA > 0.99]
print('recomposite vs original: mean', d.mean().round(3), 'p99', np.percentile(d, 99).round(2))
Image.fromarray(np.clip(plate, 0, 255).astype(np.uint8)).save(f'{SP}/plate.png')
Image.fromarray(np.clip(np.concatenate([I, comp], 1), 0, 255).astype(np.uint8)).resize((W, H // 2)).save(f'{SP}/recomp.png')
