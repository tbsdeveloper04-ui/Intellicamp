"""Step 3: clean background plate (people, cards, sparks and dot grid removed) and the card / spark / dot layers.
Each card is a fitted rounded rectangle: the part hidden behind the people (the chart card's lower edge sits behind
the mother's hair) is rebuilt in the card's own colour so the whole card can rise from behind them."""
import json, numpy as np, cv2
from PIL import Image
ROOT = __file__.rsplit('/tools/', 1)[0]
SP = '/tmp/claude-0/-home-claude/16f7c1b2-c4b4-5aea-9d02-af57212e0f40/scratchpad/parentani'
L = f'{ROOT}/assets/layers'
src = np.asarray(Image.open(f'{ROOT}/assets/hero-source.png').convert('RGBA')).astype(np.float32)
H, W = src.shape[:2]; clip = src[..., 3] / 255
I = src[..., :3].copy()
for y in range(H):                                        # give the transparent corners the nearest row colour
    xs = np.where(clip[y] > 0.99)[0]
    if len(xs): I[y, :xs[0]] = I[y, xs[0]]; I[y, xs[-1] + 1:] = I[y, xs[-1]]
ap = np.load(f'{SP}/alpha_people.npy')
people = ap > 0.02
hsv = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

CARDS = {'chart': (606, 0, 950, 250), 'heart': (976, 86, 1144, 238), 'cap': (1060, 250, 1188, 378)}
# card corners traced from a gridded close-up (TL, TR, BR, BL); the chart card's bottom-right is behind the hair
SEEDS = {'chart': [(615, 71), (931, 8), (948, 174), (632, 237)], 'heart': [(1001, 98), (1134, 108), (1119, 231), (986, 218)],
         'cap': [(1086, 260), (1182, 271), (1167, 371), (1077, 358)]}
SPARKS = [(548, 82, 598, 138), (1138, 208, 1194, 266)]
DOTS = (1098, 0, 1184, 82)
M = 26                                                    # shadow margin around each card

spark_px = np.zeros((H, W), bool)
for x0, y0, x1, y1 in SPARKS:
    m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = True
    spark_px |= m & (hsv[..., 1] > 90) & ((hsv[..., 0] < 25) | (hsv[..., 0] > 165))
spark_zone = cv2.dilate(spark_px.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)

def normconv(K, unknown):
    fill = np.zeros_like(I); have = np.zeros((H, W), bool)
    for sigma, thr in [(3, .2), (6, .15), (12, .1), (24, .06), (48, .03), (96, .01), (200, 1e-3), (400, 1e-7)]:
        num = cv2.GaussianBlur(I * K[..., None], (0, 0), sigma); den = cv2.GaussianBlur(K, (0, 0), sigma)
        est = num / np.maximum(den[..., None], 1e-12)
        take = (den > thr) & ~have & unknown; fill[take] = est[take]; have |= take
    return fill
def largest(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    return (lab == 1 + np.argmax(st[1:, 4])) if n > 1 else m.astype(bool)

# ---- pass 1: rough plate (whole boxes cleared) to find each card's real shape
unknown = cv2.dilate(people.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
for x0, y0, x1, y1 in list(CARDS.values()) + SPARKS + [DOTS]:
    unknown[max(y0 - M, 0):y1 + M, max(x0 - M, 0):x1 + M] = True
plate0 = normconv((~unknown).astype(np.float32), unknown)

# card panels: lighter than the plate, then a fitted (rotated) rectangle with rounded corners
panels = {}
for name, (x0, y0, x1, y1) in CARDS.items():
    sub = np.zeros((H, W), bool); sub[y0:y1, x0:x1] = True
    seed = np.zeros((H, W), np.uint8); cv2.fillPoly(seed, [np.array(SEEDS[name], np.int32)], 1)
    seed = cv2.dilate(seed, np.ones((9, 9), np.uint8)).astype(bool)           # traced corners, grown 4px
    light = seed & (hsv[..., 1] < 12) & (hsv[..., 2] > 235) & ~people     # card body: saturation under 10; the glow around it is 17+
    light = cv2.morphologyEx(light.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n_, lab_, st_, _ = cv2.connectedComponentsWithStats(light)
    big = st_[1:, 4].max()
    light = np.isin(lab_, [i for i in range(1, n_) if st_[i, 4] > big * 0.05])   # icon holes do not matter for the extents
    # robust rotated-rectangle fit: the angle with the smallest box over the 0.2..99.8 percentile extents
    ys_, xs_ = np.where(light); best = None
    for ang in np.arange(-20, 20.01, 0.25):
        t_ = np.deg2rad(ang); u_ = xs_ * np.cos(t_) + ys_ * np.sin(t_); v_ = -xs_ * np.sin(t_) + ys_ * np.cos(t_)
        u0, u1 = np.percentile(u_, [0.2, 99.8]); v0, v1 = np.percentile(v_, [0.2, 99.8])
        if best is None or (u1 - u0) * (v1 - v0) < best[0]: best = ((u1 - u0) * (v1 - v0), ang, u0, u1, v0, v1)
    _, ang, u0, u1, v0, v1 = best; t_ = np.deg2rad(ang)
    uc, vc = (u0 + u1) / 2, (v0 + v1) / 2
    cx, cy = uc * np.cos(t_) - vc * np.sin(t_), uc * np.sin(t_) + vc * np.cos(t_)
    rw, rh = u1 - u0 + 1.5, v1 - v0 + 1.5
    # corner radius: how much of the rect's corner is missing from the visible panel
    r = max(8.0, min(rw, rh) * 0.11)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = np.deg2rad(ang); ca, sa = np.cos(t), np.sin(t)
    u = (xx - cx) * ca + (yy - cy) * sa; v = -(xx - cx) * sa + (yy - cy) * ca
    qx = np.abs(u) - (rw / 2 - r); qy = np.abs(v) - (rh / 2 - r)
    sd = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r   # signed distance
    cov = np.clip(0.5 - sd, 0, 1)
    panels[name] = dict(cov=cov, sd=sd, rect=[float(cx), float(cy), float(rw), float(rh), float(ang), float(r)])
    print(name, 'rect', [round(v_, 1) for v_ in panels[name]['rect']], 'hidden px', int(((cov > 0.5) & people).sum()))

# ---- pass 2: plate cleared only where something really is
known_clear = ~(cv2.dilate(people.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool))
unknown = ~known_clear
for name, p in panels.items():
    unknown |= p['sd'] < M * 0.8                        # card + its soft shadow
sat = (hsv[..., 1] > 90) & ((hsv[..., 0] < 25) | (hsv[..., 0] > 165))   # orange sparks and dots
for x0, y0, x1, y1 in SPARKS + [DOTS]:
    m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = True
    unknown |= cv2.dilate((m & (sat | (hsv[..., 1] > 60))).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
fill = normconv((~unknown).astype(np.float32), unknown)
fill = np.where(unknown[..., None], cv2.GaussianBlur(np.where(unknown[..., None], fill, I), (0, 0), 2.5), I)
seam = cv2.GaussianBlur(unknown.astype(np.float32), (0, 0), 2)[..., None]
plate = np.where(unknown[..., None], I * (1 - seam) + fill * seam, I)
PAD = 24
pp = cv2.copyMakeBorder(plate, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
Image.fromarray(np.clip(pp + .5, 0, 255).astype(np.uint8)).save(f'{L}/background.jpg', quality=93, subsampling=0)
Image.fromarray(np.clip(plate, 0, 255).astype(np.uint8)).save(f'{SP}/plate.png')

def save(arr, name):
    ys, xs = np.where(arr[..., 3] > 1); x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    Image.fromarray(np.clip(arr[y0:y1, x0:x1] + .5, 0, 255).astype(np.uint8)).save(f'{L}/{name}.png')
    return dict(file=f'assets/layers/{name}.png', x=int(x0), y=int(y0), w=int(x1 - x0), h=int(y1 - y0))
def unmix(Ib, Bb):
    up = np.where(Ib > Bb, (Ib - Bb) / np.maximum(255 - Bb, 1), 0); dn = np.where(Ib < Bb, (Bb - Ib) / np.maximum(Bb, 1), 0)
    return np.clip(np.maximum(up, dn).max(2), 0, 1)

info = {'cards': {}}
A_raw = unmix(I, plate)
for name, p in panels.items():
    cov, sd = p['cov'], p['sd']
    body = cov > 0.02
    ring = (sd > 0) & (sd < M) & ~people & ~spark_zone
    shadow = np.minimum(cv2.GaussianBlur(A_raw * (hsv[..., 1] < 120) * ring, (0, 0), 2.0), 0.7) * np.clip(1 - sd / M, 0, 1) ** 1.2
    # rebuild the hidden shadow (behind the people) from the visible shadow at the same distance from the card
    vis_sh = ring & (shadow > 0.002)
    if vis_sh.any():
        dists = np.clip(sd[vis_sh].astype(int), 0, M - 1)
        prof = np.bincount(dists, weights=shadow[vis_sh], minlength=M) / np.maximum(np.bincount(dists, minlength=M), 1)
        hid = (sd > 0) & (sd < M) & people
        shadow[hid] = prof[np.clip(sd[hid].astype(int), 0, M - 1)]
    shadow = cv2.GaussianBlur(shadow, (0, 0), 2.5); shadow[shadow < 0.035] = 0
    a = np.where(body, cov, shadow)
    F = np.where(a[..., None] > 0, plate + (I - plate) / np.maximum(a[..., None], 1e-3), 0)
    # card body pixels hidden by the people: the card's own colour
    vis_body = (cov > 0.99) & ~cv2.dilate(people.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    card_col = np.median(I[vis_body & (hsv[..., 1] < 25)], axis=0)
    hidden_body = body & cv2.dilate(people.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    F[hidden_body] = card_col
    sh_col = np.median(F[ring & (shadow > 0.05) & ~people], axis=0) if (ring & (shadow > 0.05) & ~people).any() else np.array([200, 160, 130.])
    F[(sd > 0) & people] = sh_col
    full = np.dstack([np.clip(F, 0, 255), a * 255])
    # coloured icon pieces inside the card (orange / red), the card under them inpainted
    interior = cov > 0.99
    colour = interior & (hsv[..., 1] > (30 if name == 'chart' else 70)) & ~people & ((hsv[..., 0] < 30) | (hsv[..., 0] > 160))
    base = full.copy()
    hole = cv2.dilate(colour.astype(np.uint8), np.ones((7, 7), np.uint8))
    base[..., :3] = cv2.inpaint(np.clip(full[..., :3], 0, 255).astype(np.uint8), hole, 5, cv2.INPAINT_TELEA)
    c = {'card': save(base, f'card-{name}'), 'rect': p['rect']}
    n, lab, st, cen = cv2.connectedComponentsWithStats(colour.astype(np.uint8), connectivity=8)
    comps = [(lab == i, st[i], cen[i]) for i in range(1, n) if st[i, 4] > 15]
    piece = lambda m: np.dstack([full[..., :3], full[..., 3] * cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8))])
    if name == 'chart':
        # the arrow is the component that is widest relative to its area (a thin diagonal stroke)
        comps.sort(key=lambda q: -(q[1][2] * q[1][3]) / q[1][4])
        arrow = comps[0]
        inner = sd < -8                                          # bars sit well inside the card, never on its edge
        bars = sorted([q for q in comps[1:] if q[1][3] >= 30 and inner[q[0]].all()], key=lambda q: q[2][0])
        c['arrow'] = save(piece(arrow[0]), 'chart-arrow')
        c['bars'] = [save(piece(m), f'chart-bar-{i + 1}') for i, (m, _, _) in enumerate(bars)]
    else:
        allm = np.zeros((H, W), bool)
        inner = sd < -8                                          # the icon sits well inside the card, never on its edge
        for m, _, _ in comps:
            if inner[m].all(): allm |= m
        c['icon'] = save(piece(allm), f'{name}-icon')
    info['cards'][name] = c

# sparks: each tick its own layer, origin at the end nearest the card it belongs to
info['sparks'] = []
for gi, (x0, y0, x1, y1) in enumerate(SPARKS):
    m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = True; m &= sat
    n, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    target = panels['chart' if gi == 0 else 'cap']['rect'][:2]
    for i in range(1, n):
        if st[i, 4] < 10: continue
        mm = lab == i
        a = np.clip(cv2.GaussianBlur(unmix(I, plate) * cv2.dilate(mm.astype(np.uint8), np.ones((5, 5), np.uint8)), (0, 0), .5), 0, 1)
        F = np.where(a[..., None] > 0, plate + (I - plate) / np.maximum(a[..., None], 1e-3), 0)
        d = save(np.dstack([np.clip(F, 0, 255), a * 255]), f'spark-{gi + 1}-{len([s for s in info["sparks"] if s["group"] == gi]) + 1}')
        ys, xs = np.where(mm); k = np.argmin((xs - target[0]) ** 2 + (ys - target[1]) ** 2)
        d['origin'] = [round((xs[k] - d['x']) / d['w'] * 100, 1), round((ys[k] - d['y']) / d['h'] * 100, 1)]; d['group'] = gi
        info['sparks'].append(d)
# dot grid: one layer
x0, y0, x1, y1 = DOTS
m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = True
a = unmix(I, plate) * m * (hsv[..., 1] > 30)
a = np.clip(cv2.GaussianBlur(a, (0, 0), .5), 0, 1); a[a < .03] = 0
F = np.where(a[..., None] > 0, plate + (I - plate) / np.maximum(a[..., None], 1e-3), 0)
info['dots'] = save(np.dstack([np.clip(F, 0, 255), a * 255]), 'dots')
json.dump(info, open(f'{ROOT}/tools/sky_layers.json', 'w'), indent=1)

# ---- check: plate + cards + sparks + dots + people + foreground == original
comp = plate.copy()
def over(dst, l, x, y):
    l = np.asarray(l).astype(np.float32); h, w = l.shape[:2]; h = min(h, H - y); w = min(w, W - x); l = l[:h, :w]
    al = l[..., 3:] / 255; dst[y:y + h, x:x + w] = l[..., :3] * al + dst[y:y + h, x:x + w] * (1 - al)
for c in info['cards'].values():
    over(comp, Image.open(f"{ROOT}/{c['card']['file']}"), c['card']['x'], c['card']['y'])
    for k in ('arrow', 'icon'):
        if k in c: over(comp, Image.open(f"{ROOT}/{c[k]['file']}"), c[k]['x'], c[k]['y'])
    for b in c.get('bars', []): over(comp, Image.open(f"{ROOT}/{b['file']}"), b['x'], b['y'])
for s_ in info['sparks']: over(comp, Image.open(f"{ROOT}/{s_['file']}"), s_['x'], s_['y'])
over(comp, Image.open(f"{ROOT}/{info['dots']['file']}"), info['dots']['x'], info['dots']['y'])
over(comp, Image.open(f'{L}/people.png'), 0, 0)
pl = json.load(open(f'{ROOT}/tools/people_layers.json'))['foreground']
over(comp, Image.open(f'{L}/foreground.png'), pl['x'], pl['y'])
d = np.abs(comp - src[..., :3]).mean(2)[clip > 0.99]
print('recomposite vs original: mean', d.mean().round(3), 'p99', np.percentile(d, 99).round(2))
Image.fromarray(np.clip(np.concatenate([src[..., :3], comp], 1), 0, 255).astype(np.uint8)).resize((W, H // 2)).save(f'{SP}/recomp.png')
print('sparks', len(info['sparks']), 'bars', len(info['cards']['chart']['bars']))
