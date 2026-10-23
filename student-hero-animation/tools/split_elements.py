"""Step 5: split the element layers into the parts that animate on their own and write tools/parts.json
(bulb body + rays, chart card + 3 bars, list card + 4 lines, plane glyph + trail, individual dots, circle)."""
import json, numpy as np, cv2
from PIL import Image
ROOT = __file__.rsplit('/tools/', 1)[0]
L = f'{ROOT}/assets/layers'
info = json.load(open(f'{ROOT}/tools/element_layers.json'))
parts = {}

def load(n):
    return np.asarray(Image.open(f'{L}/el-{n}.png')).astype(np.float32)
def save(arr, name, ox, oy, mask=None):
    a = arr.copy()
    if mask is not None: a[..., 3] *= mask
    ys, xs = np.where(a[..., 3] > 1)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    Image.fromarray(np.clip(a[y0:y1, x0:x1] + 0.5, 0, 255).astype(np.uint8)).save(f'{L}/{name}.png')
    return dict(file=f'assets/layers/{name}.png', x=int(ox + x0), y=int(oy + y0), w=int(x1 - x0), h=int(y1 - y0))
def comps(mask, min_area=8):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(lab == i, st[i], cen[i]) for i in range(1, n) if st[i, 4] >= min_area]
def grow(m, k=3): return cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(np.float32)

# ---- bulb: largest pieces = glass + base; small pieces above / beside = rays
b = load('bulb'); ox, oy = info['bulb']['x'], info['bulb']['y']
cs = comps(b[..., 3] > 60)
cs.sort(key=lambda c: -c[1][4])
glass = cs[0]; gx, gy = glass[2]; gr = max(glass[1][2], glass[1][3]) / 2
body = np.zeros(b.shape[:2], bool); rays = []
for m, st, c in cs:
    if m is not glass[0] and np.hypot(c[0] - gx, c[1] - gy) > gr * 0.75 and c[1] < gy + gr * 0.6 and st[4] < glass[1][4] * 0.2: rays.append((m, c))
    else: body |= m
parts['bulb'] = save(b, 'bulb-body', ox, oy, grow(body))
parts['bulb']['centre'] = [ox + gx, oy + gy]
rays.sort(key=lambda r: np.arctan2(r[1][1] - gy, r[1][0] - gx))
parts['bulb_rays'] = []
for i, (m, c) in enumerate(rays):
    p = save(b, f'bulb-ray-{i + 1}', ox, oy, grow(m))
    # transform origin: the inner end (nearest the glass centre)
    ys, xs = np.where(m); d = np.hypot(xs - gx, ys - gy); k = d.argmin()
    p['origin'] = [round((ox + xs[k] - p['x']) / p['w'] * 100, 1), round((oy + ys[k] - p['y']) / p['h'] * 100, 1)]
    parts['bulb_rays'].append(p)

# ---- cards: saturated blue marks become their own layers; the card underneath is filled where they were
def card_split(name, n_marks, sort_key):
    c = load(name); ox, oy = info[name]['x'], info[name]['y']
    rgb = c[..., :3].astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    marks = (hsv[..., 1] > 110) & (c[..., 3] > 200)
    cs = sorted(comps(marks, 60), key=lambda m: -m[1][4])[:n_marks]
    allm = np.zeros(marks.shape, bool)
    for m, _, _ in cs: allm |= m
    hole = cv2.dilate(allm.astype(np.uint8), np.ones((7, 7), np.uint8))
    base = c.copy()
    base[..., :3] = cv2.inpaint(rgb, hole, 5, cv2.INPAINT_TELEA).astype(np.float32)
    out = {'card': save(base, f'{name}-card', ox, oy)}
    cs.sort(key=lambda m: sort_key(m[2]))
    out['marks'] = [save(c, f'{name}-mark-{i + 1}', ox, oy, grow(m, 5)) for i, (m, _, _) in enumerate(cs)]
    return out
parts['chart'] = card_split('chart', 3, lambda cen: cen[0])      # bars left to right
parts['list'] = card_split('list', 4, lambda cen: cen[1])        # lines top to bottom

# ---- plane: biggest piece is the plane, the dashes are the trail
p = load('plane'); ox, oy = info['plane']['x'], info['plane']['y']
cs = sorted(comps(p[..., 3] > 60), key=lambda c: -c[1][4])
plane = cs[0][0]; trail = np.zeros(plane.shape, bool)
for m, _, _ in cs[1:]: trail |= m
parts['plane'] = save(p, 'plane-glyph', ox, oy, grow(plane, 5))
parts['plane_trail'] = save(p, 'plane-trail', ox, oy, grow(trail, 5))

# ---- dots: one layer each so they can twinkle
d = load('dots'); ox, oy = info['dots']['x'], info['dots']['y']
cs = sorted(comps(d[..., 3] > 60, 4), key=lambda c: (round(c[2][1] / 8), c[2][0]))
parts['dots'] = [save(d, f'dot-{i + 1}', ox, oy, grow(m, 5)) for i, (m, _, _) in enumerate(cs)]

# ---- circle stays one piece
parts['circle'] = dict(file='assets/layers/el-circle.png', **info['circle'])
json.dump(parts, open(f'{ROOT}/tools/parts.json', 'w'), indent=1)
print('rays', len(parts['bulb_rays']), 'bars', len(parts['chart']['marks']), 'lines', len(parts['list']['marks']), 'dots', len(parts['dots']))
