"""Step 3: foliage overlays (exact original pixels of each tree / hedge group, feathered) and the foreground strip
(planter box and hedges in front of the walking students). The base underneath keeps the same pixels, so a
small sway never shows a hole."""
import json, numpy as np, cv2
from PIL import Image
ROOT = __file__.rsplit('/tools/', 1)[0]
L = f'{ROOT}/assets/layers'
src = np.asarray(Image.open(f'{ROOT}/assets/hero-source.png').convert('RGBA')).astype(np.float32)
H, W = src.shape[:2]
I = src[..., :3].astype(np.uint8)
hsv = cv2.cvtColor(I, cv2.COLOR_RGB2HSV).astype(np.float32)
green = (hsv[..., 0] > 22) & (hsv[..., 0] < 95) & (hsv[..., 1] > 55) & (hsv[..., 2] > 50)
pink = ((hsv[..., 0] > 150) | (hsv[..., 0] < 8)) & (hsv[..., 1] > 30) & (hsv[..., 2] > 150)   # flowers
leafy = green | pink
# groups: box (x0, y0, x1, y1) and how strongly it sways (1 = tree, 0.4 = hedges)
GROUPS = {
    'tree_right':  ((1078, 440, 1198, 700), 1.0),
    'tree_medium': ((570, 600, 735, 770), 1.0),
    'trees_left':  ((300, 680, 575, 792), 0.8),
    'plants_door': ((760, 690, 870, 768), 0.6),
    'plants_right':((1000, 690, 1130, 790), 0.6),
    'hedges':      ((300, 740, 760, 800), 0.4),
}
info = {}
taken = np.zeros((H, W), bool)
for name, ((x0, y0, x1, y1), amp) in GROUPS.items():
    m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = leafy[y0:y1, x0:x1] & ~taken[y0:y1, x0:x1]
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    m = np.isin(lab, [i for i in range(1, n) if st[i, 4] > 25])
    taken |= m
    a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8) * (src[..., 3] / 255)
    ys, xs = np.where(a > 0.02); bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    out = np.dstack([I[by0:by1, bx0:bx1].astype(np.float32), a[by0:by1, bx0:bx1] * 255])
    Image.fromarray(np.clip(out + .5, 0, 255).astype(np.uint8)).save(f'{L}/foliage_{name}.png')
    info[name] = dict(file=f'assets/layers/foliage_{name}.png', x=int(bx0), y=int(by0), w=int(bx1 - bx0), h=int(by1 - by0), amp=amp)
# foreground: planter box + hedges left of the steps (students walk up the steps just right of it)
FG = (600, 748, 762, 800)
fg = np.zeros((H, W), np.float32); fg[FG[1]:FG[3], FG[0]:FG[2]] = 1
fg[FG[1]:FG[3], FG[2] - 2:FG[2]] = 0.5
fg = fg * (src[..., 3] / 255)
x0, y0, x1, y1 = FG
out = np.dstack([I[y0:y1, x0:x1].astype(np.float32), fg[y0:y1, x0:x1] * 255])
Image.fromarray(np.clip(out + .5, 0, 255).astype(np.uint8)).save(f'{L}/foreground.png')
info['foreground'] = dict(file='assets/layers/foreground.png', x=x0, y=y0, w=x1 - x0, h=y1 - y0)
json.dump(info, open(f'{ROOT}/tools/foliage_layers.json', 'w'), indent=1)
print(json.dumps(info))
