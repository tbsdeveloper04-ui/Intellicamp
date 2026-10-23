"""Step 1: skyline mask (white = sky, black = building, trees, ground).
Sky is flooded from the top over pixels with a low local gradient (sky and soft clouds are smooth; the roofline,
windows and leaves are sharp). The card boxes are forced to sky. The result is cleaned per column (the sky is
everything above the first non-sky pixel in each column), then smoothed and saved."""
import json, numpy as np, cv2
from PIL import Image
ROOT = __file__.rsplit('/tools/', 1)[0]
SP = '/tmp/claude-0/-home-claude/16f7c1b2-c4b4-5aea-9d02-af57212e0f40/scratchpad/schoolani'
src = np.asarray(Image.open(f'{ROOT}/assets/hero-source.png').convert('RGBA')).astype(np.float32)
H, W = src.shape[:2]
I = src[..., :3]
g = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
gs = cv2.GaussianBlur(g, (0, 0), 1.0)
grad = np.hypot(cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3))
hsv = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
green = (hsv[..., 0] > 25) & (hsv[..., 0] < 95) & (hsv[..., 1] > 50)            # leaves
warm = (hsv[..., 0] < 25) & (hsv[..., 1] > 60)                                      # wood panels
dark = g < 150                                                                       # glass, shadows
CARDS = [(1000, 62, 1150, 236), (868, 226, 1008, 362), (1048, 252, 1178, 372)]      # x0, y0, x1, y1 with shadows
smooth = (grad < 22) & ~green & ~warm & ~dark
for x0, y0, x1, y1 in CARDS: smooth[y0:y1, x0:x1] = True
smooth = cv2.morphologyEx(smooth.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
n, lab = cv2.connectedComponents(smooth)
sky = lab == lab[5, 5]
# column-wise automatic top (good where trees meet clear sky: the far left)
auto = np.full(W, H)
for x in range(W):
    nz = np.where(~sky[:, x])[0]; auto[x] = nz[0] if len(nz) else H
# roofline traced by hand from a gridded close-up (clouds next to the white walls defeat automatic tracing),
# then each column is snapped to the strongest horizontal edge within +-10px of the traced line
ROOF = [(566, 618), (637, 589), (662, 593), (662, 551), (732, 519), (785, 534), (845, 507), (845, 480),
        (1077, 379), (1197, 459)]
xs_, ys_ = zip(*ROOF)
poly = np.full(W, H, dtype=float)
for i in range(len(ROOF) - 1):
    (x0, y0), (x1, y1) = ROOF[i], ROOF[i + 1]
    if x1 == x0: poly[x0] = min(poly[x0], y0, y1); continue
    xx = np.arange(x0, x1 + 1); poly[xx] = np.minimum(poly[xx], y0 + (y1 - y0) * (xx - x0) / (x1 - x0))
dy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3)
top = auto.astype(float).copy()
for x in range(566, W):
    py = int(round(poly[x])); lo, hi = max(py - 10, 1), min(py + 11, H - 1)
    k = lo + int(np.argmax(np.abs(dy[lo:hi, x])))
    top[x] = k - 1 if abs(k - py) <= 10 else py
# roofs are straight: replace the snapped columns of each segment by a robust straight-line fit through them
for i in range(len(ROOF) - 1):
    (x0, y0), (x1, y1) = ROOF[i], ROOF[i + 1]
    if x1 - x0 < 12: continue
    xx = np.arange(x0 + 3, x1 - 2); yy = top[xx]
    A = np.c_[xx, np.ones_like(xx)].astype(float); coef = np.linalg.lstsq(A, yy, rcond=None)[0]
    for _ in range(3):                                           # drop outliers and refit
        r = np.abs(A @ coef - yy); keep = r < max(1.5, np.percentile(r, 70))
        coef = np.linalg.lstsq(A[keep], yy[keep], rcond=None)[0]
    xs_all = np.arange(x0, x1 + 1); top[xs_all] = coef[0] * xs_all + coef[1]
# vertical walls: keep the traced x exactly
top[662] = min(top[661:664].min(), 551); top[845] = min(top[844:847].min(), 480)
top[566:] -= 4                                   # include the thin white roof coping above the snapped edge
top[530:566] = np.maximum(top[530:566], 655)     # small balcony box left of the low wing: no spikes into the sky
# trees that rise above the roof count as non-sky
gm = (green & (cv2.GaussianBlur(green.astype(np.float32), (0, 0), 3) > 0.35)).astype(np.uint8)
n2, lab2, st2, _ = cv2.connectedComponentsWithStats(gm)
big = np.isin(lab2, [i for i in range(1, n2) if st2[i, 4] > 400])
for x in range(W):
    nz = np.where(big[:, x])[0]
    if len(nz): top[x] = min(top[x], nz[0])
top_s = np.round(cv2.GaussianBlur(top.astype(np.float32).reshape(1, -1), (5, 1), 0.7).ravel()).astype(int)
top_s = np.minimum(top_s, top.astype(int))       # never smooth sky into the building
mask = (np.arange(H)[:, None] < top_s[None, :]).astype(np.float32)
mask = cv2.GaussianBlur(mask, (0, 0), 0.8)                                            # anti-aliased edge
Image.fromarray((mask * 255).astype(np.uint8)).save(f'{ROOT}/assets/layers/skyline_mask.png')
json.dump({'top': top_s.tolist()}, open(f'{ROOT}/tools/skyline.json', 'w'))
over = I.copy(); m = mask < 0.5
over[m] = over[m] * 0.45 + np.array([255, 0, 0]) * 0.55
Image.fromarray(over.astype(np.uint8)).save(f'{SP}/skyline_overlay.png')
print('roof top min/max over x 500..1198:', top_s[500:].min(), top_s[500:].max())
