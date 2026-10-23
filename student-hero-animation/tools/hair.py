"""Step 3: split the hair off boy-full.png into hair_top / hair_left / hair_right overlays and build boy.png,
whose hair is a slightly smaller solid core (outer 3px replaced by a solid dark hair colour inside an eroded
silhouette) so the moving overlays never reveal a ghost edge."""
import json, numpy as np, cv2
from PIL import Image
ROOT = __file__.rsplit('/tools/', 1)[0]
SP = '/tmp/claude-0/-home-claude/16f7c1b2-c4b4-5aea-9d02-af57212e0f40/scratchpad'
boy = np.asarray(Image.open(f'{ROOT}/assets/layers/boy-full.png')).astype(np.float32)
H, W = boy.shape[:2]
rgb = boy[..., :3]; a = boy[..., 3] / 255
hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
h, s, v = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255

# head region (measured on the 1198 x 800 image)
HX0, HX1, HY0, HY1 = 680, 1000, 60, 360
head = np.zeros((H, W), bool); head[HY0:HY1, HX0:HX1] = True
# skin: warm hue, some saturation, not dark
skin = head & (a > 0.5) & ((h < 40) | (h > 340)) & (s > 0.18) & (v > 0.38)
skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
n, lab, st, _ = cv2.connectedComponentsWithStats(skin.astype(np.uint8)); skin = lab == 1 + np.argmax(st[1:, 4])
cols = np.where(skin.any(0))[0]; FX0, FX1 = int(cols.min()), int(cols.max())
top_of_skin = np.full(W, H, int)
for x in cols: top_of_skin[x] = np.where(skin[:, x])[0].min()
forehead = int(np.median(top_of_skin[FX0 + 30:FX1 - 30]))

# hair: dark, not royal-blue hoodie, inside the head box, above the skin in face columns, and above ear bottom at the sides
blue = (h > 190) & (h < 250) & (s > 0.35)
dark = (v < 0.42) & ~blue
Y, X = np.mgrid[0:H, 0:W]
above_face = Y < (top_of_skin[None, :].repeat(H, 0) + 4)
sides = ((X < FX0 + 18) | (X > FX1 - 18)) & (Y < 300)
hair = head & (a > 0.02) & ((dark & above_face) | (dark & sides))
# the outer soft edge of the hair (light strands over background) belongs to the hair too
hair = cv2.morphologyEx(hair.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)
n, lab, st, _ = cv2.connectedComponentsWithStats(hair.astype(np.uint8)); hair = lab == 1 + np.argmax(st[1:, 4])
grow = cv2.dilate(hair.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
hair = hair | (grow & (a < 0.98) & head & ~skin)        # include soft strands at the silhouette, never skin
hm = cv2.GaussianBlur(hair.astype(np.float32), (0, 0), 1.0)   # 1px feather

# three pieces: top (above the forehead, plus everything over the face columns), left side, right side
SPLIT_Y = forehead + 6
wl = np.clip((FX0 + 30 - X) / 12.0, 0, 1) * np.clip((Y - SPLIT_Y + 6) / 12.0, 0, 1)
wr = np.clip((X - (FX1 - 30)) / 12.0, 0, 1) * np.clip((Y - SPLIT_Y + 6) / 12.0, 0, 1)
wt = np.clip(1 - wl - wr, 0, 1)
pieces = {'hair_top': wt, 'hair_left': wl, 'hair_right': wr}
info = {}
for name, w in pieces.items():
    al = a * hm * w
    out = np.dstack([rgb, al * 255])
    ys, xs = np.where(al > 0.004); x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    Image.fromarray(np.clip(out[y0:y1, x0:x1] + 0.5, 0, 255).astype(np.uint8)).save(f'{ROOT}/assets/layers/{name}.png')
    info[name] = dict(x=int(x0), y=int(y0), w=int(x1 - x0), h=int(y1 - y0))

# body layer: the hair's outer 3px becomes a solid dark core inside an eroded silhouette
outer = (a < 0.5).astype(np.uint8)
dist_out = cv2.distanceTransform(1 - outer, cv2.DIST_L2, 5)          # distance from the boy's outside
band = hair & (dist_out < 4)
core_a = np.clip((dist_out - 3.0) / 1.0, 0, 1)                         # silhouette pulled in by 3px, 1px soft
hair_col = np.median(rgb[hair & (a > 0.98) & (v < 0.3)], axis=0)
body = boy.copy()
body[band, :3] = hair_col
body[band, 3] = 255 * core_a[band]
Image.fromarray(np.clip(body + 0.5, 0, 255).astype(np.uint8)).save(f'{ROOT}/assets/layers/boy.png')
info['meta'] = dict(face=[FX0, FX1], forehead=forehead, split=SPLIT_Y, hair_colour=[int(c) for c in hair_col])
json.dump(info, open(f'{ROOT}/tools/hair_layers.json', 'w'), indent=1)
print(json.dumps(info))
vis = np.zeros((H, W, 3), np.uint8) + 255
vis[skin] = (240, 190, 150); vis[(hm > 0.3) & (wt > 0.5)] = (200, 40, 40); vis[(hm > 0.3) & (wl > 0.5)] = (40, 160, 40); vis[(hm > 0.3) & (wr > 0.5)] = (40, 40, 200)
Image.fromarray(vis[40:380, 640:1040]).resize((800, 680), Image.NEAREST).save(f'{SP}/hair_split.png')
