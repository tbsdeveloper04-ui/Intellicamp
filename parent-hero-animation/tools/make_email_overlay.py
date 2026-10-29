"""Step 4: the email cover's logo, headline and subtitle (from the design), as a still layer on top of the animation,
plus everything outside the photo (the email's card edge and corners). Text-shaped, so nothing in the photo is frozen."""
import numpy as np, cv2
from PIL import Image
ROOT = __file__.rsplit('/tools/', 1)[0]
D = '/tmp/claude-0/-home-claude/16f7c1b2-c4b4-5aea-9d02-af57212e0f40/scratchpad/parent/d2x-1.png'
des = np.asarray(Image.open(D).convert('RGB'))[:760].astype(np.float32)
hsv = cv2.cvtColor(des.astype(np.uint8), cv2.COLOR_RGB2HSV)
ink = (des.sum(2) < 500) | ((hsv[..., 1] > 120) & (hsv[..., 2] > 80))
boxes = np.zeros(ink.shape, bool)
for x0, y0, x1, y1 in [(80, 80, 400, 176), (80, 270, 500, 470), (80, 478, 640, 556)]: boxes[y0:y1, x0:x1] = True
m = cv2.dilate((ink & boxes).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32)
m[80:176, 80:400] = np.maximum(m[80:176, 80:400], cv2.dilate((ink[80:176, 80:400]).astype(np.uint8), np.ones((15, 15), np.uint8)))
src = Image.open(f'{ROOT}/assets/hero-source.png').convert('RGBA').resize((1204, 804), Image.LANCZOS)
pa = np.asarray(src)[..., 3].astype(np.float32)[1:761, 2:1202] / 255          # photo placed at (-2, -1), scale 1.005
m = np.maximum(m, 1 - pa); m[725:, :] = 1
m = cv2.GaussianBlur(m, (0, 0), 1.2)
Image.fromarray(np.clip(np.dstack([des, m * 255]) + .5, 0, 255).astype(np.uint8)).resize((600, 380), Image.LANCZOS).save(f'{ROOT}/assets/email-overlay.png')
print('ok')
