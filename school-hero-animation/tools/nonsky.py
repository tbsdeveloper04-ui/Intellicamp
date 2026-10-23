"""Step 4: nonsky.png = the original building, trees and ground (alpha = 1 - sky mask), drawn over the clouds and
cards so both are always behind the school."""
import numpy as np
from PIL import Image
ROOT = __file__.rsplit('/tools/', 1)[0]
src = np.asarray(Image.open(f'{ROOT}/assets/hero-source.png').convert('RGBA')).astype(np.float32)
sky = np.asarray(Image.open(f'{ROOT}/assets/layers/skyline_mask.png')).astype(np.float32) / 255
a = (1 - sky) * src[..., 3] / 255
ys, xs = np.where(a > 0.004); x0, y0 = xs.min(), ys.min()
out = np.dstack([src[..., :3], a * 255])[y0:, x0:]
Image.fromarray(np.clip(out + .5, 0, 255).astype(np.uint8)).save(f'{ROOT}/assets/layers/nonsky.png')
print('nonsky', x0, y0, out.shape[1], out.shape[0])
