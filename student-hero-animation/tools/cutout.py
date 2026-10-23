"""Step 1: segment the boy from assets/hero-source.png with two rembg models (run directly with onnxruntime,
because pip installs are blocked here), refine the edge with a guided filter, and save the raw soft mask.
usage: python3 tools/cutout.py <models_dir>"""
import sys, numpy as np, cv2, onnxruntime as ort
from PIL import Image
ROOT = __file__.rsplit('/tools/', 1)[0]
MODELS = sys.argv[1]
src = Image.open(f'{ROOT}/assets/hero-source.png').convert('RGBA')
W, H = src.size
rgb = np.asarray(Image.alpha_composite(Image.new('RGBA', src.size, (255, 255, 255, 255)), src).convert('RGB')).astype(np.float32) / 255

def run(model, size, mean, std):
    s = ort.InferenceSession(f'{MODELS}/{model}.onnx', providers=['CPUExecutionProvider'])
    x = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = (x - np.array(mean)) / np.array(std)
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    y = s.run(None, {s.get_inputs()[0].name: x})[0][0, 0]
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    return cv2.resize(y, (W, H), interpolation=cv2.INTER_LINEAR)

m_isnet = run('isnet-general-use', 1024, (0.485, 0.456, 0.406), (1.0, 1.0, 1.0))
m_human = run('u2net_human_seg', 320, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
np.save('/tmp/claude-0/-home-claude/16f7c1b2-c4b4-5aea-9d02-af57212e0f40/scratchpad/m_isnet.npy', m_isnet)
np.save('/tmp/claude-0/-home-claude/16f7c1b2-c4b4-5aea-9d02-af57212e0f40/scratchpad/m_human.npy', m_human)
vis = np.concatenate([m_isnet, m_human], 1)
Image.fromarray((vis * 255).astype(np.uint8)).resize((W, H // 2)).save('/tmp/claude-0/-home-claude/16f7c1b2-c4b4-5aea-9d02-af57212e0f40/scratchpad/masks.png')
print('done', m_isnet.mean(), m_human.mean())
