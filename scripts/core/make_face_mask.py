# scripts/make_face_mask.py
# --- Phase-1 Determinism Guard (tiny, safe) ---
import os, random
import numpy as np

# Read from env if set; default to 77 (your project seed)
SEED = int(os.environ.get("NST_SEED", "77"))

random.seed(SEED)
np.random.seed(SEED)

try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch >=1.12: opt into deterministic algorithms (warn_only avoids hard errors)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
except Exception:
    pass

# For some CUDA BLAS paths; harmless on CPU
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# --- end determinism guard ---

import cv2, argparse
from PIL import Image, ImageDraw
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--content", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--expand", type=float, default=1.25, help="ellipse scale around detected face")
args = ap.parse_args()

img = cv2.imread(args.content)
if img is None:
    raise SystemExit(f"Could not read {args.content}")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# OpenCV built-in frontal face model
clf = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

h, w = gray.shape[:2]
mask = Image.new("L", (w,h), 0)
draw = ImageDraw.Draw(mask)

if len(faces) == 0:
    # Fallback: center ellipse
    cx, cy = w//2, h//2
    rx, ry = int(w*0.22), int(h*0.3)
else:
    # Largest face
    x,y,fw,fh = max(faces, key=lambda b: b[2]*b[3])
    cx, cy = x + fw//2, y + fh//2
    rx, ry = int(fw*0.6), int(fh*0.8)

rx = int(rx*args.expand); ry = int(ry*args.expand)
draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=255)

mask.save(args.out)
print("Saved mask:", args.out)
