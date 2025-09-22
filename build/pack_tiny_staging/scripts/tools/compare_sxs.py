# tools/compare_sxs.py
import sys, os
from PIL import Image, ImageDraw, ImageFont

def side_by_side(a, b, out, label_a="control", label_b="experiment", pad=16):
    A = Image.open(a).convert("RGB")
    B = Image.open(b).convert("RGB")
    h = max(A.height, B.height)
    # scale both to same height
    def scale_to_h(img, H):
        w = int(round(img.width * H / img.height))
        return img.resize((w, H), Image.Resampling.LANCZOS)
    A = scale_to_h(A, h)
    B = scale_to_h(B, h)
    # canvas with labels area
    label_h = 40
    W = A.width + B.width + pad*3
    H = h + label_h + pad*2
    canvas = Image.new("RGB", (W, H), (255,255,255))
    draw = ImageDraw.Draw(canvas)
    # paste
    x = pad
    y = pad + label_h
    canvas.paste(A, (x, y)); x += A.width + pad
    canvas.paste(B, (x, y))
    # labels
    draw.text((pad, pad+10), label_a, fill=(0,0,0))
    draw.text((A.width + pad*2, pad+10), label_b, fill=(0,0,0))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    canvas.save(out, "PNG")

if __name__ == "__main__":
    # usage: python tools/compare_sxs.py <control_img> <experiment_img> <out_png> [label_control] [label_experiment]
    a, b, out = sys.argv[1], sys.argv[2], sys.argv[3]
    la = sys.argv[4] if len(sys.argv) > 4 else "control"
    lb = sys.argv[5] if len(sys.argv) > 5 else "experiment"
    side_by_side(a, b, out, la, lb)
