#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AdaIN/WCT baseline evidence (Point 3)
- Input folders:
    breadth/gatys_vgg19/*
    breadth/adain_vgg16/*
    breadth/wct_vgg16/*
  (filenames just need to CONTAIN both the content stem and style stem)
- Picks 3 contents × 2 styles (configurable), and for each (c,s) finds:
    Gatys, AdaIN, WCT images.
- Computes simple metrics:
    * Colorfulness (Hasler–Süsstrunk)
    * Edge-IoU vs CONTENT (Canny IoU)
    * (Optional) LPIPS vs CONTENT (lower = closer to content)
- Outputs:
    * 6×3 composite grid PNG
    * CSV (per-row metrics)
    * Markdown mini-table (method means)
"""

import argparse, math, os
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import cv2

# ---------- metrics ----------
def colorfulness_hasler(img_rgb_uint8):
    R = img_rgb_uint8[...,0].astype(np.float32)
    G = img_rgb_uint8[...,1].astype(np.float32)
    B = img_rgb_uint8[...,2].astype(np.float32)
    rg = R - G
    yb = 0.5*(R + G) - B
    std_rg, mean_rg = rg.std(), np.abs(rg).mean()
    std_yb, mean_yb = yb.std(), np.abs(yb).mean()
    return math.sqrt(std_rg**2 + std_yb**2) + 0.3*math.sqrt(mean_rg**2 + mean_yb**2)

def edge_iou_vs_content(stylized_u8, content_u8):
    # unify size to content (stylized -> content size)
    H, W = content_u8.shape[:2]
    st = cv2.resize(stylized_u8, (W, H), interpolation=cv2.INTER_AREA)
    # gray + Canny
    g_c = cv2.cvtColor(content_u8,  cv2.COLOR_RGB2GRAY)
    g_s = cv2.cvtColor(st,          cv2.COLOR_RGB2GRAY)
    e_c = cv2.Canny(g_c, 100, 200)
    e_s = cv2.Canny(g_s, 100, 200)
    inter = np.logical_and(e_c>0, e_s>0).sum()
    union = np.logical_or(e_c>0,  e_s>0).sum()
    return float(inter) / float(max(union, 1))

def lpips_vs_content(stylized_u8, content_u8, device="cpu"):
    try:
        import torch
        import lpips
        loss_fn = lpips.LPIPS(net='alex').to(device).eval()
        def to_tensor(im):
            # im uint8 RGB -> [-1,1] CHW
            t = torch.from_numpy(im.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
            t = t*2.0 - 1.0
            return t.to(device)
        H, W = content_u8.shape[:2]
        st = cv2.resize(stylized_u8, (W, H), interpolation=cv2.INTER_AREA)
        d = loss_fn(to_tensor(st), to_tensor(content_u8))
        return float(d.detach().cpu().numpy().squeeze())
    except Exception:
        return None

# ---------- IO helpers ----------
def load_rgb_u8(p):
    return np.array(Image.open(p).convert("RGB"))

def find_best_match(folder: Path, content_stem: str, style_stem: str):
    # Accepts any filename that contains BOTH tokens (case-insensitive), any common image ext
    if not folder or not folder.exists():
        return None
    tokens = (content_stem.lower(), style_stem.lower())
    best = None
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        for f in folder.glob(ext):
            name = f.stem.lower()
            if all(t in name for t in tokens):
                # prefer shorter names (tighter match)
                if (best is None) or (len(f.name) < len(best.name)):
                    best = f
    return best

def draw_grid(rows, cols, cell_size, images, col_titles, row_titles, out_path):
    Wc, Hc = cell_size
    W = cols*Wc + 220  # left label gutter
    H = rows*Hc + 80   # top header gutter
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    # basic font (system fallback)
    try:
        font_h = ImageFont.truetype("arial.ttf", 22)
        font_r = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font_h = ImageFont.load_default()
        font_r = ImageFont.load_default()

    # headers (col titles)
    for j, title in enumerate(col_titles):
        x = 220 + j*Wc + Wc//2
        draw.text((x, 20), title, fill=(240,240,240), anchor="mm", font=font_h)

    # rows
    idx = 0
    for i in range(rows):
        # row label
        draw.text((10, 80 + i*Hc + Hc//2), row_titles[i], fill=(230,230,230), anchor="lm", font=font_r)
        for j in range(cols):
            im = images[idx]
            idx += 1
            if im is None:
                tile = Image.new("RGB", cell_size, (60,60,60))
            else:
                tile = ImageOps.fit(Image.fromarray(im), cell_size, method=Image.Resampling.LANCZOS, centering=(0.5,0.5))
            canvas.paste(tile, (220 + j*Wc, 80 + i*Hc))
    canvas.save(out_path)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gatys_dir", default="breadth/gatys_vgg19")
    ap.add_argument("--adain_dir", default="breadth/adain_vgg16")
    ap.add_argument("--wct_dir",   default="breadth/wct_vgg16")
    ap.add_argument("--contents",  default="portrait2,city,Still_Life", help="comma separated content stems (no extension)")
    ap.add_argument("--styles",    default="Monet,Ukiyo-e_print",       help="comma separated style stems (no extension)")
    ap.add_argument("--out_grid",  default="report/breadth/adain_wct_grid.png")
    ap.add_argument("--out_csv",   default="report/breadth/adain_wct_metrics.csv")
    ap.add_argument("--out_md",    default="report/breadth/adain_wct_table.md")
    ap.add_argument("--lpips",     type=int, default=1, help="1=compute LPIPS vs content (optional)")
    ap.add_argument("--device",    default="cpu", help="for LPIPS (cpu/cuda)")
    ap.add_argument("--cell",      default="320x320", help="cell WxH for the grid")
    args = ap.parse_args()

    cont_stems = [x.strip() for x in args.contents.split(",") if x.strip()]
    style_stems= [x.strip() for x in args.styles.split(",") if x.strip()]
    pairs = [(c,s) for c in cont_stems for s in style_stems]  # 3×2 = 6

    gatys_dir = Path(args.gatys_dir)
    adain_dir = Path(args.adain_dir)
    wct_dir   = Path(args.wct_dir)

    Wc, Hc = [int(x) for x in args.cell.lower().split("x")]

    rows = []
    grid_images = []
    csv_rows = [ "content,style,method,colorfulness,colorfulness_delta,edge_iou,lpips_content\n" ]

    for (c,s) in pairs:
        # find files
        g = find_best_match(gatys_dir, c, s)
        a = find_best_match(adain_dir, c, s)
        w = find_best_match(wct_dir,   c, s)
        # we also need the content image to compute deltas/Iou/LPIPS
        content_path_guess = Path("data/content")/f"{c}.jpg"
        if not content_path_guess.exists():
            # try png
            content_path_guess = Path("data/content")/f"{c}.png"
        if not content_path_guess.exists():
            raise SystemExit(f"[missing] content image for stem '{c}' not found under data/content/*.jpg|*.png")

        content_u8 = load_rgb_u8(content_path_guess)

        row_title = f"{c}  |  {s}"
        rows.append(row_title)
        triplet = []
        for method_name, fp in [("Gatys", g), ("AdaIN", a), ("WCT", w)]:
            if fp is None:
                triplet.append(None)
                csv_rows.append(f"{c},{s},{method_name},,,,\n")
                continue
            im = load_rgb_u8(fp)
            triplet.append(im)

            cf   = colorfulness_hasler(im)
            cf_c = colorfulness_hasler(content_u8)
            dcf  = cf - cf_c
            eiou = edge_iou_vs_content(im, content_u8)
            lp   = lpips_vs_content(im, content_u8, device=args.device) if args.lpips else None
            csv_rows.append(f"{c},{s},{method_name},{cf:.4f},{dcf:.4f},{eiou:.4f},{'' if lp is None else f'{lp:.4f}'}\n")
        grid_images.extend(triplet)

    # write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("".join(csv_rows), encoding="utf-8")

    # write MD mini-table (method means)
    import pandas as pd
    import io
    df = pd.read_csv(io.StringIO("".join(csv_rows)))
    md_lines = ["| Method | Colorfulness ↑ | ΔColorfulness ↑ | Edge IoU (vs content) ↑ | LPIPS vs content ↓ |",
                "|:--|--:|--:|--:|--:|"]
    for method in ["Gatys","AdaIN","WCT"]:
        sub = df[df["method"]==method]
        if len(sub)==0: continue
        cf   = sub["colorfulness"].mean()
        dcf  = sub["colorfulness_delta"].mean()
        eiou = sub["edge_iou"].mean()
        if "lpips_content" in sub and sub["lpips_content"].notna().any():
            lp = sub["lpips_content"].mean()
            md_lines.append(f"| {method} | {cf:.3f} | {dcf:.3f} | {eiou:.3f} | {lp:.3f} |")
        else:
            md_lines.append(f"| {method} | {cf:.3f} | {dcf:.3f} | {eiou:.3f} | — |")
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    # draw 6×3 grid
    out_grid = Path(args.out_grid)
    out_grid.parent.mkdir(parents=True, exist_ok=True)
    draw_grid(rows=len(pairs), cols=3, cell_size=(Wc,Hc),
              images=grid_images,
              col_titles=["Gatys (VGG19)","AdaIN (VGG16)","WCT (VGG16)"],
              row_titles=rows,
              out_path=out_grid)
    print(f"[ok] Wrote:\n - {out_grid}\n - {out_csv}\n - {out_md}")

if __name__ == "__main__":
    main()
