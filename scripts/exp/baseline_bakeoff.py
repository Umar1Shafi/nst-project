#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline bake-off: Gatys (NST) vs AdaIN vs WCT
- Picks 3 contents × 2 styles (configurable)
- Collects images from:
    Gatys   = out/phase2/finals
    AdaIN   = breadth/adain_vgg16
    WCT     = breadth/wct_vgg16
- Computes metrics per image: Colorfulness, Edge IoU vs content, optional LPIPS(content, output)
- Writes:
    report/baselines/bakeoff_metrics.csv
    report/baselines/bakeoff_metrics.md
    report/baselines/adain_wct_grid.png  (3 rows = methods; 6 cols = content×style pairs)
"""

import argparse, os, re, glob, math, csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Optional LPIPS
try:
    import torch
    import lpips
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False

# --------- metrics ----------
def colorfulness_hs(img_rgb_uint8):
    # Hasler & Süsstrunk colorfulness metric
    img = img_rgb_uint8.astype(np.float32)
    R, G, B = img[...,0], img[...,1], img[...,2]
    rg  = np.abs(R - G)
    yb  = np.abs(0.5*(R + G) - B)
    std_rg, mean_rg = rg.std(), rg.mean()
    std_yb, mean_yb = yb.std(), yb.mean()
    return math.sqrt(std_rg**2 + std_yb**2) + 0.3*math.sqrt(mean_rg**2 + mean_yb**2)

def edge_mask_uint8(img_rgb_uint8):
    # Canny on luminance; tuned for robustness
    gray = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160, L2gradient=True)
    return (edges > 0).astype(np.uint8)

def iou(maskA, maskB):
    inter = (maskA & maskB).sum()
    union = (maskA | maskB).sum()
    return float(inter)/float(union+1e-6)

def lpips_distance(content_rgb_uint8, output_rgb_uint8):
    if not _HAS_LPIPS:
        return None
    # LPIPS expects normalized tensors [-1,1]
    with torch.no_grad():
        to_t = lambda x: torch.from_numpy(x.astype(np.float32)/127.5 - 1.0).permute(2,0,1).unsqueeze(0)
        x = to_t(content_rgb_uint8).contiguous()
        y = to_t(output_rgb_uint8).contiguous()
        loss_fn = lpips.LPIPS(net='alex').eval()
        d = loss_fn(x, y).item()
        return float(d)

# --------- helpers ----------
def load_rgb(path, size=None):
    im = Image.open(path).convert("RGB")
    if size is not None:
        im = im.resize((size, size), Image.Resampling.LANCZOS)
    return im

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_best_image(root, content_key, style_key):
    """
    Try to find an image in 'root' whose filename contains both content_key and style_key (case-insensitive).
    Returns best match path or None.
    """
    root = Path(root)
    pats = ["*.jpg", "*.png", "*.jpeg", "*.webp"]
    candidates = []
    ck = content_key.lower()
    sk = style_key.lower()
    for pat in pats:
        for p in root.rglob(pat):
            name = p.name.lower()
            if ck in name and sk in name:
                candidates.append(p)
    if not candidates:
        return None
    # Prefer files with '_final' (Gatys finals), else first
    finals = [p for p in candidates if "_final" in p.stem.lower()]
    return (finals[0] if finals else candidates[0])

def annotate(img_pil, text, pad=6, fill=(255, 255, 255)):
    """
    Draw a small black label box + white text in the top-left corner.
    Works across Pillow versions (handles single/multi-line).
    """
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    def _measure(draw, text, font):
        # Prefer modern bbox APIs (multi-line safe), then fall back.
        try:
            # Pillow >= 8: multiline_textbbox
            bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            try:
                # Older: textbbox
                bbox = draw.textbbox((0, 0), text, font=font)
                return bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                # Last resort: sum line heights from font.getsize or rough estimate
                widths, heights = [], []
                lines = text.splitlines() or [""]
                for line in lines:
                    try:
                        w, h = font.getsize(line) if font else (len(line) * 7, 12)
                    except Exception:
                        w, h = len(line) * 7, 12
                    widths.append(w); heights.append(h)
                width = max(widths) if widths else 0
                height = sum(heights) + max(0, len(lines) - 1) * 2
                return width, height

    w, h = _measure(draw, text, font)
    box = [pad - 3, pad - 3, pad + w + 3, pad + h + 3]
    draw.rectangle(box, fill=(0, 0, 0))
    try:
        if "\n" in text:
            draw.multiline_text((pad, pad), text, fill=fill, font=font, spacing=2)
        else:
            draw.text((pad, pad), text, fill=fill, font=font)
    except Exception:
        # If drawing fails for any reason, just return the image unlabelled
        pass
    return img_pil


# --------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gatys_root", default="out/phase2/finals")
    ap.add_argument("--adain_root", default="breadth/adain_vgg16")
    ap.add_argument("--wct_root",   default="breadth/wct_vgg16")
    ap.add_argument("--content_dir", default="data/content")
    ap.add_argument("--styles", default="Monet,Ukiyo-e_print")
    ap.add_argument("--contents", default="portrait2,city,Still_Life")
    ap.add_argument("--square", type=int, default=384, help="Resize display to NxN for fair visual grid")
    ap.add_argument("--out_dir", default="report/baselines")
    ap.add_argument("--lpips", type=int, default=0, help="1 to compute LPIPS(content, output)")
    args = ap.parse_args()

    ensure_dir(Path(args.out_dir))
    contents = [c.strip() for c in args.contents.split(",") if c.strip()]
    styles   = [s.strip() for s in args.styles.split(",") if s.strip()]

    # Gather content paths (for Edge/LPIPS reference)
    content_paths = { Path(c).stem: Path(args.content_dir)/f"{c}.jpg" if not c.lower().endswith(('.jpg','.png','.jpeg')) else Path(args.content_dir)/c
                     for c in contents }
    for cname, p in content_paths.items():
        if not p.exists():
            sys.exit(f"[missing] content file: {p}")

    # Make selection list: 6 pairs
    pairs = [(c, s) for c in contents for s in styles]

    methods = [
        ("Gatys", args.gatys_root),
        ("AdaIN", args.adain_root),
        ("WCT",   args.wct_root),
    ]

    grid_imgs = []  # rows
    metrics_rows = []

    for method_name, root in methods:
        row_tiles = []
        for (cname, sname) in pairs:
            p = find_best_image(root, cname, sname)
            if p is None:
                # Keep a placeholder tile with a warning
                tile = Image.new("RGB", (args.square, args.square), (30,30,30))
                annotate(tile, f"{method_name}\n{cname}×{sname}\n[missing]")
                row_tiles.append(tile)
                metrics_rows.append({
                    "pair": f"{cname}__{sname}", "method": method_name,
                    "colorfulness": None, "edge_iou": None, "lpips": None,
                    "path": None
                })
                continue

            out_pil = load_rgb(p, size=args.square)
            # metrics
            content_im = load_rgb(content_paths[cname], size=args.square)
            content_np = np.array(content_im)
            out_np     = np.array(out_pil)

            colr = colorfulness_hs(out_np)
            edge_i = iou(edge_mask_uint8(out_np), edge_mask_uint8(content_np))
            lp = lpips_distance(content_np, out_np) if (args.lpips and _HAS_LPIPS) else None

            tile = annotate(out_pil.copy(), f"{method_name} • {cname}×{sname}")
            row_tiles.append(tile)
            metrics_rows.append({
                "pair": f"{cname}__{sname}", "method": method_name,
                "colorfulness": round(colr, 3),
                "edge_iou": round(edge_i, 3),
                "lpips": (None if lp is None else round(lp, 4)),
                "path": str(p)
            })

        # concat tiles horizontally
        row = Image.new("RGB", (args.square*len(pairs), args.square), (0,0,0))
        for i, tile in enumerate(row_tiles):
            row.paste(tile, (i*args.square, 0))
        grid_imgs.append(row)

    # stack rows (methods)
    grid = Image.new("RGB", (args.square*len(pairs), args.square*len(methods)), (0,0,0))
    for r, row in enumerate(grid_imgs):
        grid.paste(row, (0, r*args.square))

    out_grid = Path(args.out_dir) / "adain_wct_grid.png"
    grid.save(out_grid)

    # CSV
    out_csv = Path(args.out_dir) / "bakeoff_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair","method","colorfulness","edge_iou","lpips","path"])
        w.writeheader()
        w.writerows(metrics_rows)

    # per-method means
    by_method = {}
    for r in metrics_rows:
        m = r["method"]
        by_method.setdefault(m, {"colorfulness": [], "edge_iou": [], "lpips": []})
        if r["colorfulness"] is not None: by_method[m]["colorfulness"].append(r["colorfulness"])
        if r["edge_iou"]     is not None: by_method[m]["edge_iou"].append(r["edge_iou"])
        if r["lpips"]        is not None and r["lpips"] != "": by_method[m]["lpips"].append(r["lpips"])

    out_md = Path(args.out_dir) / "bakeoff_metrics.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| Method | Mean Colorfulness | Mean Edge IoU | Mean LPIPS (content, out) |\n")
        f.write("|---|---:|---:|---:|\n")
        for m in ["Gatys","AdaIN","WCT"]:
            vals = by_method.get(m, {})
            mc = np.mean(vals.get("colorfulness", [np.nan]))
            me = np.mean(vals.get("edge_iou", [np.nan]))
            ml = np.mean(vals.get("lpips", [np.nan])) if (args.lpips and _HAS_LPIPS) else float('nan')
            f.write(f"| {m} | {mc:.3f} | {me:.3f} | {ml if not math.isnan(ml) else '—'} |\n")

    print(f"[ok] Grid:   {out_grid}")
    print(f"[ok] CSV:    {out_csv}")
    print(f"[ok] Table:  {out_md}")
    if args.lpips and not _HAS_LPIPS:
        print("[note] LPIPS requested but package not installed. Run: pip install lpips torch torchvision")

if __name__ == "__main__":
    main()
