#!/usr/bin/env python3
"""
Evaluate NST core vs finisher by pairing images from:
  - RAW:    phase2/imgs/
  - FINISH: phase2/finals/

Pairs are matched by a canonical stem derived from filenames:
  e.g., portrait2__Monet__sw20000_tv0.0028_edge0.10_fp0.75_cpm0.6_layers45

Outputs
-------
- CSV with per-stem metrics:
    stem, content, style, params,
    raw_path, fin_path, content_path,
    edge_iou_raw (vs content edges), edge_iou_fin (vs content edges), delta_edge_iou,
    edge_density_raw, edge_density_fin, delta_edge_density,
    color_raw, color_fin, delta_color,
    lpips_raw_fin (optional)

- Optional plots (two figures):
    - mean ΔEdge-IoU by (content, style)
    - mean ΔColorfulness by (content, style)

Usage
-----
# Auto-detect under a phase2 root:
python scripts/eval/eval_pairs_imgs_vs_finals.py --root out/phase2 \
  --csv report/pairs/pairs_delta_metrics.csv --plots report/pairs/plots --lpips 0

# Or specify raw/finals explicitly:
python scripts/eval/eval_pairs_imgs_vs_finals.py \
  --raw out/phase2/imgs --fin out/phase2/finals \
  --csv report/pairs/pairs_delta_metrics.csv --plots report/pairs/plots

# Provide content images (optional) to compute IoU vs content edges:
python scripts/eval/eval_pairs_imgs_vs_finals.py --root out/phase2 \
  --content_dir data/content --lpips 1
"""

import argparse, csv, os, re, math
from pathlib import Path

from PIL import Image
import numpy as np

# Optional OpenCV (faster edges); fallback uses numpy.gradient (no scipy needed)
try:
    import cv2
except Exception:
    cv2 = None

# Optional LPIPS
def _lpips_model_or_none(enable: bool):
    if not enable:
        return None
    try:
        import torch
        import lpips
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = lpips.LPIPS(net='vgg').to(dev)
        model.eval()
        return (model, dev)
    except Exception as e:
        print("[warn] LPIPS not available:", e)
        return None

def imread_rgb(path: Path):
    im = Image.open(path).convert("RGB")
    return np.array(im)

def to_gray(img_rgb: np.ndarray):
    r, g, b = img_rgb[...,0].astype(np.float32), img_rgb[...,1].astype(np.float32), img_rgb[...,2].astype(np.float32)
    gray = 0.299*r + 0.587*g + 0.114*b
    return gray / 255.0

def edge_map(gray: np.ndarray, size_hint: int = None):
    """
    Returns a boolean edge map. Uses Canny if cv2 is present, else numpy.gradient magnitude
    with a percentile threshold.
    """
    g = gray
    if size_hint is not None and max(g.shape) > size_hint:
        scale = size_hint / float(max(g.shape))
        new_h = int(round(g.shape[0]*scale))
        new_w = int(round(g.shape[1]*scale))
        g = np.array(Image.fromarray((g*255).astype(np.uint8)).resize((new_w, new_h), Image.BILINEAR)) / 255.0

    if cv2 is not None:
        u8 = (np.clip(g,0,1)*255).astype(np.uint8)
        v = np.median(u8)
        t1 = max(0, 0.66*v)
        t2 = min(255, 1.33*v + 32)
        e = cv2.Canny(u8, t1, t2) > 0
        return e

    # numpy fallback: gradient magnitude + percentile threshold
    gy, gx = np.gradient(g)
    mag = np.sqrt(gx*gx + gy*gy)
    thr = np.percentile(mag, 90.0)
    e = mag >= max(thr, 1e-6)
    return e

def iou(a: np.ndarray, b: np.ndarray):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a,b).sum()
    union = np.logical_or(a,b).sum()
    if union == 0:
        return 0.0
    return float(inter)/float(union)

def edge_density(e: np.ndarray):
    total = e.size
    if total == 0: return 0.0
    return float(e.sum())/float(total)

def colorfulness(img_rgb: np.ndarray):
    # Hasler & Süsstrunk (2003) colorfulness metric
    R = img_rgb[...,0].astype(np.float32)
    G = img_rgb[...,1].astype(np.float32)
    B = img_rgb[...,2].astype(np.float32)
    rg = np.abs(R - G)
    yb = np.abs(0.5*(R + G) - B)
    mu_rg, sigma_rg = rg.mean(), rg.std(ddof=1)
    mu_yb, sigma_yb = yb.mean(), yb.std(ddof=1)
    return math.sqrt(sigma_rg**2 + sigma_yb**2) + 0.3*math.sqrt(mu_rg**2 + mu_yb**2)

def parse_stem(name: str):
    """
    From filename -> canonical stem (strip _final and size suffixes like _384/_768).
    """
    stem = name.rsplit('.',1)[0]
    if stem.endswith('_final'):
        stem = stem[:-6]
    m = re.search(r'_(\d{3,4})$', stem)
    if m:
        stem = stem[:-(len(m.group(0)))]
    return stem

def split_stem(stem: str):
    parts = stem.split('__')
    content = parts[0] if len(parts)>0 else ''
    style   = parts[1] if len(parts)>1 else ''
    params  = parts[2] if len(parts)>2 else ''
    return content, style, params

def find_content_image(content_dir: Path, content_name: str):
    if not content_dir: return None
    for ext in ['.jpg','.jpeg','.png','.webp','.bmp']:
        p = content_dir / f"{content_name}{ext}"
        if p.exists():
            return p
    return None

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None, help="phase2 root containing imgs/ and finals/")
    ap.add_argument("--raw", type=str, default=None, help="explicit path to phase2/imgs")
    ap.add_argument("--fin", type=str, default=None, help="explicit path to phase2/finals")
    ap.add_argument("--content_dir", type=str, default=None, help="optional folder with original content images")
    ap.add_argument("--csv", type=str, default="pairs_delta_metrics.csv")
    ap.add_argument("--plots", type=str, default=None, help="output directory for plots")
    ap.add_argument("--lpips", type=int, default=0, help="compute LPIPS(raw,finish) if 1 and library available")
    ap.add_argument("--max", type=int, default=None, help="max pairs to process (debug)")
    args = ap.parse_args()

    root = Path(args.root) if args.root else None
    raw_dir = Path(args.raw) if args.raw else (root / "imgs" if root else None)
    fin_dir = Path(args.fin) if args.fin else (root / "finals" if root else None)
    content_dir = Path(args.content_dir) if args.content_dir else None

    assert raw_dir and fin_dir, "Could not resolve raw and finals directories. Use --root or --raw/--fin."
    assert raw_dir.exists(), f"Raw dir not found: {raw_dir}"
    assert fin_dir.exists(), f"Finals dir not found: {fin_dir}"
    if content_dir and not content_dir.exists():
        print("[warn] content_dir does not exist:", content_dir)
        content_dir = None

    # Map stems -> paths
    raw_map, fin_map = {}, {}
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png",".webp",".bmp"):
            raw_map[parse_stem(p.name)] = p
    for p in fin_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png",".webp",".bmp"):
            fin_map[parse_stem(p.name)] = p

    stems = sorted(set(raw_map.keys()) & set(fin_map.keys()))
    if args.max:
        stems = stems[:args.max]
    print(f"[info] paired stems: {len(stems)}")

    lpips_state = _lpips_model_or_none(bool(args.lpips))

    rows = []
    for i, stem in enumerate(stems, 1):
        raw_p = raw_map[stem]
        fin_p = fin_map[stem]

        content, style, params = split_stem(stem)
        content_p = find_content_image(content_dir, content) if content_dir else None

        # Load images
        raw_rgb = imread_rgb(raw_p)
        fin_rgb = imread_rgb(fin_p)

        # Resize fin to raw size if needed
        if fin_rgb.shape[:2] != raw_rgb.shape[:2]:
            fin_rgb = np.array(Image.fromarray(fin_rgb).resize((raw_rgb.shape[1], raw_rgb.shape[0]), Image.BILINEAR))

        # Metrics: colorfulness
        c_raw = colorfulness(raw_rgb)
        c_fin = colorfulness(fin_rgb)

        # Edge density
        g_raw = to_gray(raw_rgb)
        g_fin = to_gray(fin_rgb)
        e_raw = edge_map(g_raw, size_hint=768)
        e_fin = edge_map(g_fin, size_hint=768)
        ed_raw = edge_density(e_raw)
        ed_fin = edge_density(e_fin)

        # Edge IoU vs content edges (if content image is available)
        ei_raw = np.nan
        ei_fin = np.nan
        if content_p is not None:
            c_rgb = imread_rgb(content_p)
            if c_rgb.shape[:2] != raw_rgb.shape[:2]:
                c_rgb = np.array(Image.fromarray(c_rgb).resize((raw_rgb.shape[1], raw_rgb.shape[0]), Image.BILINEAR))
            e_content = edge_map(to_gray(c_rgb), size_hint=768)
            if e_raw.shape != e_content.shape:
                e_raw = np.array(Image.fromarray((e_raw.astype(np.uint8)*255))).resize((e_content.shape[1], e_content.shape[0]))
                e_raw = np.array(e_raw) > 0
            if e_fin.shape != e_content.shape:
                e_fin = np.array(Image.fromarray((e_fin.astype(np.uint8)*255))).resize((e_content.shape[1], e_content.shape[0]))
                e_fin = np.array(e_fin) > 0
            ei_raw = iou(e_raw, e_content)
            ei_fin = iou(e_fin, e_content)

        # LPIPS raw vs finish
        lpf = ""
        if lpips_state is not None:
            try:
                import torch
                model, dev = lpips_state
                def to_t(img):
                    arr = img.astype(np.float32)/255.0
                    arr = (arr*2.0 - 1.0).transpose(2,0,1)[None,...]
                    return torch.from_numpy(arr).to(dev)
                d = model(to_t(raw_rgb), to_t(fin_rgb))
                lpf = float(d.detach().cpu().numpy().squeeze())
            except Exception as e:
                lpf = ""
                print("[warn] LPIPS failure @", stem, ":", e)

        rows.append({
            "stem": stem,
            "content": content,
            "style": style,
            "params": params,
            "raw_path": str(raw_p),
            "fin_path": str(fin_p),
            "content_path": str(content_p) if content_p is not None else "",
            "edge_iou_raw": ei_raw,
            "edge_iou_fin": ei_fin,
            "delta_edge_iou": (ei_fin - ei_raw) if (not np.isnan(ei_raw) and not np.isnan(ei_fin)) else "",
            "edge_density_raw": ed_raw,
            "edge_density_fin": ed_fin,
            "delta_edge_density": ed_fin - ed_raw,
            "color_raw": c_raw,
            "color_fin": c_fin,
            "delta_color": c_fin - c_raw,
            "lpips_raw_fin": lpf
        })

        if i % 200 == 0:
            print(f"[info] processed {i}/{len(stems)}")

    # Write CSV
    csv_out = Path(args.csv)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else [
            "stem","content","style","params","raw_path","fin_path","content_path",
            "edge_iou_raw","edge_iou_fin","delta_edge_iou",
            "edge_density_raw","edge_density_fin","delta_edge_density",
            "color_raw","color_fin","delta_color","lpips_raw_fin"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("[ok] wrote CSV:", csv_out, "rows:", len(rows))

    # Plots (optional)
    if args.plots:
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            out_dir = Path(args.plots); out_dir.mkdir(parents=True, exist_ok=True)
            df = pd.read_csv(csv_out)

            def agg_mean(col):
                return df.groupby(["content","style"], dropna=False)[col].mean().reset_index()

            # Δ Edge-IoU (only if computed)
            if "delta_edge_iou" in df.columns and df["delta_edge_iou"].dtype != object:
                d1 = agg_mean("delta_edge_iou")
                plt.figure()
                labels = d1["content"] + " × " + d1["style"]
                plt.bar(labels, d1["delta_edge_iou"])
                plt.ylabel("Mean Δ Edge-IoU (finish − raw)")
                plt.xlabel("Content × Style")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(out_dir / "delta_edge_iou.png", dpi=160)
                plt.close()

            # Δ Colorfulness
            d2 = agg_mean("delta_color")
            plt.figure()
            labels = d2["content"] + " × " + d2["style"]
            plt.bar(labels, d2["delta_color"])
            plt.ylabel("Mean Δ Colorfulness (finish − raw)")
            plt.xlabel("Content × Style")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(out_dir / "delta_color.png", dpi=160)
            plt.close()

            print("[ok] plots saved to", out_dir)
        except Exception as e:
            print("[warn] plotting failed:", e)

if __name__ == "__main__":
    main()
