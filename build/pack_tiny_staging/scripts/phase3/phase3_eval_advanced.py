#!/usr/bin/env python
# Evaluate existing stylized images (no re-generation).
# Metrics: PSNR, SSIM, Edge-IoU, Colorfulness, Entropy; optional LPIPS & ArcFace.

import argparse, csv, json, math, os, sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from PIL import Image

# ---------- Optional deps (loaded lazily) ----------
def _try_imports():
    mods = {}
    try:
        from skimage.metrics import structural_similarity as ssim
        mods["ssim"] = ssim
    except Exception:
        mods["ssim"] = None
    try:
        import torch
        mods["torch"] = torch
    except Exception:
        mods["torch"] = None
    try:
        import lpips
        mods["lpips"] = lpips
    except Exception:
        mods["lpips"] = None
    try:
        import insightface
        from insightface.app import FaceAnalysis
        mods["insightface"] = insightface
        mods["FaceAnalysis"] = FaceAnalysis
    except Exception:
        mods["insightface"] = None
        mods["FaceAnalysis"] = None
    return mods

MODS = _try_imports()

# ---------- Small image utils ----------
def _resample():
    # Pillow API compatibility
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return getattr(Image, "LANCZOS", Image.BICUBIC)

def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def read_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return pil_to_np(im)

def resize_to(img: np.ndarray, size_xy: Tuple[int, int]) -> np.ndarray:
    W, H = size_xy
    return np.array(Image.fromarray(img).resize((W, H), _resample()))

def to_gray(img: np.ndarray) -> np.ndarray:
    # Luma transform
    return (0.2989*img[...,0] + 0.5870*img[...,1] + 0.1140*img[...,2]).astype(np.float32)

# ---------- Metrics ----------
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32))**2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim_metric(a_gray: np.ndarray, b_gray: np.ndarray) -> Optional[float]:
    ssim = MODS["ssim"]
    if ssim is None:
        return None
    try:
        return float(ssim(a_gray, b_gray, data_range=255.0))
    except Exception:
        return None

def sobel_edges(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray.astype(np.float32))
    mag = np.sqrt(gx*gx + gy*gy)
    thr = np.percentile(mag, 75.0)
    return (mag > thr).astype(np.uint8)

def edge_iou(e1: np.ndarray, e2: np.ndarray) -> float:
    inter = np.logical_and(e1 > 0, e2 > 0).sum()
    union = np.logical_or(e1 > 0, e2 > 0).sum()
    return float(inter / union) if union > 0 else 1.0

def colorfulness(img_rgb: np.ndarray) -> float:
    R = img_rgb[...,0].astype(np.float32)
    G = img_rgb[...,1].astype(np.float32)
    B = img_rgb[...,2].astype(np.float32)
    rg = R - G
    yb = 0.5*(R + G) - B
    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3*np.sqrt(mean_rg**2 + mean_yb**2))

def entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0,255), density=True)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())

def lpips_score(a_rgb: np.ndarray, b_rgb: np.ndarray) -> Optional[float]:
    torch = MODS["torch"]; lpips = MODS["lpips"]
    if torch is None or lpips is None:
        return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = getattr(lpips, "LPIPS")(net='alex').to(dev).eval()
    A = torch.tensor(a_rgb).permute(2,0,1).unsqueeze(0).float()/255.0*2-1
    B = torch.tensor(b_rgb).permute(2,0,1).unsqueeze(0).float()/255.0*2-1
    with torch.no_grad():
        v = net(A.to(dev), B.to(dev)).mean().item()
    return float(v)

_ARC_APP = None
def _load_arcface():
    global _ARC_APP
    if _ARC_APP is not None:
        return _ARC_APP
    FaceAnalysis = MODS["FaceAnalysis"]
    if FaceAnalysis is None:
        return None
    try:
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(256,256))
        _ARC_APP = app
        return app
    except Exception:
        return None

def arcface_cosine(a_rgb: np.ndarray, b_rgb: np.ndarray) -> Optional[float]:
    app = _load_arcface()
    if app is None:
        return None
    # insightface expects BGR
    a_bgr = a_rgb[..., ::-1].copy()
    b_bgr = b_rgb[..., ::-1].copy()
    fa = app.get(a_bgr); fb = app.get(b_bgr)
    if not fa or not fb:
        return None
    va = fa[0].normed_embedding; vb = fb[0].normed_embedding
    return float((va * vb).sum())

# ---------- Row building & sidecars ----------
def guess_subject_by_aspect(img_path: Path) -> str:
    try:
        with Image.open(img_path) as im:
            w, h = im.size
        return "portrait" if h >= w else "scene"
    except Exception:
        return "portrait"

def build_rows(outputs: List[Path], portrait_ref: Path, scene_ref: Path) -> List[Dict[str, Any]]:
    rows = []
    for out_path in outputs:
        if not out_path.exists():
            print(f"[warn] missing output: {out_path}")
            continue
        subject = guess_subject_by_aspect(out_path)
        in_path = portrait_ref if subject == "portrait" else scene_ref

        out_rgb = read_image(out_path)
        in_rgb  = read_image(in_path)
        in_rs   = resize_to(in_rgb, (out_rgb.shape[1], out_rgb.shape[0]))

        psnr_val = psnr(in_rs, out_rgb)
        ssim_val = ssim_metric(to_gray(in_rs), to_gray(out_rgb))
        e1 = sobel_edges(to_gray(in_rs)); e2 = sobel_edges(to_gray(out_rgb))
        edge_val = edge_iou(e1, e2)
        color_val = colorfulness(out_rgb)
        ent_val = entropy(to_gray(out_rgb))
        lpips_val = lpips_score(in_rs, out_rgb)
        arc_val = arcface_cosine(in_rs, out_rgb) if subject == "portrait" else None

        rows.append({
            "output": str(out_path),
            "input": str(in_path),
            "subject_guess": subject,
            "psnr": round(psnr_val, 3),
            "ssim": None if ssim_val is None else round(ssim_val, 4),
            "edge_iou": round(edge_val, 4),
            "colorfulness": round(color_val, 2),
            "entropy": round(ent_val, 3),
            "lpips": None if lpips_val is None else round(lpips_val, 4),
            "arcface_cos": None if arc_val is None else round(arc_val, 4),
        })
    return rows

def write_sidecars(rows: List[Dict[str, Any]], outdir: Path):
    side_dir = outdir / "sidecars"
    side_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        out_name = Path(r["output"]).name
        sc = side_dir / (Path(out_name).with_suffix(".json").name)
        payload = {
            "input": r["input"],
            "output": r["output"],
            "subject_guess": r["subject_guess"],
            "metrics": {
                "psnr": r["psnr"],
                "ssim": r["ssim"],
                "edge_iou": r["edge_iou"],
                "colorfulness": r["colorfulness"],
                "entropy": r["entropy"],
                "lpips": r["lpips"],
                "arcface_cos": r["arcface_cos"],
            },
            "env": {
                "python": sys.version.split()[0],
                "platform": os.name,
                "has_torch": MODS["torch"] is not None,
                "has_lpips": MODS["lpips"] is not None,
                "has_insightface": MODS["insightface"] is not None,
            },
            "notes": "Generated by phase3_eval_advanced.py (images-only path).",
        }
        sc.write_text(json.dumps(payload, indent=2), encoding="utf-8")

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser(description="Phase 3 advanced eval (images-only).")
    p.add_argument("--outputs-glob", required=True,
                   help="Glob for outputs (e.g. Extra/images/*.png or out/phase3/**/*.png)")
    p.add_argument("--portrait-input", required=True, help="Reference portrait input image")
    p.add_argument("--scene-input", required=True, help="Reference scene/street input image")
    p.add_argument("--outdir", default="report/phase3_eval_advanced", help="Output dir for CSV & sidecars")
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    outputs = sorted(Path().glob(args.outputs_glob))
    if not outputs:
        print(f"[error] No outputs matched: {args.outputs_glob}")
        sys.exit(2)

    portrait_ref = Path(args.portrait_input)
    scene_ref = Path(args.scene_input)
    if not portrait_ref.exists() or not scene_ref.exists():
        print("[error] One or both reference inputs do not exist.")
        sys.exit(3)

    rows = build_rows(outputs, portrait_ref, scene_ref)

    csv_path = outdir / "phase3_metrics_advanced.csv"
    cols = ["output","input","subject_guess","psnr","ssim","edge_iou",
            "colorfulness","entropy","lpips","arcface_cos"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})

    write_sidecars(rows, outdir)

    # Console summaries
    portraits = [r for r in rows if r["subject_guess"] == "portrait" and r["arcface_cos"] is not None]
    if portraits:
        vals = [r["arcface_cos"] for r in portraits]
        print(f"ArcFace cosine (portraits): mean={sum(vals)/len(vals):.3f} "
              f"range=({min(vals):.3f},{max(vals):.3f})")
    else:
        print("ArcFace: n/a (missing model or faces not detected)")

    lp = [r["lpips"] for r in rows if r["lpips"] is not None]
    if lp:
        print(f"LPIPS: mean={sum(lp)/len(lp):.3f} range=({min(lp):.3f},{max(lp):.3f})")
    else:
        print("LPIPS: n/a (missing torch/lpips)")

    print("[write] CSV:", csv_path)
    print("[write] sidecars:", outdir/"sidecars")

if __name__ == "__main__":
    main()
