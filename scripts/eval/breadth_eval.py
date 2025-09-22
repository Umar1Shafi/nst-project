# scripts/breadth_eval.py
"""
Breadth & evaluation:
- Runs stylization for all (content × style × method × backbone) combos.
- Methods: gatys (nst_hybrid_ready.py), adain (baseline_adain_loss.py), wct (baseline_wct_loss.py)
- Backbones: vgg19, vgg16
- Metrics per result:
    * ssim_face: SSIM(content, output) averaged over face ROI (if mask available)
    * ssim_full: SSIM(content, output) full image
    * edge_ratio_full: Canny edge density over whole image
    * edge_ratio_bg: Canny edge density over background (outside face ROI)
    * lpips_to_content: LPIPS(content, output) (lower is closer to content)
    * palette_strong: count of strong palette colors (top-k occupying ≥ min_frac each)
- Optional --skip_existing: do not re-run stylization if output already exists (still compute metrics)
- Auto face mask:
    * Use "--face_masks" as semicolon-separated list aligned with contents.
      - "auto" -> use (and cache) `CONTENTDIR/CONTENTNAME_face_mask.png`
      - "" or missing slot -> no mask for that content
      - explicit path -> use that path
"""
import argparse, os, sys, json, time, subprocess, re, csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import metrics, color, feature
from skimage.transform import resize

# Optional torch for VRAM probing
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# LPIPS (single instance)
try:
    import lpips
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False


# ---------- robust core-path resolution (works from scripts/ or scripts/eval/) ----------
def _find_core_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        cand = parent / "scripts" / "core"
        if cand.exists():
            return cand
    # Fallback: relative if running from repo root
    return (Path.cwd() / "scripts" / "core").resolve()

SCRIPTS_CORE = _find_core_dir()
PATHS = {
    "gatys":  SCRIPTS_CORE / "nst_hybrid_ready.py",
    "adain":  SCRIPTS_CORE / "baseline_adain_loss.py",
    "wct":    SCRIPTS_CORE / "baseline_wct_loss.py",
}
for k, p in PATHS.items():
    if not p.exists():
        raise SystemExit(f"[ERR] Missing baseline script for '{k}': {p}")


# --- subprocess helper (Windows-safe) ---
def run_cmd(args_list):
    """
    Run a subprocess with captured output; raise on failure.
    Returns wall-clock seconds.
    """
    t0 = time.perf_counter()
    cp = subprocess.run(args_list, capture_output=True, text=True)
    t1 = time.perf_counter()
    if cp.returncode != 0:
        print("\n--- Subprocess STDOUT ---\n", cp.stdout)
        print("\n--- Subprocess STDERR ---\n", cp.stderr)
        raise subprocess.CalledProcessError(cp.returncode, cp.args)
    return t1 - t0


# --- I/O + metrics helpers ---
def load_gray_01(path):
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return color.rgb2gray(arr)

def load_rgb_pil(path):
    return Image.open(path).convert("RGB")

def ssim01(a, b):
    return float(metrics.structural_similarity(a, b, data_range=1.0))

def ssim_masked(a, b, mask01):
    score, ssim_map = metrics.structural_similarity(a, b, data_range=1.0, full=True)
    if mask01 is None or mask01.sum() == 0:
        return float(score)
    return float((ssim_map * mask01).sum() / (mask01.sum() + 1e-6))

def edge_ratio_full(gray):
    edges = feature.canny(gray, sigma=1.2)
    return float(edges.mean())

def edge_ratio_bg(gray, mask01):
    edges = feature.canny(gray, sigma=1.2)
    if mask01 is None:
        return float(edges.mean())
    bg = (1.0 - mask01) > 0.5
    if bg.sum() == 0:
        return float(edges.mean())
    return float(edges[bg].mean())

def load_mask01(path, target_hw):
    m = Image.open(path).convert("L")
    m = np.asarray(m, dtype=np.float32) / 255.0
    m = resize(m, target_hw, mode="reflect", anti_aliasing=True)
    return (m > 0.5).astype(np.float32)

def palette_counts_pil(pil_img, k=8, min_frac=0.05):
    small = pil_img.resize((256, 256), Image.Resampling.LANCZOS)
    pal   = small.convert("P", palette=Image.ADAPTIVE, colors=k)
    counts = np.array(pal.getcolors(256*256), dtype=np.int64)[:, 0]
    fracs  = counts / counts.sum()
    return int((fracs >= min_frac).sum())


# --- Auto face mask (opencv) ---
def auto_mask_path_for_content(content_path):
    p = Path(content_path)
    return str(p.with_name(p.stem + "_face_mask.png"))

def ensure_auto_face_mask(content_path, out_mask_path):
    """
    If out_mask_path exists -> return it.
    Otherwise, try to detect face (largest) and save a soft ellipse mask.
    Returns out_mask_path if created, else None.
    """
    if os.path.exists(out_mask_path):
        print(f"[mask] loaded: {out_mask_path}")
        return out_mask_path

    try:
        import cv2
    except Exception:
        print("[mask] OpenCV not available; cannot auto-generate mask.")
        return None

    pil = load_rgb_pil(content_path)
    # center-crop square to match stylization policy
    w, h = pil.size
    side = min(w, h)
    left, top = (w - side)//2, (h - side)//2
    sq = pil.crop((left, top, left+side, top+side))
    arr = np.asarray(sq.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64,64))
    if len(faces) == 0:
        print(f"[mask] no face detected for {content_path}")
        return None

    x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
    mask = np.zeros((sq.height, sq.width), dtype=np.uint8)
    cx, cy = x + w//2, y + h//2
    ax, ay = int(w * 0.62), int(h * 0.82)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, thickness=-1)
    k = max(3, (min(mask.shape)//128)|1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)

    m_pil = Image.fromarray(mask, mode="L")
    os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)
    m_pil.save(out_mask_path)
    print(f"[mask] saved: {out_mask_path}")
    return out_mask_path


# --- LPIPS singleton ---
class LPIPSHelper:
    def __init__(self, device_str):
        self.dev = device_str
        self.model = None
        if HAS_LPIPS:
            try:
                self.model = lpips.LPIPS(net='alex').to(self.dev).eval()
            except Exception as e:
                print(f"[lpips] init failed: {e}")
                self.model = None

    def to_tensor_m11(self, pil_img):
        # returns 1x3xH xW in [-1,1] on self.dev
        arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)  # 1x3xHxW
        t = t * 2.0 - 1.0
        return t.to(self.dev)

    def lpips(self, pil_a, pil_b):
        if (not HAS_LPIPS) or (self.model is None):
            return ""
        # ensure same size
        if pil_a.size != pil_b.size:
            pil_a = pil_a.resize(pil_b.size, Image.Resampling.LANCZOS)
        ta = self.to_tensor_m11(pil_a)
        tb = self.to_tensor_m11(pil_b)
        with torch.no_grad():
            v = self.model(ta, tb).item()
        return float(v)


# --- name utils ---
def safe_name(p):
    p = Path(p)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", p.stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contents", required=True,
                    help='Semicolon-separated list of content images.')
    ap.add_argument("--styles", required=True,
                    help='Semicolon-separated list of style images.')
    ap.add_argument("--face_masks", default="",
                    help='Semicolon-separated list aligned with contents. "auto" or explicit path or empty.')
    ap.add_argument("--methods", default="gatys,adain,wct",
                    help='Comma-separated methods among: gatys,adain,wct')
    ap.add_argument("--backbones", default="vgg19,vgg16",
                    help='Comma-separated backbones: vgg19,vgg16')
    ap.add_argument("--sizes", default="384,768")
    ap.add_argument("--steps", default="250,350")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_root", default="out/breadth")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip stylization if output already exists.")
    ap.add_argument("--palette_k", type=int, default=8)
    ap.add_argument("--palette_min_frac", type=float, default=0.05)

    args = ap.parse_args()
    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    contents = [c for c in args.contents.split(";") if c]
    styles   = [s for s in args.styles.split(";") if s]
    masks_in = [m for m in args.face_masks.split(";")] if args.face_masks else []
    while len(masks_in) < len(contents):
        masks_in.append("")  # pad

    methods   = [m.strip() for m in args.methods.split(",") if m.strip()]
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]

    # Prepare LPIPS singleton
    dev = ("cuda" if (HAS_TORCH and torch.cuda.is_available() and args.device != "cpu") else "cpu")
    lp = LPIPSHelper(dev)

    rows = []

    # Pre-ensure auto masks for all contents (so metrics can use them)
    resolved_masks = []
    for ci, cpath in enumerate(contents):
        entry = masks_in[ci].strip().lower()
        if entry == "auto":
            guessed = auto_mask_path_for_content(cpath)
            got = ensure_auto_face_mask(cpath, guessed)
            resolved_masks.append(got)
        elif entry == "":
            resolved_masks.append(None)
        else:
            resolved_masks.append(entry if os.path.exists(entry) else None)

    # Iterate all combos
    for c_idx, cpath in enumerate(contents):
        c_name = safe_name(cpath)
        c_pil_full = load_rgb_pil(cpath)
        c_gray_full = None  # lazy

        for s_idx, spath in enumerate(styles):
            s_name = safe_name(spath)

            for meth in methods:
                for bb in backbones:
                    out_dir = Path(args.out_root) / f"{meth}_{bb}" / f"{c_name}__{s_name}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_img = out_dir / "out.jpg"

                    # Build child command with absolute script paths
                    if meth == "gatys":
                        script = str(PATHS["gatys"])
                        extra  = ["--style_layers","conv4_1,conv5_1",
                                  "--content_layer","conv4_2",
                                  "--steps", args.steps]
                    elif meth == "adain":
                        script = str(PATHS["adain"])
                        extra  = ["--content_layer","conv4_2",
                                  "--style_layers","conv4_1,conv5_1",
                                  "--steps", args.steps]
                    elif meth == "wct":
                        script = str(PATHS["wct"])
                        extra  = ["--layer","conv4_1",
                                  "--steps", args.steps]
                    else:
                        print(f"[warn] unknown method '{meth}', skipping.")
                        continue

                    cmd = [sys.executable, script,
                           "--backbone", bb,
                           "--content", cpath,
                           "--style", spath,
                           "--out", str(out_img),
                           "--sizes", args.sizes,
                           "--seed", str(args.seed),
                           "--device", args.device] + extra

                    # Pass face mask only to GATYS (it supports --face_mask)
                    if meth == "gatys":
                        mask_for_content = masks_in[c_idx].strip().lower()
                        if mask_for_content == "auto":
                            cmd += ["--face_mask", "auto"]
                        elif resolved_masks[c_idx]:
                            cmd += ["--face_mask", resolved_masks[c_idx]]

                    # Track VRAM (best-effort)
                    peak_gb = None
                    if HAS_TORCH and torch.cuda.is_available() and args.device != "cpu":
                        torch.cuda.reset_peak_memory_stats()

                    # Run (or skip)
                    secs = None
                    if args.skip_existing and out_img.exists():
                        print(f"[skip] exists: {out_img}")
                    else:
                        print(f"[RUN] {meth.upper()} {bb}  {c_name} × {s_name}")
                        # Uncomment for debugging:
                        # print("[DEBUG CMD]", " ".join(cmd))
                        secs = run_cmd(cmd)

                    if HAS_TORCH and torch.cuda.is_available() and args.device != "cpu":
                        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

                    # ---- Metrics ----
                    if not out_img.exists():
                        print(f"[warn] missing output: {out_img} (skipping metrics)")
                        continue

                    out_pil = load_rgb_pil(out_img)
                    out_gray = load_gray_01(out_img)

                    # Resize content gray to match output for SSIM
                    if c_gray_full is None:
                        c_gray_full = load_gray_01(cpath)
                    if c_gray_full.shape != out_gray.shape:
                        c_gray = resize(c_gray_full, out_gray.shape, mode="reflect", anti_aliasing=True)
                    else:
                        c_gray = c_gray_full

                    # Mask for metrics (face ROI)
                    mask01 = None
                    mask_path = resolved_masks[c_idx]
                    if mask_path and os.path.exists(mask_path):
                        mask01 = load_mask01(mask_path, out_gray.shape)

                    ssim_full = ssim01(c_gray, out_gray)
                    ssim_face = ssim_masked(c_gray, out_gray, mask01) if mask01 is not None else ssim_full

                    edge_full = edge_ratio_full(out_gray)
                    edge_bg   = edge_ratio_bg(out_gray, mask01)

                    palette_strong = palette_counts_pil(out_pil, k=args.palette_k, min_frac=args.palette_min_frac)

                    # LPIPS to content (full)
                    lpips_val = ""
                    if HAS_LPIPS and lp.model is not None:
                        c_pil_for_lpips = c_pil_full.resize(out_pil.size, Image.Resampling.LANCZOS)
                        lpips_val = lp.lpips(c_pil_for_lpips, out_pil)

                    # Prefer VRAM reported by child JSON, if present
                    meta_path = out_img.with_suffix(".json")
                    if meta_path.exists():
                        try:
                            with open(meta_path, "r") as f:
                                meta = json.load(f)
                            mv = meta.get("peak_vram_gb", None)
                            if isinstance(mv, (int, float)) and (peak_gb is None or peak_gb == 0.0):
                                peak_gb = float(mv)
                        except Exception:
                            pass

                    rows.append({
                        "content": cpath,
                        "style": spath,
                        "content_name": c_name,
                        "style_name": s_name,
                        "method": meth,
                        "backbone": bb,
                        "runtime_sec": (round(secs, 3) if isinstance(secs, (int, float)) else ""),
                        "peak_vram_gb": (round(peak_gb, 3) if isinstance(peak_gb, (int,float)) else ""),
                        "sizes": args.sizes,
                        "steps": args.steps,
                        "seed": args.seed,
                        "ssim_face": round(ssim_face, 4),
                        "ssim_full": round(ssim_full, 4),
                        "edge_ratio_full": round(edge_full, 4),
                        "edge_ratio_bg": round(edge_bg, 4),
                        "lpips_to_content": (round(lpips_val, 4) if isinstance(lpips_val, float) else ""),
                        "palette_strong": int(palette_strong),
                        "out": str(out_img)
                    })

    # Write CSV
    csv_path = Path(args.out_root) / "breadth_results.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            fieldnames = list(rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)
        print(f"\nSaved results to: {csv_path}")
        for r in rows:
            print(r)
    else:
        print("[info] no rows produced.")


if __name__ == "__main__":
    main()
