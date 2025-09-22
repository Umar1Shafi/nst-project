# scripts/bench_compare.py
import os, time, json, subprocess, argparse, csv, sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import metrics, color, feature  # pip install scikit-image
from skimage.transform import resize

# Optional torch (for CUDA VRAM stats)
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_CORE = REPO_ROOT / "scripts" / "core"
PATHS = {
    "gatys":  SCRIPTS_CORE / "nst_hybrid_ready.py",
    "adain":  SCRIPTS_CORE / "baseline_adain_loss.py",
    "wct":    SCRIPTS_CORE / "baseline_wct_loss.py",
}
def run_cmd(args_list):
    """
    Run a subprocess and return wall-clock seconds.
    Captures stdout/stderr so we can show them on failure.
    """
    t0 = time.perf_counter()
    cp = subprocess.run(args_list, capture_output=True, text=True)
    t1 = time.perf_counter()
    if cp.returncode != 0:
        print("\n--- Subprocess STDOUT ---\n", cp.stdout)
        print("\n--- Subprocess STDERR ---\n", cp.stderr)
        raise subprocess.CalledProcessError(cp.returncode, cp.args)
    return t1 - t0


def load_gray_01(path):
    """Load image and return grayscale float32 array in [0,1], shape HxW."""
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    gray = color.rgb2gray(arr)
    return gray


def ssim01(a, b):
    """SSIM for two same-sized grayscale arrays in [0,1]."""
    return float(metrics.structural_similarity(a, b, data_range=1.0))


def ssim_masked(a, b, mask01):
    """
    SSIM map, averaged under the given HxW mask (values in [0,1]).
    If mask is None/empty, returns global SSIM.
    """
    score, ssim_map = metrics.structural_similarity(a, b, data_range=1.0, full=True)
    if mask01 is None or mask01.sum() == 0:
        return float(score)
    return float((ssim_map * mask01).sum() / (mask01.sum() + 1e-6))


def edge_ratio(gray):
    """Canny edge density (0..1) on grayscale [0,1]."""
    edges = feature.canny(gray, sigma=1.2)
    return float(edges.mean())


def load_mask01(path, target_hw):
    """Load mask as HxW in [0,1] and resize to target (H,W)."""
    m = Image.open(path).convert("L")
    m = np.asarray(m, dtype=np.float32) / 255.0
    m = resize(m, target_hw, mode="reflect", anti_aliasing=True)
    m = (m > 0.5).astype(np.float32)  # binarize
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--out_root", default="out/bench")
    ap.add_argument("--sizes", default="384,768")
    ap.add_argument("--steps", default="300,450")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--face_mask", default=None,
                    help="Optional face mask PNG (white=face). Resized to output size for ROI metrics.")
    args = ap.parse_args()

    sizes = args.sizes      # pass-through string for child scripts
    steps = args.steps      # pass-through string for child scripts
    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    # Methods x backbones
    jobs = [
        # (name,   script,                          extra args list)
        ("gatys", PATHS["gatys"], [
            "--style_layers", "conv4_1,conv5_1",
            "--content_layer", "conv4_2",
            "--steps", steps
        ]),
        ("adain", PATHS["adain"], [
            "--content_layer", "conv4_2",
            "--style_layers", "conv4_1,conv5_1",
            "--steps", steps
        ]),
        ("wct",   PATHS["wct"], [
            "--layer", "conv4_1",
            "--steps", steps
        ]),
    ]
    backbones = ["vgg19", "vgg16"]

    # Load content gray once (we will resize as needed later)
    content_gray_full = load_gray_01(args.content)

    rows = []
    for meth, script, extra in jobs:
        for bb in backbones:
            out_dir = Path(args.out_root) / f"{meth}_{bb}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_img = str(out_dir / "out.jpg")

            # Build command
            cmd = [sys.executable, script,
                   "--backbone", bb,
                   "--content", args.content,
                   "--style", args.style,
                   "--out", out_img,
                   "--sizes", sizes,
                   "--seed", str(args.seed),
                   "--device", args.device] + extra

            # Only Gatys supports --face_mask; others ignore it
            if meth == "gatys" and args.face_mask:
                cmd += ["--face_mask", args.face_mask]

            # Track GPU memory if possible
            peak_gb = None
            if HAS_TORCH and torch.cuda.is_available() and args.device != "cpu":
                torch.cuda.reset_peak_memory_stats()

            # Run once
            secs = run_cmd(cmd)

            # Query peak VRAM
            if HAS_TORCH and torch.cuda.is_available() and args.device != "cpu":
                peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

            # Compute quick metrics
            out_gray = load_gray_01(out_img)

            # Resize content gray to match output size (safety)
            if content_gray_full.shape != out_gray.shape:
                c_resz = resize(content_gray_full, out_gray.shape, mode="reflect", anti_aliasing=True)
            else:
                c_resz = content_gray_full

            # Optional face mask for ROI-SSIM
            mask_gray = None
            if args.face_mask and os.path.exists(args.face_mask):
                try:
                    mask_gray = load_mask01(args.face_mask, out_gray.shape)
                except Exception:
                    mask_gray = None

            ssim_full = ssim01(c_resz, out_gray)
            ssim_face = ssim_masked(c_resz, out_gray, mask_gray) if mask_gray is not None else ssim_full
            edge_val = edge_ratio(out_gray)

            # Prefer VRAM from child JSON if present (and we didn't get a good one)
            meta_path = out_img.replace(".jpg", ".json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    mv = meta.get("peak_vram_gb", None)
                    if isinstance(mv, (int, float)) and (peak_gb is None or peak_gb == 0.0):
                        peak_gb = float(mv)
                except Exception:
                    pass

            row = {
                "method": meth,
                "backbone": bb,
                "runtime_sec": round(secs, 3),
                "peak_vram_gb": round(peak_gb, 3) if peak_gb is not None else "",
                "sizes": sizes,
                "steps": steps,
                "seed": args.seed,
                # Keep legacy column name (used by your earlier table) and add the components:
                "ssim_to_content": round(ssim_face if mask_gray is not None else ssim_full, 4),
                "ssim_face": round(ssim_face, 4),
                "ssim_full": round(ssim_full, 4),
                "edge_ratio": round(edge_val, 4),
                "out": out_img
            }
            rows.append(row)

    # Write CSV
    csv_path = Path(args.out_root) / "bench_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {csv_path}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
