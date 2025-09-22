# scripts/phase1/smoke_test_phase1.py
import os, sys, subprocess, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
OUT = ROOT / "out" / "phase1" / "smoke"
REP = ROOT / "report"

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    os.environ.setdefault("NST_SEED", "77")
    OUT.mkdir(parents=True, exist_ok=True)
    (REP / "phase1_smoke").mkdir(parents=True, exist_ok=True)

    # 1) Run a tiny baseline render (512px) on bundled sample pair
    out_img = OUT / "smoke_vg_still.png"
    run([
        sys.executable, str(ROOT/"scripts/core/nst_hybrid_ready.py"),
        "--content", str(ROOT/"data/content/Still_Life.jpg"),
        "--style",   str(ROOT/"data/style/Van_Gogh.jpg"),
        "--output",  str(out_img),
        "--steps", "200", "--content-weight", "1.0", "--style-weight", "5.0",
        "--tv-weight", "1e-6", "--max-side", "512"
    ])
    assert out_img.exists() and out_img.stat().st_size > 0, "Output image missing"

    # 2) Build a tiny index with just this pair
    idx = REP / "phase1_smoke" / "master_index.csv"
    run([
        sys.executable, str(ROOT/"nst_phase1_toolkit/01_build_master_index.py"),
        "--content-dir", str(ROOT/"data/content"),
        "--style-dir",   str(ROOT/"data/style"),
        "--out",         str(idx),
        "--limit",       "1"           # if supported; else omit this flag
    ])
    assert idx.exists(), "Index CSV missing"

    # 3) Compute metrics
    metrics = REP / "phase1_smoke" / "metrics.csv"
    run([
        sys.executable, str(ROOT/"nst_phase1_toolkit/03_compute_metrics.py"),
        "--index",  str(idx),
        "--out",    str(metrics)
    ])
    assert metrics.exists() and metrics.stat().st_size > 0, "Metrics CSV missing"

    print("\nSmoke test OK ✅")
    print("Image:", out_img)
    print("Index:", idx)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
