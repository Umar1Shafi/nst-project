# Phase-3 Smoke Test (images-only)
# - Prefers Extra/images/*.png; falls back to out/phase3/**/*.png
# - Requires: data/content/portrait2.jpg and data/content/street.jpg

import os, sys, glob, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "report" / "phase3_eval_advanced"

def run(cmd, env=None):
    print(">>", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, env=env)

def main():
    # 1) Find outputs
    cands = sorted(glob.glob(str(ROOT / "Extra" / "images" / "*.png")))
    if not cands:
        cands = sorted(glob.glob(str(ROOT / "out" / "phase3" / "**" / "*.png"), recursive=True))
    assert cands, "No Phase-3 outputs found under Extra/images or out/phase3/**"

    # 2) References
    portrait = ROOT / "data" / "content" / "portrait2.jpg"
    scene    = ROOT / "data" / "content" / "street.jpg"
    assert portrait.exists() and scene.exists(), "Missing reference inputs (portrait2.jpg or street.jpg)"

    # 3) Run evaluator
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT/"scripts/phase3/phase3_eval_advanced.py"),
        "--outputs-glob", str(ROOT/"Extra/images/*.png") if "Extra/images" in str(cands[0]) else str(ROOT/"out/phase3/**/*.png"),
        "--portrait-input", str(portrait),
        "--scene-input", str(scene),
        "--outdir", str(OUTDIR)
    ]
    run(cmd)

    # 4) Check artifacts
    csv_path = OUTDIR/"phase3_metrics_advanced.csv"
    sc_dir = OUTDIR/"sidecars"
    assert csv_path.exists() and csv_path.stat().st_size > 0, "Metrics CSV missing or empty"
    assert sc_dir.exists() and any(sc_dir.glob("*.json")), "No sidecars were written"

    print("\nSmoke test OK ✅")
    print("CSV:      ", csv_path)
    print("Sidecars: ", sc_dir)

if __name__ == "__main__":
    main()
