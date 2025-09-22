import sys, subprocess, os
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]  # adjust if needed
OUT = ROOT / "out" / "phase1_smoke"
OUT.mkdir(parents=True, exist_ok=True)

def run(cmd):
    print(">>", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)

def main():
    # tiny deterministic smoke
    os.environ["NST_SEED"] = os.environ.get("NST_SEED", "77")

    out_img = OUT / "smoke_vg_still.png"
    run([
        sys.executable, str(ROOT/"scripts/core/nst_hybrid_ready.py"),
        "--content", str(ROOT/"data/content/Still_Life.jpg"),
        "--style",   str(ROOT/"data/style/Van_Gogh.jpg"),
        "--out",     str(out_img),
        "--backbone", "vgg19",
        "--content_layer", "conv4_2",
        "--style_layers", "conv4_1,conv5_1",
        "--opt", "lbfgs",
        "--sizes", "384,512",
        "--steps", "200,200",
        "--style_weight", "15000",
        "--tv_weight", "0.0024",
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
    ])
    print(f"[ok] wrote {out_img}")

if __name__ == "__main__":
    main()
