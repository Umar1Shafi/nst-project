#!/usr/bin/env python
import argparse, csv
from pathlib import Path
from PIL import Image
import numpy as np
import torch

def psnr(a, b):
    a = np.asarray(a).astype(np.float32) / 255.0
    b = np.asarray(b).astype(np.float32) / 255.0
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))

def try_ssim(a, b):
    try:
        from skimage.metrics import structural_similarity as ssim
        a = np.asarray(a).astype(np.float32) / 255.0
        b = np.asarray(b).astype(np.float32) / 255.0
        s, _ = ssim(a, b, channel_axis=2, full=True, data_range=1.0)
        return float(s)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="out/ablation_phase1/ablation_runs.csv")
    ap.add_argument("--content", required=True, help="The content image used in the ablation")
    ap.add_argument("--out-csv", default="out/ablation_phase1/ablation_metrics.csv")
    args = ap.parse_args()

    content = Image.open(args.content).convert("RGB")

    # LPIPS
    import lpips
    loss_fn = lpips.LPIPS(net='alex').to('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rows = []
    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out_path = Path(r["out"])
            if not out_path.exists(): 
                continue
            out_img = Image.open(out_path).convert("RGB")
            # size-match for fair LPIPS
            w, h = content.size
            out_img = out_img.resize((w, h), Image.LANCZOS)

            # LPIPS
            tA = torch.from_numpy(np.array(content).transpose(2,0,1))[None]/255*2-1
            tB = torch.from_numpy(np.array(out_img).transpose(2,0,1))[None]/255*2-1
            tA = tA.to(device).float(); tB = tB.to(device).float()
            lp = float(loss_fn(tA, tB).detach().cpu().item())

            rows.append({
                **r,
                "lpips": round(lp, 6),
                "psnr": round(psnr(content, out_img), 3),
                "ssim": (lambda x: round(x, 4) if x is not None else "")(try_ssim(content, out_img)),
            })

    # write
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[write] {out_csv}")

if __name__ == "__main__":
    main()
