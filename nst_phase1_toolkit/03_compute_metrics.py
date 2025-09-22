import os, argparse, json
from pathlib import Path
from PIL import Image, ImageOps  
import numpy as np

def load_csv(path):
    import csv
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            rows.append(r)
    return rows

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def try_open(path):
    try:
        img = Image.open(path).convert("RGB")
        img = ImageOps.exif_transpose(img)      # <-- add this line
        return img
    except Exception:
        return None

def resize_to_max_side(img, max_side):
    if max_side is None or max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC)

def ssim_img(a, b):
    """Prefer skimage SSIM; if unavailable or fails, use a tiny MSE-based proxy."""
    try:
        from skimage.metrics import structural_similarity as ssim
        a_np = np.array(a)
        b_np = np.array(b)
        # use channel-wise SSIM and average to reduce memory
        vals = []
        for ch in range(3):
            vals.append(ssim(a_np[:,:,ch], b_np[:,:,ch], data_range=255))
        return float(np.mean(vals))
    except Exception:
        # fallback proxy: 1 - normalized MSE
        a_np = np.array(a, dtype=np.float32)
        b_np = np.array(b, dtype=np.float32)
        mse = np.mean((a_np - b_np) ** 2)
        return float(max(0.0, 1.0 - mse / (255.0**2)))

class LPIPSWrapper:
    def __init__(self, device="cpu", net="alex"):
        import torch, lpips
        self.torch = torch
        self.device = device
        self.model = lpips.LPIPS(net=net).to(device).eval()

    @staticmethod
    def _to_tensor(img, torch):
        arr = np.array(img)
        t = torch.tensor(arr).permute(2,0,1).unsqueeze(0).float() / 255.0
        t = t * 2 - 1  # [-1,1]
        return t

    def __call__(self, a, b):
        with self.torch.no_grad():
            ta = self._to_tensor(a, self.torch).to(self.device, non_blocking=True)
            tb = self._to_tensor(b, self.torch).to(self.device, non_blocking=True)
            d = self.model(ta, tb).item()
        return float(d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lpips", type=int, default=0, help="Enable LPIPS (1=on, 0=off). Default 0.")
    ap.add_argument("--lpips_net", default="alex", choices=["alex","vgg","squeeze"], help="LPIPS backbone (default alex).")
    ap.add_argument("--device", default="cpu", help="LPIPS device: cpu or cuda (default cpu).")
    ap.add_argument("--max_eval_side", type=int, default=768, help="Max image side for metrics (default 768).")
    args = ap.parse_args()

    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    eval_dir = Path(cfg["eval_dir"]); ensure_dir(eval_dir / "metrics")
    tables_csv = eval_dir / "tables" / "master_index.csv"
    rows = load_csv(str(tables_csv))

    # Optional LPIPS (init once)
    lpips_fn = None
    if args.lpips == 1:
        try:
            lpips_fn = LPIPSWrapper(device=args.device, net=args.lpips_net)
        except Exception as e:
            print(f"[WARN] LPIPS disabled (init failed): {e}")
            lpips_fn = None

    content_root_override = cfg.get("content_root_override") or ""
    finals_dir = Path(cfg["finals_dir"]); imgs_dir = Path(cfg["imgs_dir"])

    out_rows = []
    for r in rows:
        content_p = (r.get("content") or "").replace("\\","/")
        outp      = (r.get("out_final") or r.get("out_img") or "").replace("\\","/")

        def resolve(p, override):
            pth = Path(p)
            if pth.exists(): return pth
            if override:
                alt = Path(override) / p
                if alt.exists(): return alt
            return pth

        content_fp = resolve(content_p, content_root_override)
        out_fp     = Path(outp)
        if not content_fp.exists():
            print(f"[WARN] Missing content image: {content_fp}")
            continue
        if not out_fp.exists():
            base = os.path.basename(str(out_fp))
            for trial in [finals_dir/base, imgs_dir/base]:
                if trial.exists(): out_fp = trial; break
        if not out_fp.exists():
            print(f"[WARN] Missing output image: {out_fp}")
            continue

        a = try_open(str(content_fp))
        b = try_open(str(out_fp))
        if a is None or b is None:
            print(f"[WARN] Failed to open images for {out_fp}")
            continue

        # Downscale for metrics to control memory/VRAM
        a_s = resize_to_max_side(a, args.max_eval_side)
        b_s = resize_to_max_side(b, args.max_eval_side)

        if b_s.size != a_s.size:
            b_s = b_s.resize(a_s.size, Image.BICUBIC)

        # SSIM
        ssim_val = ssim_img(a_s, b_s)

        # LPIPS (with graceful GPU fallback)
        lp_val = None
        if lpips_fn is not None:
            try:
                lp_val = lpips_fn(a_s, b_s)
            except Exception as e:
                # If CUDA OOM, fallback to CPU once and retry
                msg = str(e).lower()
                if "cuda out of memory" in msg or "cublas" in msg:
                    try:
                        print("[INFO] LPIPS GPU OOM, falling back to CPU for this pair.")
                        lpips_cpu = LPIPSWrapper(device="cpu", net=args.lpips_net)
                        lp_val = lpips_cpu(a_s, b_s)
                    except Exception as e2:
                        print(f"[WARN] LPIPS failure on {out_fp}: {e2}")
                        lp_val = None
                else:
                    print(f"[WARN] LPIPS failure on {out_fp}: {e}")
                    lp_val = None

        met = {
            "manifest": r.get("manifest"),
            "content_name": r.get("content_name"),
            "style_name": r.get("style_name"),
            "out_path": str(out_fp),
            "style_weight": r.get("style_weight"),
            "tv_weight": r.get("tv_weight"),
            "layers": r.get("layers"),
            "ssim": ssim_val,
            "lpips": lp_val
        }
        out_rows.append(met)

    import csv
    met_csv = eval_dir / "metrics" / "metrics.csv"
    keys = ["manifest","content_name","style_name","out_path","style_weight","tv_weight","layers","ssim","lpips"]
    with open(met_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for m in out_rows: w.writerow(m)
    print(f"[OK] Wrote {met_csv} ({len(out_rows)} rows).")

if __name__ == "__main__":
    main()
