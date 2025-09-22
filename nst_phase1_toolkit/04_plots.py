
import os, argparse, json
from pathlib import Path
import csv
import matplotlib.pyplot as plt

def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            rows.append(r)
    return rows

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--by", default="style_weight")  # hyperparam on x-axis
    args = ap.parse_args()

    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    eval_dir = Path(cfg["eval_dir"]); ensure_dir(eval_dir / "figures")
    met_csv = eval_dir / "metrics" / "metrics.csv"
    rows = load_csv(str(met_csv))

    # group by (content_name, style_name)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["content_name"], r["style_name"])].append(r)

    # plot SSIM vs hyperparam (and LPIPS if available)
    for (cname, sname), lst in groups.items():
        # sort by x
        def key_fn(x):
            try: return float(x.get(args.by, 0))
            except: return 0.0
        lst = sorted(lst, key=key_fn)

        xs, ssim_ys, lpips_ys = [], [], []
        for r in lst:
            try: xs.append(float(r.get(args.by, 0)))
            except: xs.append(0.0)
            try: ssim_ys.append(float(r.get("ssim", 0)))
            except: ssim_ys.append(0.0)
            lp = r.get("lpips", "")
            lpips_ys.append(float(lp) if lp not in ("", None) else None)

        # SSIM plot
        plt.figure()
        plt.plot(xs, ssim_ys, marker="o")
        plt.xlabel(args.by)
        plt.ylabel("SSIM")
        plt.title(f"{cname} x {sname} — SSIM vs {args.by}")
        outp = eval_dir / "figures" / f"ssim__{cname}__{sname}__by-{args.by}.png"
        plt.savefig(outp, bbox_inches="tight", dpi=150)
        plt.close()

        # LPIPS plot if present
        if any(v is not None for v in lpips_ys):
            xs2, lp2 = [], []
            for x, v in zip(xs, lpips_ys):
                if v is not None:
                    xs2.append(x); lp2.append(v)
            if xs2:
                plt.figure()
                plt.plot(xs2, lp2, marker="o")
                plt.xlabel(args.by)
                plt.ylabel("LPIPS (lower is better)")
                plt.title(f"{cname} x {sname} — LPIPS vs {args.by}")
                outp = eval_dir / "figures" / f"lpips__{cname}__{sname}__by-{args.by}.png"
                plt.savefig(outp, bbox_inches="tight", dpi=150)
                plt.close()

        print(f"[OK] Plots for {cname} x {sname}")

if __name__ == "__main__":
    main()
