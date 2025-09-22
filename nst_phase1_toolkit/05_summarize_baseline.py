
import os, argparse, json, csv
from pathlib import Path
import math

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
    args = ap.parse_args()

    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    eval_dir = Path(cfg["eval_dir"]); ensure_dir(eval_dir / "tables")

    met_csv = eval_dir / "metrics" / "metrics.csv"
    rows = load_csv(str(met_csv))

    # group by (content_name, style_name)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["content_name"], r["style_name"])].append(r)

    choices = []
    for (cname, sname), lst in groups.items():
        best = None
        best_score = -1e9
        for r in lst:
            try:
                ssim = float(r.get("ssim", 0))
            except:
                ssim = 0.0
            lpips = r.get("lpips", "")
            lp = float(lpips) if lpips not in ("", None) else None

            # Balanced score: prefer higher SSIM (content), lower LPIPS (perceptual difference)
            score = ssim
            if lp is not None:
                score = 0.7 * ssim - 0.3 * lp

            if score > best_score:
                best_score = score
                best = r

        if best:
            choices.append({
                "content_name": cname,
                "style_name": sname,
                "style_weight": best.get("style_weight"),
                "tv_weight": best.get("tv_weight"),
                "layers": best.get("layers"),
                "ssim": best.get("ssim"),
                "lpips": best.get("lpips"),
                "out_path": best.get("out_path"),
                "score": best_score
            })

    out_csv = eval_dir / "tables" / "baseline_choices.csv"
    keys = ["content_name","style_name","style_weight","tv_weight","layers","ssim","lpips","score","out_path"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for c in choices: w.writerow(c)

    print(f"[OK] Suggested baseline choices -> {out_csv}")
    for c in choices[:10]:
        print(f" - {c['content_name']} x {c['style_name']}: sw={c['style_weight']} tv={c['tv_weight']} layers={c['layers']}  (SSIM={c['ssim']} LPIPS={c['lpips']})")

if __name__ == "__main__":
    main()
