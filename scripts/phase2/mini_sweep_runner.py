# scripts/phase2/mini_sweep_runner.py  (v2 - matches phase2b_run.py plan schema)
"""
Mini Sweep Runner (Phase 2) — v2
Writes a plan JSON in the format expected by scripts/exp/phase2b_run.py:

{
  "plans": [
    {
      "content": "Still_Life",
      "style": "Monet",
      "neighborhood": [
        {"style_weight": 15000, "tv_weight": 0.0020, "edge_w": 0.04, "face_preserve": 0.36},
        {"style_weight": 25000, "tv_weight": 0.0032, "edge_w": 0.04, "face_preserve": 0.36}
      ]
    }
  ]
}

Usage (PowerShell example):
  python scripts\phase2\mini_sweep_runner.py ^
    --contents "data\content\Still_Life.jpg" ^
    --styles   "data\style\Monet.jpg" ^
    --sw "15000,25000" --tv "0.0020" --edge "0.04" ^
    --out-root "out\phase2_smoke" --device "cpu" --seed 77 --limit 2 --resume
"""
import os, sys, json, itertools, subprocess
from pathlib import Path
import argparse

def parse_floats_csv(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]

def parse_list_csv(s: str):
    items = [x.strip().strip('"').strip("'") for x in s.split(",")]
    return [i for i in items if i]

def to_token(p: str) -> str:
    """Convert a path or token into the content/style token expected by phase2b_run.py maps."""
    # If it's already a bare token like 'Monet' or 'Still_Life', keep it.
    if Path(p).suffix == "":
        return p
    stem = Path(p).stem  # e.g., 'Still_Life' or 'Ukiyo-e print'
    return stem.replace(" ", "_")  # 'Ukiyo-e_print'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contents", required=True, help="Comma-separated content paths or tokens (e.g., data/content/Still_Life.jpg or Still_Life)")
    ap.add_argument("--styles",   required=True, help="Comma-separated style paths or tokens (e.g., data/style/Monet.jpg or Monet)")
    ap.add_argument("--sw", required=True, help="Comma-separated style-weight grid, e.g. '15000,25000'")
    ap.add_argument("--tv", required=True, help="Comma-separated tv-weight grid, e.g. '0.0020,0.0032'")
    ap.add_argument("--edge", required=True, help="Comma-separated edge_w grid, e.g. '0.04'")
    ap.add_argument("--face-preserve", type=float, default=0.36, help="Face preserve weight for all jobs (used by finisher defaults)")
    ap.add_argument("--limit", type=int, default=None, help="Cap total jobs for quick demos")
    ap.add_argument("--out-root", default="out/phase2_smoke")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    contents = [to_token(x) for x in parse_list_csv(args.contents)]
    styles   = [to_token(x) for x in parse_list_csv(args.styles)]
    SW = parse_floats_csv(args.sw)
    TV = parse_floats_csv(args.tv)
    ED = parse_floats_csv(args.edge)

    # Build grouped plan
    jobs_per_group = len(SW) * len(TV) * len(ED)
    total_jobs = len(contents) * len(styles) * jobs_per_group

    plan = {"plans": []}
    for c in contents:
        for s in styles:
            neighborhood = []
            for sw in SW:
                for tv in TV:
                    for ed in ED:
                        neighborhood.append({
                            "style_weight": float(sw),
                            "tv_weight": float(tv),
                            "edge_w": float(ed),
                            "face_preserve": float(args.face_preserve),
                        })
            if args.limit is not None:
                neighborhood = neighborhood[: args.limit]
            plan["plans"].append({
                "content": c,
                "style": s,
                "neighborhood": neighborhood
            })

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    plan_path = out_root / "plan_generated.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print(f"[plan] wrote {plan_path} with {sum(len(p['neighborhood']) for p in plan['plans'])} jobs")
    env = {**os.environ, "NST_SEED": str(args.seed)}
    cmd = [
        sys.executable, "scripts/exp/phase2b_run.py",
        "--plan", str(plan_path),
        "--out_root", str(out_root),
        "--device", args.device,
        "--seed", str(args.seed),
    ]
    if args.resume:
        cmd.append("--resume")

    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print("[done] phase2b_run complete. See finals under", out_root/"finals")

if __name__ == "__main__":
    main()
