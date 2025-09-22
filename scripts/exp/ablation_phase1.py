#!/usr/bin/env python
import argparse, itertools, json, os, shlex, subprocess, time
from pathlib import Path

SETS = [
    # tag, layers
    ("L45",  "conv4_1,conv5_1"),
    ("L345", "conv3_1,conv4_1,conv5_1"),
    ("L12345","conv1_1,conv2_1,conv3_1,conv4_1,conv5_1"),
]

def run(cmd, cwd=None, env=None):
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=cwd, env=env)
    return time.perf_counter() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--out-root", default="out/ablation_phase1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=77)
    # common NST params
    ap.add_argument("--sizes", default="384,768")
    ap.add_argument("--steps", default="300,400")
    ap.add_argument("--style-weight", type=float, default=20000.0)
    ap.add_argument("--tv-weight", type=float, default=0.0028)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]  # project root
    out_root = repo / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    backbones = ["vgg19", "vgg16"]
    opts = ["lbfgs", "adam"]
    results = []

    env = os.environ.copy()
    env["NST_SEED"] = str(args.seed)

    for backbone, opt, (tag, layers) in itertools.product(backbones, opts, SETS):
        c_stem = Path(args.content).stem
        s_stem = Path(args.style).stem
        out = out_root / f"{c_stem}__{s_stem}__{backbone}__{opt}__{tag}.png"
        core_script = repo / "scripts" / "core" / "nst_hybrid_ready.py"
        py = os.fspath(Path(os.sys.executable))
        cmd = [
            os.fspath(core_script),
            "--content", args.content,
            "--style",   args.style,
            "--out",     os.fspath(out),
            "--backbone", backbone,
            "--content_layer", "conv4_2",
            "--style_layers", layers,
            "--opt", opt,
            "--sizes", args.sizes,
            "--steps", args.steps,
            "--style_weight", str(args.style_weight),
            "--tv_weight", str(args.tv_weight),
            "--edge_w", "0.08", "--edge_face_down", "0.4",
            "--color_prematch_strength", "0.6",
            "--device", args.device,
            "--seed", str(args.seed),                 # also add this (see §2)
        ]
        full = [py] + cmd
        elapsed = run(full, cwd=repo, env=env)

        results.append({
            "content": args.content,
            "style": args.style,
            "out": os.fspath(out),
            "backbone": backbone,
            "opt": opt,
            "layers_tag": tag,
            "layers": layers,
            "sizes": args.sizes,
            "steps": args.steps,
            "style_weight": args.style_weight,
            "tv_weight": args.tv_weight,
            "device": args.device,
            "seed": args.seed,
            "elapsed_sec": round(elapsed, 3),
        })

    # write manifest + CSV
    (out_root / "manifest.json").write_text(json.dumps(results, indent=2))
    csv = "content,style,out,backbone,opt,layers_tag,layers,sizes,steps,style_weight,tv_weight,device,seed,elapsed_sec\n"
    for r in results:
        csv += ",".join(str(r[k]) for k in ["content","style","out","backbone","opt","layers_tag","layers","sizes","steps","style_weight","tv_weight","device","seed","elapsed_sec"]) + "\n"
    (out_root / "ablation_runs.csv").write_text(csv)
    print(f"[write] {out_root/'ablation_runs.csv'}")

if __name__ == "__main__":
    main()
