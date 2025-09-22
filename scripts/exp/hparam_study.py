#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter study runner for NST:
  1) Content weight mini-sweep     (2 x 3 = 6 runs)
  2) Style-layer presets           (3 x 3 x 2 = 18 runs)
  3) Size vs time (quality vs time) (3 runs)

Reuses your existing:
- scripts/core/nst_hybrid_ready.py
- scripts/finish/hybrid_pipeline.py
- scripts/eval/eval_pairs_imgs_vs_finals.py
"""

import argparse, subprocess, sys, json
from pathlib import Path
from time import time

# -------- helpers --------
def run(cmd, cwd=None):
    print("[cmd]", " ".join(map(str, cmd)))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f"[error] command failed with exit code {r.returncode}")

def exists_or_die(p: Path, hint: str):
    if not p.exists():
        raise SystemExit(f"[missing] {p} — {hint}")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -------- core call wrappers --------
def call_nst_core(py, core_script, *, content, style, out_path, device, seed,
                  sizes, steps, backbone, style_weight, tv_weight, edge_w,
                  face_preserve, cpm_strength, content_weight=None, style_layers=None):
    cmd = [py, str(core_script),
           "--content", str(content),
           "--style",   str(style),
           "--out",     str(out_path),
           "--device",  device,
           "--seed",    str(seed),
           "--sizes",   str(sizes),
           "--steps",   str(steps),
           "--backbone", backbone,
           "--style_weight", str(style_weight),
           "--tv_weight", str(tv_weight),
           "--edge_w", str(edge_w),
           "--face_preserve", str(face_preserve),
           "--color_prematch_strength", str(cpm_strength)]
    if content_weight is not None:
        cmd += ["--content_weight", str(content_weight)]
    if style_layers:
        cmd += ["--style_layers", style_layers]
    t0 = time()
    run(cmd)
    return time() - t0

def call_finisher(py, finisher_script, *, raw_path, style_path, out_path):
    """
    Your finisher expects:
      --inp <raw image>  --style <style image>  --out <final image>
    """
    cmd = [py, str(finisher_script),
           "--inp",   str(raw_path),
           "--style", str(style_path),
           "--out",   str(out_path)]
    run(cmd)

def call_evaluator(py, evaluator_script, *, root, content_dir, csv_out, plots_dir, lpips=False, max_pairs=None):
    cmd = [py, str(evaluator_script),
           "--root", str(root),
           "--csv",  str(csv_out),
           "--plots", str(plots_dir)]
    if content_dir:
        cmd += ["--content_dir", str(content_dir)]
    if lpips:
        cmd += ["--lpips", "1"]
    if max_pairs:
        cmd += ["--max", str(max_pairs)]
    run(cmd)

# -------- naming helper --------
def stem_for(content_name, style_name, sw, tv, edge, fp, cpm, tag, sizes, steps, cw=None):
    stem = f"{content_name}__{style_name}__sw{int(sw)}_tv{tv:.4f}_edge{edge:.2f}_fp{fp:.2f}_cpm{cpm:.2f}"
    if tag:
        stem += f"_{tag}"
    if cw is not None:
        stem += f"_cw{cw:g}"
    stem += f"_{sizes}_{steps}"
    return stem

# -------- experiments --------
def ex_content_weight(args, py, core_script, finisher_script, evaluator_script):
    out_root = Path(args.out_root) / "content_weight"
    d_imgs   = out_root / "imgs"
    d_fin    = out_root / "finals"
    d_logs   = out_root / "logs"
    ensure_dir(d_imgs); ensure_dir(d_fin); ensure_dir(d_logs)

    content = Path(args.content_dir) / "portrait2.jpg"
    style   = Path(args.style_dir)   / "Monet.jpg"
    exists_or_die(content, "expected content: portrait2.jpg")
    exists_or_die(style,   "expected style: Monet.jpg")

    tv, edge, fp, cpm = args.tv_weight, args.edge_w, args.face_preserve, args.cpm_strength
    sizes, steps = args.sizes, args.steps
    sw_anchors   = [17000, 24000]
    cw_values    = [0.5, 1.0, 1.5]
    style_layers = "conv3_1,conv4_1,conv5_1"  # 'layersC' mid preset
    tag = "layersC"

    runtimes = []
    for sw in sw_anchors:
        for cw in cw_values:
            stem = stem_for("portrait2", "Monet", sw, tv, edge, fp, cpm, tag, sizes, steps, cw=cw)
            raw_path = d_imgs / f"{stem}.jpg"
            fin_path = d_fin  / f"{stem}_final.jpg"

            dt = call_nst_core(py, core_script,
                               content=content, style=style, out_path=raw_path,
                               device=args.device, seed=args.seed, sizes=sizes, steps=steps,
                               backbone=args.backbone, style_weight=sw, tv_weight=tv, edge_w=edge,
                               face_preserve=fp, cpm_strength=cpm,
                               content_weight=cw, style_layers=style_layers)
            call_finisher(py, finisher_script, raw_path=raw_path, style_path=style, out_path=fin_path)
            runtimes.append({"stem": stem, "seconds": dt, "sizes": sizes, "steps": steps,
                             "style_weight": sw, "content_weight": cw})

    (d_logs / "runtime.json").write_text(json.dumps(runtimes, indent=2))
    call_evaluator(py, evaluator_script,
                   root=out_root, content_dir=args.content_dir,
                   csv_out=Path(args.report_root) / "content_weight" / "pairs_delta_metrics.csv",
                   plots_dir=Path(args.report_root) / "content_weight" / "plots",
                   lpips=args.lpips)

def ex_style_layers(args, py, core_script, finisher_script, evaluator_script):
    out_root = Path(args.out_root) / "style_layers"
    d_imgs   = out_root / "imgs"
    d_fin    = out_root / "finals"
    d_logs   = out_root / "logs"
    ensure_dir(d_imgs); ensure_dir(d_fin); ensure_dir(d_logs)

    contents = ["portrait2.jpg", "city.jpg", "Still_Life.jpg"]
    styles   = ["Monet.jpg", "Ukiyo-e_print.jpg"]
    for c in contents:
        exists_or_die(Path(args.content_dir) / c, f"missing content: {c}")
    for s in styles:
        exists_or_die(Path(args.style_dir) / s, f"missing style: {s}")

    presets = [
        ("layersA", "conv1_1,conv2_1,conv3_1,conv4_1,conv5_1"),
        ("layersB", "conv3_1,conv4_1"),
        ("layersC", "conv3_1,conv4_1,conv5_1"),
    ]
    tv, edge, fp, cpm = args.tv_weight, args.edge_w, args.face_preserve, args.cpm_strength
    sw, sizes, steps  = args.style_weight, args.sizes, args.steps

    runtimes = []
    for tag, style_layers in presets:
        for c in contents:
            for s in styles:
                cstem = Path(c).stem
                sstem = Path(s).stem
                stem  = stem_for(cstem, sstem, sw, tv, edge, fp, cpm, tag, sizes, steps)
                raw_path = d_imgs / f"{stem}.jpg"
                fin_path = d_fin  / f"{stem}_final.jpg"

                dt = call_nst_core(py, core_script,
                                   content=Path(args.content_dir) / c,
                                   style=Path(args.style_dir) / s,
                                   out_path=raw_path, device=args.device, seed=args.seed,
                                   sizes=sizes, steps=steps, backbone=args.backbone,
                                   style_weight=sw, tv_weight=tv, edge_w=edge,
                                   face_preserve=fp, cpm_strength=cpm,
                                   style_layers=style_layers)
                call_finisher(py, finisher_script, raw_path=raw_path,
                              style_path=Path(args.style_dir) / s, out_path=fin_path)
                runtimes.append({"stem": stem, "seconds": dt, "sizes": sizes, "steps": steps,
                                 "style_layers": tag})

    (d_logs / "runtime.json").write_text(json.dumps(runtimes, indent=2))
    call_evaluator(py, evaluator_script,
                   root=out_root, content_dir=args.content_dir,
                   csv_out=Path(args.report_root) / "style_layers" / "pairs_delta_metrics.csv",
                   plots_dir=Path(args.report_root) / "style_layers" / "plots",
                   lpips=args.lpips)

def ex_size_time(args, py, core_script, finisher_script, evaluator_script):
    out_root = Path(args.out_root) / "size_time"
    d_imgs   = out_root / "imgs"
    d_fin    = out_root / "finals"
    d_logs   = out_root / "logs"
    ensure_dir(d_imgs); ensure_dir(d_fin); ensure_dir(d_logs)

    content = Path(args.content_dir) / "portrait2.jpg"
    style   = Path(args.style_dir)   / "Monet.jpg"
    exists_or_die(content, "expected portrait2.jpg")
    exists_or_die(style,   "expected Monet.jpg")

    tv, edge, fp, cpm = args.tv_weight, args.edge_w, args.face_preserve, args.cpm_strength
    sw, steps         = args.style_weight, args.steps
    style_layers      = "conv3_1,conv4_1,conv5_1"
    tag               = "layersC"

    sizes_list = [384, 512, 768]
    runtimes = []
    for sz in sizes_list:
        stem = stem_for("portrait2", "Monet", sw, tv, edge, fp, cpm, tag, sz, steps)
        raw_path = d_imgs / f"{stem}.jpg"
        fin_path = d_fin  / f"{stem}_final.jpg"

        dt = call_nst_core(py, core_script,
                           content=content, style=style, out_path=raw_path,
                           device=args.device, seed=args.seed, sizes=sz, steps=steps,
                           backbone=args.backbone, style_weight=sw, tv_weight=tv, edge_w=edge,
                           face_preserve=fp, cpm_strength=cpm, style_layers=style_layers)
        call_finisher(py, finisher_script, raw_path=raw_path, style_path=style, out_path=fin_path)
        runtimes.append({"stem": stem, "seconds": dt, "sizes": sz, "steps": steps})

    (d_logs / "runtime.json").write_text(json.dumps(runtimes, indent=2))
    call_evaluator(py, evaluator_script,
                   root=out_root, content_dir=args.content_dir,
                   csv_out=Path(args.report_root) / "size_time" / "pairs_delta_metrics.csv",
                   plots_dir=Path(args.report_root) / "size_time" / "plots",
                   lpips=args.lpips)

# -------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable, help="python executable")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--backbone", default="vgg19")

    # defaults you can tweak
    ap.add_argument("--style_weight", type=int, default=20000)
    ap.add_argument("--tv_weight", type=float, default=0.0028)
    ap.add_argument("--edge_w", type=float, default=0.08)
    ap.add_argument("--face_preserve", type=float, default=0.75)
    ap.add_argument("--cpm_strength", type=float, default=0.60)
    ap.add_argument("--sizes", type=int, default=512)
    ap.add_argument("--steps", type=int, default=500)

    ap.add_argument("--content_dir", default="data/content")
    ap.add_argument("--style_dir", default="data/style")
    ap.add_argument("--out_root",   default="out/hparam_study")
    ap.add_argument("--report_root", default="report/hparam")
    ap.add_argument("--lpips", type=int, default=0, help="1 to compute LPIPS in evaluator (requires torch+lpips)")

    ap.add_argument("--do", choices=["all","content_weight","style_layers","size_time"], default="all")

    args = ap.parse_args()

    repo = Path(".").resolve()
    core_script      = repo / "scripts/core/nst_hybrid_ready.py"
    finisher_script  = repo / "scripts/finish/hybrid_pipeline.py"
    evaluator_script = repo / "scripts/eval/eval_pairs_imgs_vs_finals.py"

    for p, hint in [(core_script,"missing core NST runner"),
                    (finisher_script,"missing finisher"),
                    (evaluator_script,"missing evaluator")]:
        exists_or_die(p, hint)

    for sub in ["content_weight","style_layers","size_time"]:
        ensure_dir(Path(args.report_root) / sub)

    if args.do in ("all", "content_weight"):
        print("\n=== 1) Content-weight mini-sweep ===")
        ex_content_weight(args, args.python, core_script, finisher_script, evaluator_script)

    if args.do in ("all", "style_layers"):
        print("\n=== 2) Style-layer presets ===")
        ex_style_layers(args, args.python, core_script, finisher_script, evaluator_script)

    if args.do in ("all", "size_time"):
        print("\n=== 3) Size vs time ===")
        ex_size_time(args, args.python, core_script, finisher_script, evaluator_script)

    print("\n[done] Hyperparameter study complete.")
    print(f"CSV/plots under: {Path(args.report_root).resolve()}")

if __name__ == "__main__":
    main()
