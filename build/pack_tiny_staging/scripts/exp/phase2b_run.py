# scripts/exp/phase2b_run.py
import argparse, json, os, shutil, subprocess, time, re, math, csv, sys
from pathlib import Path

# Map tag tokens -> actual file names on disk
CONTENT_MAP = {
    "portrait2":  "portrait2.jpg",
    "Still_Life": "Still_Life.jpg",
    "city":       "city.jpg",
    "animal":     "animal.jpg",
}
STYLE_MAP = {
    "Ukiyo-e_print": "Ukiyo-e_print.jpg",
    "Monet":         "Monet.jpg",
    "Matisse":       "Matisse.jpg",
}

# --------- utilities ---------
def safe_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def run_cmd(cmd):
    t0 = time.perf_counter()
    cp = subprocess.run(cmd, text=True, capture_output=True)
    t1 = time.perf_counter()
    secs = t1 - t0
    if cp.returncode != 0:
        print("\n--- STDOUT ---\n", cp.stdout)
        print("\n--- STDERR ---\n", cp.stderr)
        raise RuntimeError(f"Subprocess failed: {cmd}")
    return secs

def is_bad(x):
    try:
        xf = float(x)
    except Exception:
        return True
    return (math.isnan(xf) or math.isinf(xf))

def coerce(cfg, key, default):
    """Return a good float for cfg[key]; if missing/NaN/inf/str->bad, return default."""
    v = cfg.get(key, default)
    try:
        v = float(v)
    except Exception:
        return float(default)
    if math.isnan(v) or math.isinf(v):
        return float(default)
    return float(v)

# --------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="out/phase2/plan_phase2B.json")
    ap.add_argument("--out_root", default="out/phase2B")
    ap.add_argument("--data_content", default="data/content")
    ap.add_argument("--data_style",   default="data/style")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_imgs = out_root / "nst"
    out_fins = out_root / "finals"
    out_logs = out_root / "logs"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_fins.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    log_csv   = out_logs / "phase2B_runs.csv"
    fixes_csv = out_logs / "phase2B_fixes.csv"
    if not log_csv.exists():
        log_csv.write_text("tag,status,secs\n", encoding="utf-8")
    if not fixes_csv.exists():
        fixes_csv.write_text("content,style,idx,field,original,replaced\n", encoding="utf-8")

    with open(args.plan, "r", encoding="utf-8") as f:
        plan = json.load(f)
    groups = plan.get("plans", [])

    n_ok = n_fail = n_skip = n_fix = 0

    for g in groups:
        content = g["content"]
        style   = g["style"]
        neigh   = g.get("neighborhood", [])

        # Resolve paths
        content_file = Path(args.data_content) / CONTENT_MAP.get(content, f"{content}.jpg")
        style_file   = Path(args.data_style)   / STYLE_MAP.get(style,     f"{style}.jpg")

        if not content_file.exists():
            print(f"[skip] missing content: {content_file}")
            n_skip += len(neigh); continue
        if not style_file.exists():
            print(f"[skip] missing style:   {style_file}")
            n_skip += len(neigh); continue

        is_portrait = ("portrait" in content.lower())

        # Finisher defaults by content class
        if is_portrait:
            fin_edge_gain, fin_edge_face = 0.32, 0.18
            mask_arg = ["--face_mask","auto"]
            fp_default = 0.95
            ew_default = 0.08
        elif content in ("city","animal"):
            fin_edge_gain, fin_edge_face = 0.38, 0.24
            mask_arg = []
            fp_default = 0.75
            ew_default = 0.10
        else:  # Still_Life & other non-portraits
            fin_edge_gain, fin_edge_face = 0.36, 0.22
            mask_arg = []
            fp_default = 0.75
            ew_default = 0.10

        # Global defaults
        sw_default  = 20000.0
        tv_default  = 0.0028
        cpm_default = 0.60

        for idx, cfg in enumerate(neigh):
            # Keep originals for fix log
            orig = {
                "style_weight": cfg.get("style_weight", None),
                "tv_weight": cfg.get("tv_weight", None),
                "edge_w": cfg.get("edge_w", None),
                "face_preserve": cfg.get("face_preserve", None),
                "color_prematch_strength": cfg.get("color_prematch_strength", None),
            }

            # Coerce with robust defaults
            sw  = coerce(cfg, "style_weight", sw_default)
            tv  = coerce(cfg, "tv_weight", tv_default)
            ew  = coerce(cfg, "edge_w", ew_default)
            fp  = coerce(cfg, "face_preserve", fp_default)
            cpm = coerce(cfg, "color_prematch_strength", cpm_default)

            # If anything was invalid, log a fix
            for k, dv in (("style_weight", sw_default),
                          ("tv_weight", tv_default),
                          ("edge_w", ew_default),
                          ("face_preserve", fp_default),
                          ("color_prematch_strength", cpm_default)):
                ov = orig.get(k, None)
                try:
                    bad = is_bad(ov)
                except Exception:
                    bad = True
                if bad:
                    with open(fixes_csv, "a", encoding="utf-8") as fx:
                        fx.write(f"{content},{style},{idx},{k},{ov},{locals()[k.split('_')[0] if k!='color_prematch_strength' else 'cpm']}\n")
                    n_fix += 1

            # Build tag (now safe)
            tag = f"{content}__{style}__sw{int(round(sw))}_tv{tv:.4f}_ew{ew:.2f}_fp{fp:.2f}_cpm{cpm:.2f}"
            tag = safe_tag(tag)

            out_img = out_imgs / f"{tag}.jpg"
            out_fin = out_fins / f"{tag}_final.jpg"
            if args.resume and out_fin.exists():
                n_skip += 1
                continue

            # ---- NST ----
            nst_cmd = [
                sys.executable, "scripts/core/nst_hybrid_ready.py",
                "--content", str(content_file),
                "--style",   str(style_file),
                "--out",     str(out_img),
                "--backbone","vgg19",
                "--style_layers","conv4_1,conv5_1",
                "--content_layer","conv4_2",
                "--opt","lbfgs",
                "--style_weight", str(sw),
                "--tv_weight",    f"{tv:.5f}",
                "--edge_w",       f"{ew:.3f}",
                "--edge_face_down","0.4",
                "--face_preserve",f"{fp:.3f}",
                "--color_prematch_strength", f"{cpm:.2f}",
                "--sizes","384,768", "--steps","300,400",
                "--device", args.device, "--seed", str(args.seed),
            ] + mask_arg

            try:
                secs_nst = run_cmd(nst_cmd)
            except Exception as e:
                print(f"[FAIL:NST] {tag}: {e}")
                with open(log_csv, "a", encoding="utf-8") as f:
                    f.write(f"{tag},FAIL_NST,0.0\n")
                n_fail += 1
                continue

            # ---- Finisher ----
            fin_cmd = [
                sys.executable, "scripts/finish/hybrid_pipeline.py",
                "--inp", str(out_img),
                "--style", str(style_file),
                "--out", str(out_fin),
                "--use_guided",
                "--bg_colors","10","--face_colors","16",
                "--edge_gain", f"{fin_edge_gain:.2f}",
                "--edge_gain_face", f"{fin_edge_face:.2f}",
                "--edge_thickness","1",
                "--paper","--paper_strength","0.06",
                "--seed", str(args.seed)
            ] + (mask_arg if is_portrait else [])

            try:
                secs_fin = run_cmd(fin_cmd)
            except Exception as e:
                print(f"[FAIL:FIN] {tag}: {e}")
                with open(log_csv, "a", encoding="utf-8") as f:
                    f.write(f"{tag},FAIL_FIN,0.0\n")
                n_fail += 1
                continue

            # Copy NST meta JSON next to final so metrics can see hyperparams
            nst_meta   = out_img.with_suffix(".json")
            final_meta = out_fin.with_suffix(".json")
            if nst_meta.exists():
                try: shutil.copy2(nst_meta, final_meta)
                except Exception: pass

            with open(log_csv, "a", encoding="utf-8") as f:
                f.write(f"{tag},OK,{secs_nst+secs_fin:.3f}\n")
            print(f"[OK] {tag}")
            n_ok += 1

    print(f"Done. OK={n_ok}  SKIP={n_skip}  FAIL={n_fail}  FIXES={n_fix}")

if __name__ == "__main__":
    main()
