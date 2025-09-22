# scripts/sweep_phase2.py
import os, sys, json, time, argparse
from pathlib import Path
import subprocess
import shlex

# ---------- Defaults (coarse) ----------
COARSE_SW   = [15000, 20000, 25000, 30000]
COARSE_TV   = [0.0020, 0.0026, 0.0032, 0.0040]
COARSE_EDGE = [0.04, 0.08, 0.12]

# Finisher edge gains (keep fixed in Phase-2 to isolate NST effects)
FIN_PORTRAIT = dict(edge_gain="0.32", edge_gain_face="0.18")
FIN_OTHER    = dict(edge_gain="0.36", edge_gain_face="0.22")

def run(cmd:list, cwd=None):
    t0=time.perf_counter()
    cp=subprocess.run(cmd, text=True, capture_output=True, cwd=cwd)
    dt=time.perf_counter()-t0
    if cp.returncode!=0:
        print("\n--- CMD ---\n", " ".join(shlex.quote(x) for x in cmd))
        print("\n--- STDOUT ---\n", cp.stdout)
        print("\n--- STDERR ---\n", cp.stderr)
        raise subprocess.CalledProcessError(cp.returncode, cmd)
    return dt, cp.stdout

def stem(p): return Path(p).stem

def make_tag(content_path, style_path, sw, tv, edge, fp):
    c = stem(content_path)
    s = stem(style_path).replace(" ", "_")
    return f"{c}__{s}__sw{int(round(sw/1000))}k_tv{tv:.4f}_edge{edge:.2f}_fp{fp:.2f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contents", default="data/content/portrait2.jpg;data/content/Still_Life.jpg;data/content/city.jpg;data/content/animal.jpg",
                    help="Semicolon list of content images (some may not exist).")
    ap.add_argument("--styles", default="data/style/Ukiyo-e_print.jpg;data/style/Monet.jpg",
                    help="Semicolon list of style images.")
    ap.add_argument("--portrait_contents", default="data/content/portrait2.jpg",
                    help="Semicolon list of contents to treat as portraits (face mask + higher face_preserve).")
    ap.add_argument("--out_root", default="out/phase2", help="Root folder for phase-2 artifacts.")
    ap.add_argument("--stage", choices=["coarse"], default="coarse")
    ap.add_argument("--resume", action="store_true", help="Skip runs whose FINAL already exists.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_workers", type=int, default=1, help="Set >1 only if you know your GPU has headroom.")
    args = ap.parse_args()

    contents = [c for c in args.contents.split(";") if c]
    styles   = [s for s in args.styles.split(";") if s]
    portrait_list = set(x.strip() for x in args.portrait_contents.split(";") if x.strip())

    root = Path(args.out_root)
    (root/"imgs").mkdir(parents=True, exist_ok=True)
    (root/"finals").mkdir(parents=True, exist_ok=True)
    (root/"logs").mkdir(parents=True, exist_ok=True)
    (root/"manifests").mkdir(parents=True, exist_ok=True)

    runs_log = root/"logs"/"runs_log.csv"
    if not runs_log.exists():
        runs_log.write_text("tag,content,style,sw,tv,edge_w,face_preserve,portrait,out_img,out_final,runtime_sec,status\n")

    # Parameter grid (coarse)
    sw_list   = COARSE_SW
    tv_list   = COARSE_TV
    edge_list = COARSE_EDGE

    # Build all planned runs
    plan=[]
    for c in contents:
        if not Path(c).exists():
            print(f"[skip] missing content: {c}")
            continue
        is_portrait = (c in portrait_list) or ("portrait" in stem(c).lower()) or ("face" in stem(c).lower())
        fp_values = ([0.90, 0.95, 1.00] if is_portrait else [0.70, 0.75])

        for s in styles:
            if not Path(s).exists():
                print(f"[skip] missing style: {s}")
                continue

            for sw in sw_list:
                for tv in tv_list:
                    for edge in edge_list:
                        for fp in fp_values:
                            tag = make_tag(c, s, sw, tv, edge, fp)
                            out_img   = str(root/"imgs"/f"{tag}.jpg")
                            out_final = str(root/"finals"/f"{tag}_final.jpg")
                            plan.append(dict(
                                tag=tag, content=c, style=s,
                                sw=sw, tv=tv, edge=edge, fp=fp,
                                out_img=out_img, out_final=out_final,
                                is_portrait=is_portrait
                            ))

    print(f"[plan] total runs: {len(plan)}")

    # Sequential by default (L-BFGS VRAM heavy)
    for i,job in enumerate(plan, 1):
        tag=job["tag"]; c=job["content"]; s=job["style"]
        sw=job["sw"]; tv=job["tv"]; edge=job["edge"]; fp=job["fp"]
        out_img=job["out_img"]; out_final=job["out_final"]
        is_portrait=job["is_portrait"]

        if args.resume and Path(out_final).exists():
            line=f'{tag},{c},{s},{sw},{tv},{edge},{fp},{int(is_portrait)},{out_img},{out_final},,SKIP\n'
            with open(runs_log,"a",newline="") as f: f.write(line)
            print(f"[{i}/{len(plan)}] SKIP {tag}")
            continue

        # 1) NST
        nst_cmd = [
            sys.executable, "scripts/core/nst_hybrid_ready.py",
            "--content", c,
            "--style", s,
            "--out", out_img,
            "--backbone", "vgg19",
            "--style_layers", "conv4_1,conv5_1",
            "--content_layer", "conv4_2",
            "--opt", "lbfgs",
            "--style_weight", str(sw),
            "--tv_weight", str(tv),
            "--sizes", "384,768",
            "--steps", "300,400",
            "--color_prematch_strength", "0.6",
            "--edge_w", str(edge),
            "--edge_face_down", "0.4",
            "--device", args.device,
            "--seed", str(args.seed),
            "--face_preserve", f"{fp:.2f}",
        ]
        if is_portrait:
            nst_cmd += ["--face_mask","auto"]

        # 2) Finisher (fixed in Phase-2)
        fin = FIN_PORTRAIT if is_portrait else FIN_OTHER
        fin_cmd = [
            sys.executable, "scripts/finish/hybrid_pipeline.py",
            "--inp", out_img,
            "--style", s,
            "--out", out_final,
            "--use_guided",
            "--bg_colors", "10",
            "--face_colors", "16",
            "--edge_gain", fin["edge_gain"],
            "--edge_gain_face", fin["edge_gain_face"],
            "--edge_thickness", "1",
            "--paper", "--paper_strength", "0.06",
            "--seed", str(args.seed),
        ]
        if is_portrait:
            # Use cached portrait mask if it exists; otherwise hybrid will ignore
            mask_guess = str(Path(c).with_name(Path(c).stem + "_face_mask.png"))
            if Path(mask_guess).exists():
                fin_cmd += ["--face_mask", mask_guess]

        # Manifest
        manifest = dict(
            tag=tag, content=c, style=s, is_portrait=is_portrait,
            params=dict(sw=sw, tv=tv, edge_w=edge, face_preserve=fp),
            fixed=dict(backbone="vgg19", style_layers="conv4_1,conv5_1", content_layer="conv4_2",
                       sizes="384,768", steps="300,400", optimizer="lbfgs",
                       color_prematch=0.6, edge_face_down=0.4, seed=args.seed, device=args.device),
            out=dict(nst=out_img, final=out_final),
            cmd=dict(nst=" ".join(shlex.quote(x) for x in nst_cmd),
                     fin=" ".join(shlex.quote(x) for x in fin_cmd))
        )
        (root/"manifests"/f"{tag}.json").write_text(json.dumps(manifest, indent=2))

        print(f"[{i}/{len(plan)}] RUN {tag}")
        try:
            t1,_=run(nst_cmd)
            t2,_=run(fin_cmd)
            status="OK"; rt=round(t1+t2,3)
        except subprocess.CalledProcessError:
            status="FAIL"; rt=""
        with open(runs_log,"a",newline="") as f:
            f.write(f'{tag},{c},{s},{sw},{tv},{edge},{fp},{int(is_portrait)},{out_img},{out_final},{rt},{status}\n')

    print(f"Done. Log: {runs_log}")

if __name__ == "__main__":
    main()
