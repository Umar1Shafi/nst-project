import argparse, os, sys, subprocess, shlex, re
from pathlib import Path

try:
    import yaml  # pip install pyyaml if missing
except ImportError:
    print("Missing dependency: pyyaml. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "-", s.strip().lower())


def guess_style_from_filename(p: Path) -> str:
    name = p.stem.lower()
    for s in ("anime", "cyberpunk", "cinematic", "noir"):
        if s in name:
            return s
    return "cinematic"  # safe default


def collect_jobs_from_jobs_cfg(cfg):
    # cfg is either a list[dict] or {jobs: list[dict]}
    jobs = None
    if isinstance(cfg, dict):
        jobs = cfg.get("jobs") or cfg.get("tasks")
    if jobs is None:
        jobs = cfg
    if not isinstance(jobs, list):
        raise SystemExit("Config must be a list of jobs or {jobs:[...]} or {tasks:[...]}")
    norm = []
    for j in jobs:
        # required keys: style, input, output ; optional: variant, route_args
        if not all(k in j for k in ("style", "input", "output")):
            raise SystemExit("Each job needs 'style', 'input', 'output'. Optional: 'variant', 'route_args'")
        norm.append({
            "style": j["style"],
            "input": j["input"],
            "output": j["output"],
            "variant": j.get("variant"),
            "route_args": j.get("route_args", "")
        })
    return norm


def collect_jobs_from_style_pack(cfg, cfg_path: Path, args):
    """
    Accepts style preset packs such as configs/phase3/anime_presets.yaml.
    We only need variant *names*; actual parameterization is handled inside route_and_run via registry.
    Heuristics:
      - style: from cfg['style'] or filename
      - variants: prefer cfg['variants'] (list), else keys under cfg['presets'] (dict), else
                  all top-level keys whose values are dicts (treated as variants)
    Inputs:
      - portrait input defaults to data/content/portrait2.jpg
      - scene input defaults to data/content/street.jpg (cyberpunk may prefer city.jpg)
      - can be overridden via CLI flags
    """
    style = (cfg.get("style") if isinstance(cfg, dict) else None) or guess_style_from_filename(cfg_path)

    # figure out variant names present in file
    variants = []
    if isinstance(cfg, dict):
        if isinstance(cfg.get("variants"), list):
            variants = cfg["variants"]
        elif isinstance(cfg.get("presets"), dict):
            variants = list(cfg["presets"].keys())
        else:
            # fall back: any top-level dict-like keys except obvious metadata
            meta = {"style", "description", "notes", "meta"}
            for k, v in cfg.items():
                if k not in meta and isinstance(v, dict):
                    variants.append(k)
    if not variants:
        # no explicit variants; run a single default (router will choose defaults)
        variants = [None]

    # filter by CLI --variants, if provided
    if args.variants and args.variants.lower() != "all":
        want = [v.strip() for v in args.variants.split(",")]
        variants = [v for v in variants if (v in want) or (v is None and "default" in [w.lower() for w in want])]

    # choose input by subject expectation
    # (anime ≈ portrait; others ≈ scene; override via CLI)
    portrait_input = args.portrait_input or "data/content/portrait2.jpg"
    # prefer city for cyberpunk if present
    default_scene = "data/content/city.jpg" if style == "cyberpunk" and Path("data/content/city.jpg").exists() else "data/content/street.jpg"
    scene_input = args.scene_input or default_scene

    subject_map = {"anime": "portrait", "cyberpunk": "scene", "cinematic": "scene", "noir": "scene"}
    subject = subject_map.get(style, "scene")
    inp = portrait_input if subject == "portrait" else scene_input

    out_root = Path(args.out_root or "out/phase3")
    out_dir = out_root / style

    jobs = []
    for v in variants:
        # output pattern: out/phase3/<style>/<inputStem>__<style>__<variant|default>.png
        stem = Path(inp).stem
        vtag = slug(v) if isinstance(v, str) else "default"
        out = out_dir / f"{stem}__{style}__{vtag}.png"
        jobs.append({"style": style, "input": inp, "output": str(out), "variant": v, "route_args": ""})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML file: either a jobs list or a style preset pack")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--variants", default="all", help="Comma list of variants to run, or 'all' (default)")
    ap.add_argument("--portrait-input", dest="portrait_input", default=None)
    ap.add_argument("--scene-input", dest="scene_input", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="Skip jobs whose output file already exists")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        if cfg is None:
            raise SystemExit(f"Empty YAML file: {cfg_path}")

    jobs = None
    # Try jobs mode first
    try:
        jobs = collect_jobs_from_jobs_cfg(cfg)
    except SystemExit:
        # Not jobs-mode; try style-pack mode
        jobs = collect_jobs_from_style_pack(cfg, cfg_path, args)

    # Execute via router
    for j in jobs:
        # skip if requested and file already exists
        if args.skip_existing and Path(j["output"]).is_file():
            print(f"[skip] {j['output']} (exists)")
            continue
        cmd = [args.python, "scripts/phase3/route_and_run.py", "--style", j["style"], "-i", j["input"], "-o", j["output"]]
        if j["variant"]:
            cmd += ["--variant", j["variant"]]
        if j["route_args"]:
            cmd += shlex.split(j["route_args"], posix=(os.name != "nt"))

        if not Path(j["input"]).is_file():
            raise SystemExit(f"Input not found: {j['input']}")

        print(">>>", " ".join(cmd))
        if not args.dry_run:
            Path(j["output"]).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()



