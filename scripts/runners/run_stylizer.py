import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLUGIN_MAP = {
    "anime":     ROOT / "scripts" / "plugins" / "anime"     / "anime_stylize_v2.py",
    "cinematic": ROOT / "scripts" / "plugins" / "cinematic" / "cinematic_stylize_v5.py",
    "cyberpunk": ROOT / "scripts" / "plugins" / "cyberpunk" / "cyberpunk_stylize_v3.py",
    "noir":      ROOT / "scripts" / "plugins" / "noir"      / "noir_stylize.py",
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stylizer", choices=list(PLUGIN_MAP), required=True)
    p.add_argument("-i","--input", required=True)
    p.add_argument("-o","--output", required=True)
    p.add_argument("--stylizer-args", default="", help="Raw args string forwarded to the stylizer")
    args = p.parse_args()

    script = PLUGIN_MAP[args.stylizer]
    if not script.exists():
        sys.exit(f"Stylizer script missing: {script}")

    out = Path(args.output)
    if "phase3" not in out.parts:
        out = ROOT / "out" / "phase3" / args.stylizer / out.name
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(script), "-i", args.input, "-o", str(out)]
    if args.stylizer_args:
        cmd += args.stylizer_args.split()

    print("Running:", " ".join(cmd))
    cp = subprocess.run(cmd)
    sys.exit(cp.returncode)

if __name__ == "__main__":
    main()

