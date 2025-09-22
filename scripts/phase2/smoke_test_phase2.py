# scripts/phase2/smoke_test_phase2.py
"""
Phase-2 Smoke Test (fast)
- Runs a tiny plan (2 jobs) via mini_sweep_runner (CPU), seed=77.
- Verifies at least one output image and a log CSV exist.
Run:
  python scripts\phase2\smoke_test_phase2.py
"""
import os, sys, glob, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "out" / "phase2_smoke"

def run(cmd, env=None):
    print(">>", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, env=env)

def main():
    os.environ.setdefault("NST_SEED", "77")
    OUT.mkdir(parents=True, exist_ok=True)

    # Tiny sweep
    cmd = [
        sys.executable, str(ROOT/"scripts/phase2/mini_sweep_runner.py"),
        "--contents", str(ROOT/"data/content/Still_Life.jpg"),
        "--styles",   str(ROOT/"data/style/Monet.jpg"),
        "--sw", "15000,25000",
        "--tv", "0.0020",
        "--edge", "0.04",
        "--out-root", str(OUT),
        "--device", "cpu",
        "--seed", "77",
        "--limit", "2",
        "--resume"
    ]
    run(cmd, env={**os.environ, "NST_SEED": "77"})

    # Check for outputs (final images)
    finals = list((OUT/"finals").glob("*.jpg")) + list((OUT/"finals").glob("*.png"))
    if not finals:
        finals = list((OUT/"final").glob("*.jpg")) + list((OUT/"final").glob("*.png"))
    assert finals, "No final images found under out/phase2_smoke/finals (or final)"

    # Check for a log CSV
    log_candidates = [
        OUT/"log_phase2B.csv",
        OUT/"logs"/"phase2B_runs.csv",
        OUT/"logs"/"runs_log.csv",
        OUT/"runs_log.csv"
    ]
    has_log = any(p.exists() for p in log_candidates)
    assert has_log, "No Phase-2 log CSV found (expected one of phase2B_runs.csv, log_phase2B.csv, or logs/runs_log.csv)"

    print("\nSmoke test OK ✅")
    print("Sample final:", finals[0])
    print("Logs checked:", ", ".join(str(p) for p in log_candidates))

if __name__ == "__main__":
    main()
