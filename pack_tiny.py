#!/usr/bin/env python3
"""
Tiny packer for NST submission.

Creates a small, reproducible archive with:
  - scripts/, configs/, requirements/
  - README.md, LICENSE, CITATION.cff (and Model_Asset_Licenses.md if present)
  - minimal data/ (1–2 small content + 1–2 small style images)
  - minimal report/ (key CSVs + 1 final docx/pdf if present)
  - minimal out/ (handful of representative PNG/JPGs)
  - MANIFEST.json (what got packed, sizes, warnings)

Usage:
  python pack_tiny.py
  python pack_tiny.py --max-mb 40 --zip-name nst_tiny_submit.zip
  python pack_tiny.py --content data/content/portrait2.jpg --style data/style/Ukiyo-e_print.jpg

Notes:
  - Enforces a total size budget and shows the heaviest files if exceeded.
  - Canonicalizes duplicate Ukiyo-e style asset (prefers underscore variant).
"""

import argparse, json, os, re, shutil, sys, time
from pathlib import Path

# -------------------------
# Defaults / knobs
# -------------------------
WHITELIST_DIRS = [
    "scripts",
    "configs",
    "requirements",
]

WHITELIST_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "Model_Asset_Licenses.md",  # optional
]

# Curate small samples automatically (can be overridden via CLI)
DATA_CONTENT_DIR = Path("data/content")
DATA_STYLE_DIR   = Path("data/style")

# Minimal out/ and report/ selection rules
REPORT_DIR = Path("report")
OUT_DIR    = Path("out")

# File filters
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
CSV_EXTS = {".csv"}
DOC_EXTS = {".pdf", ".docx"}

# Limits
DEFAULT_MAX_MB            = 50      # total archive budget
MAX_OUT_IMAGES            = 12      # cap representative outputs
MAX_REPORT_CSVS           = 6
MAX_REPORT_DOCS           = 1
AUTO_SELECT_MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB per auto-selected file

# Output locations
DIST_DIR = Path("dist")
BUILD_DIR = Path("build/pack_tiny_staging")

# Canonicalization for Ukiyo-e
UKIYO_SPACE = DATA_STYLE_DIR / "Ukiyo-e print.jpg"
UKIYO_UNDER = DATA_STYLE_DIR / "Ukiyo-e_print.jpg"

# -------------------------
# Helpers
# -------------------------
def log(msg): print(f"[pack] {msg}")

def bytes_to_mb(n): return n / (1024*1024)

def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def copy_rel(src: Path, dst_root: Path, base: Path):
    rel = src.relative_to(base)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst

def glob_small_images(root: Path, limit: int, max_bytes: int):
    files = []
    if not root.exists(): return files
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            if sz <= max_bytes:
                files.append((sz, p))
    # Prefer smallest first
    files.sort(key=lambda t: t[0])
    return [p for _, p in files[:limit]]

def choose_report_files():
    csvs, docs = [], []
    if REPORT_DIR.exists():
        for p in sorted(REPORT_DIR.rglob("*")):
            if not p.is_file(): continue
            ext = p.suffix.lower()
            if ext in CSV_EXTS:
                csvs.append(p)
            elif ext in DOC_EXTS:
                docs.append(p)
    # Heuristic: prefer CSVs with "phase3" or "advanced" in name
    def weight_csv(p: Path):
        name = p.name.lower()
        return (("phase3" in name) or ("advanced" in name), name)
    csvs.sort(key=lambda p: weight_csv(p), reverse=True)

    # Prefer a single "final" doc if present
    def weight_doc(p: Path):
        name = p.name.lower()
        return (("final" in name) or ("report" in name), name)
    docs.sort(key=lambda p: weight_doc(p), reverse=True)

    return csvs[:MAX_REPORT_CSVS], docs[:MAX_REPORT_DOCS]

def choose_out_images():
    picks = []
    if not OUT_DIR.exists(): return picks
    # Prefer phase3, then phase2 sxs, then anything else, smallest first
    candidates = []
    for p in OUT_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            tier = 2
            low = str(p).lower()
            if "phase3" in low: tier = 0
            elif "phase2" in low or "sxs" in low: tier = 1
            candidates.append((tier, sz, p))
    candidates.sort(key=lambda t: (t[0], t[1]))
    for _, _, p in candidates:
        picks.append(p)
        if len(picks) >= MAX_OUT_IMAGES:
            break
    return picks

def canonicalize_ukiyoe(staging_root: Path, manifest: dict):
    """Ensure only one Ukiyo-e style file exists in the pack; prefer underscore variant."""
    # If both exist in source, keep underscore as canonical in staging
    # and rewrite the spaced variant path to point to the underscore file.
    kept = None
    if UKIYO_UNDER.exists():
        # Copy underscore file (if not already copied)
        dst = copy_rel(UKIYO_UNDER, staging_root, Path("."))
        kept = dst
        manifest["notes"].append("Kept data/style/Ukiyo-e_print.jpg as canonical.")
    elif UKIYO_SPACE.exists():
        dst = copy_rel(UKIYO_SPACE, staging_root, Path("."))
        kept = dst
        manifest["notes"].append("Only spaced Ukiyo-e present; consider renaming to underscore in source.")
    # If both existed, add a tiny placeholder symlink/copy for compatibility
    if kept and UKIYO_SPACE.exists() and UKIYO_UNDER.exists():
        # Also place a duplicate under the spaced name (tiny copy for compatibility)
        spaced_dst = staging_root / UKIYO_SPACE
        spaced_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(kept, spaced_dst)
        manifest["notes"].append("Duplicate placed at data/style/Ukiyo-e print.jpg for compatibility (both names).")

def check_author(staging_root: Path, manifest: dict):
    """Warn if LICENSE/CITATION authors do not match 'Mohammad Bin Shafi'."""
    expected = re.compile(r"(?i)mohammad\s+bin\s+shafi")
    lic = staging_root / "LICENSE"
    cit = staging_root / "CITATION.cff"
    for f in [lic, cit]:
        if f.exists():
            try:
                t = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if not expected.search(t):
                manifest["warnings"].append(f"{f} does not appear to contain author 'Mohammad Bin Shafi'.")

def human(num_bytes: int) -> str:
    mb = bytes_to_mb(num_bytes)
    if mb < 1:
        return f"{num_bytes} B"
    return f"{mb:.2f} MB"

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-name", default="nst_tiny_submit.zip", help="Output zip name inside dist/")
    ap.add_argument("--max-mb", type=int, default=DEFAULT_MAX_MB, help="Max total size (MB)")
    ap.add_argument("--content", action="append", default=[], help="Explicit content image(s) to include")
    ap.add_argument("--style", action="append", default=[], help="Explicit style image(s) to include")
    ap.add_argument("--dry-run", action="store_true", help="Don't create zip; just print manifest summary")
    args = ap.parse_args()

    max_bytes = args.max_mb * 1024 * 1024

    ensure_clean_dir(BUILD_DIR)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "zip_name": args.zip_name,
        "max_mb": args.max_mb,
        "files": [],
        "warnings": [],
        "notes": [],
        "totals": {"bytes": 0, "files": 0},
    }

    # 1) Required top-level files
    for f in WHITELIST_FILES:
        p = Path(f)
        if p.exists():
            dst = copy_rel(p, BUILD_DIR, Path("."))
            manifest["files"].append(str(dst.relative_to(BUILD_DIR)))
        else:
            if f in ["README.md", "LICENSE", "CITATION.cff"]:
                manifest["warnings"].append(f"Missing required file: {f}")

    # 2) Key directories
    for d in WHITELIST_DIRS:
        p = Path(d)
        if not p.exists(): 
            manifest["warnings"].append(f"Missing directory: {d}")
            continue
        for src in p.rglob("*"):
            if src.is_file():
                # skip cache/junk
                low = str(src).lower()
                if any(tag in low for tag in [".ipynb_checkpoints", "__pycache__", ".pytest_cache"]):
                    continue
                dst = copy_rel(src, BUILD_DIR, Path("."))
                manifest["files"].append(str(dst.relative_to(BUILD_DIR)))

    # 3) Minimal data/ (content + style)
    # If user provided explicit assets, take those; else auto-pick small files.
    selected_content = [Path(x) for x in args.content] if args.content else glob_small_images(DATA_CONTENT_DIR, limit=2, max_bytes=AUTO_SELECT_MAX_FILE_SIZE)
    selected_style   = [Path(x) for x in args.style]   if args.style   else glob_small_images(DATA_STYLE_DIR,   limit=2, max_bytes=AUTO_SELECT_MAX_FILE_SIZE)

    for p in selected_content + selected_style:
        if p.exists() and p.is_file():
            dst = copy_rel(p, BUILD_DIR, Path("."))
            manifest["files"].append(str(dst.relative_to(BUILD_DIR)))
        else:
            manifest["warnings"].append(f"Requested data file not found: {p}")

    # Canonicalize Ukiyo-e duplicates if present
    canonicalize_ukiyoe(BUILD_DIR, manifest)

    # 4) Minimal report/ (CSVs + a final doc if present)
    csvs, docs = choose_report_files()
    for p in (csvs + docs):
        if p.exists():
            dst = copy_rel(p, BUILD_DIR, Path("."))
            manifest["files"].append(str(dst.relative_to(BUILD_DIR)))

    # 5) Minimal out/ (few representative images)
    for p in choose_out_images():
        dst = copy_rel(p, BUILD_DIR, Path("."))
        manifest["files"].append(str(dst.relative_to(BUILD_DIR)))

    # 6) Compute totals + prune if over budget
    total_bytes = 0
    sized = []
    for rel in manifest["files"]:
        fp = BUILD_DIR / rel
        try:
            sz = fp.stat().st_size
        except OSError:
            sz = 0
        total_bytes += sz
        sized.append((sz, rel))

    manifest["totals"]["bytes"] = total_bytes
    manifest["totals"]["files"] = len(manifest["files"])

    if total_bytes > max_bytes:
        sized.sort(reverse=True)  # largest first
        over = total_bytes - max_bytes
        manifest["warnings"].append(f"Over budget by {human(over)}; consider removing these heavy files first:")
        # list top 10 heavy files
        for sz, rel in sized[:10]:
            manifest["warnings"].append(f"  - {rel} ({human(sz)})")

    # 7) Author check warnings
    check_author(BUILD_DIR, manifest)

    # 8) Write manifest
    manifest_path = BUILD_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 9) Zip (unless dry run)
    if args.dry_run:
        log("Dry run. Manifest summary:")
        log(json.dumps(manifest, indent=2))
        return

    zip_base = DIST_DIR / Path(args.zip_name).with_suffix("")
    shutil.make_archive(str(zip_base), "zip", root_dir=str(BUILD_DIR))
    final_zip = zip_base.with_suffix(".zip")
    log(f"Created: {final_zip} ({human(final_zip.stat().st_size)})")

    # Final hints
    if total_bytes > max_bytes:
        log("WARNING: Archive content exceeds the max budget. See MANIFEST.json warnings.")
    else:
        log("Within size budget.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
