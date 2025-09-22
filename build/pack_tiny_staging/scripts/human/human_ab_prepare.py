# scripts/human_ab_prepare.py
import os, re, random, argparse, csv, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

METHODS = {"gatys","adain","wct"}
BACKBONES = {"vgg16","vgg19"}

def split_list(s):
    if not s: return []
    s = s.replace(";", ",")
    return [x.strip() for x in s.split(",") if x.strip()]

def stem_pair_from_name(stem):
    # Accept "<content>__<style>" or "<content>--<style>". Return (c, s) or (None, None).
    if "__" in stem:
        return stem.split("__", 1)
    if "--" in stem:
        return stem.split("--", 1)
    return None, None

def parse_mb(dirname):
    """Accept 'gatys_vgg16', 'gatys-vgg16', 'GATYS_VGG16', etc."""
    name = dirname.lower().replace("-", "_")
    m = re.match(r"^([a-z0-9_]+)_(vgg16|vgg19)$", name)
    if not m:
        return None, None
    method, backbone = m.group(1), m.group(2)
    # normalize known method tokens
    for cand in METHODS:
        if method.startswith(cand):
            method = cand
            break
    if method not in METHODS or backbone not in BACKBONES:
        return None, None
    return method, backbone

def find_runs(roots, methods, backbones, verbose=True):
    """
    Looks for files under: <root>/<method>_<backbone>/**/*.jpg
    Accepts any filename; tries to infer content/style from filename or from a sibling .json.
    Returns dict: keys[(content, style, backbone)][method] = path
    """
    keys = {}
    checked_dirs = 0
    found_files = 0

    for r in roots:
        rpath = Path(r)
        if not rpath.exists(): 
            continue
        for mb_dir in rpath.iterdir():
            if not mb_dir.is_dir(): 
                continue
            method, backbone = parse_mb(mb_dir.name)
            if method is None: 
                continue
            if methods and method not in methods: 
                continue
            if backbones and backbone not in backbones: 
                continue

            checked_dirs += 1
            # crawl for images
            for jpg in mb_dir.rglob("*.jpg"):
                stem = jpg.stem
                # skip obvious non-run composites
                if "contact_sheet" in stem or "grid" in stem:
                    continue
                c_stem, s_stem = stem_pair_from_name(stem)

                # try to recover from JSON if needed
                if c_stem is None or s_stem is None:
                    meta_path = jpg.with_suffix(".json")
                    if meta_path.exists():
                        try:
                            meta = json.loads(meta_path.read_text())
                            c_path = meta.get("content", "")
                            s_path = meta.get("style", "")
                            c_stem = c_stem or Path(c_path).stem
                            s_stem = s_stem or Path(s_path).stem
                        except Exception:
                            pass
                # last resort: keep but tag unknowns
                if not c_stem: c_stem = "content"
                if not s_stem: s_stem = "style"

                key = (c_stem, s_stem, backbone)
                keys.setdefault(key, {})[method] = str(jpg)
                found_files += 1

    if verbose:
        print(f"[scan] checked_dirs={checked_dirs}, found_files={found_files}, unique triplets={len(keys)}")
    return keys

def choose_pairs(keys, comparisons, total_pairs, backbone_filter, rng):
    """
    comparisons: list like ["gatys_vs_adain", "gatys_vs_wct", "adain_vs_wct"]
    Returns list of dicts with A/B filepaths+meta. A/B assignment randomized.
    """
    pools = {cmp: [] for cmp in comparisons}
    for (content, style, backbone), by_method in keys.items():
        if backbone_filter and backbone != backbone_filter:
            continue
        for cmp in comparisons:
            m1, m2 = cmp.split("_vs_")
            if m1 in by_method and m2 in by_method:
                pools[cmp].append({
                    "content": content, "style": style, "backbone": backbone,
                    "m1": m1, "m2": m2,
                    "p1": by_method[m1], "p2": by_method[m2],
                })

    # balance across comparisons
    out = []
    cmp_list = list(comparisons)
    share = max(1, total_pairs // max(1, len(cmp_list)))
    for cmp in cmp_list:
        rng.shuffle(pools[cmp])
        take = min(share, len(pools[cmp]))
        out.extend(pools[cmp][:take])

    if len(out) < total_pairs:
        leftovers = []
        for cmp in cmp_list:
            leftovers.extend(pools[cmp][share:])
        rng.shuffle(leftovers)
        out.extend(leftovers[:max(0, total_pairs - len(out))])

    out = out[:total_pairs]

    ab_pairs = []
    for i, it in enumerate(out, 1):
        if rng.random() < 0.5:
            A_path, B_path = it["p1"], it["p2"]
            A_method, B_method = it["m1"], it["m2"]
        else:
            A_path, B_path = it["p2"], it["p1"]
            A_method, B_method = it["m2"], it["m1"]
        ab_pairs.append({
            "pair_id": f"pair_{i:02d}",
            "content": it["content"], "style": it["style"], "backbone": it["backbone"],
            "comparison": f"{it['m1']}_vs_{it['m2']}",
            "A_path": A_path, "B_path": B_path,
            "A_method": A_method, "B_method": B_method,
        })
    return ab_pairs

def draw_side_by_side(a_path, b_path, save_path, margin=12, label=True):
    A = Image.open(a_path).convert("RGB")
    B = Image.open(b_path).convert("RGB")
    if A.size != B.size:
        B = B.resize(A.size, Image.Resampling.LANCZOS)
    w, h = A.size
    canvas = Image.new("RGB", (w*2 + margin*3, h + margin*2), (255,255,255))
    canvas.paste(A, (margin, margin))
    canvas.paste(B, (w + margin*2, margin))
    if label:
        d = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        d.text((margin+6, margin+6), "A", fill=(0,0,0), font=font)
        d.text((w + margin*2 + 6, margin+6), "B", fill=(0,0,0), font=font)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path, quality=95)

def write_manifest(pairs, out_csv):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "pair_id","content","style","backbone","comparison",
            "A_path","B_path","A_method","B_method","pair_image"
        ])
        w.writeheader()
        for row in pairs:
            w.writerow(row)

def write_ballots(pairs, rater_names, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    header = ["pair_id","pair_image","prompt","choice","confidence_1to5"]
    prompt = "Which looks more like the target style while keeping identity? (A/B, or T for tie)"
    # template
    tpl = Path(out_dir) / "ballot_TEMPLATE.csv"
    with open(tpl, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for p in pairs:
            w.writerow([p["pair_id"], p["pair_image"], prompt, "", ""])
    # per-rater copies
    for rn in rater_names:
        pth = Path(out_dir) / f"ballot_{rn}.csv"
        with open(pth, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header)
            for p in pairs:
                w.writerow([p["pair_id"], p["pair_image"], prompt, "", ""])

def write_rater_readme(out_path):
    txt = """How to rate (blinded A/B)
--------------------------------
Open each image listed in the CSV under the 'pair_image' column.
Each file shows two versions of the same content+style side-by-side, labeled A and B.

Answer per row:
  - 'choice': A or B (or T if you truly cannot decide).
  - 'confidence_1to5': optional (1=not confident, 5=very confident).

Save the CSV and send it back unchanged except for your two columns.
Do NOT rename files. Please do not try to infer which method is which.
Thanks!
"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="out/breadth",
                    help="One or more roots containing <method>_<backbone> folders (comma/semicolon-separated).")
    ap.add_argument("--methods", default="gatys,adain,wct")
    ap.add_argument("--backbones", default="vgg19,vgg16")
    ap.add_argument("--backbone_filter", default="",
                    help="If set (e.g., vgg16), only build pairs from that backbone.")
    ap.add_argument("--comparisons", default="gatys_vs_adain,gatys_vs_wct,adain_vs_wct",
                    help="Comma/semicolon list of method pairs like m1_vs_m2.")
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--rater_names", default="r1,r2,r3,r4,r5,r6")
    ap.add_argument("--out_root", default="out/human_ab")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    roots = split_list(args.roots)
    methods = set(split_list(args.methods))
    backbones = set(split_list(args.backbones))
    comparisons = split_list(args.comparisons)
    rater_names = split_list(args.rater_names)
    out_root = Path(args.out_root)
    pairs_dir = out_root / "pairs"
    ballots_dir = out_root / "ballots"
    manifest_csv = out_root / "pairs_manifest.csv"

    # 1) scan
    keys = find_runs(roots, methods, backbones, verbose=True)
    if not keys:
        print("[ERR] No runs found. Check --roots and directory layout (need <method>_<backbone> subfolders).")
        return

    # 2) choose pairs
    b_filter = args.backbone_filter.strip().lower() or ""
    pairs = choose_pairs(keys, comparisons, args.pairs, b_filter, rng)
    if not pairs:
        print("[ERR] No valid A/B pairs available for requested comparisons/backbone.")
        return

    # 3) compose side-by-sides
    for p in pairs:
        save_path = pairs_dir / f"{p['pair_id']}.jpg"
        draw_side_by_side(p["A_path"], p["B_path"], save_path)
        p["pair_image"] = str(save_path)

    # 4) write outputs
    write_manifest(pairs, manifest_csv)
    write_ballots(pairs, rater_names, ballots_dir)
    write_rater_readme(out_root / "README_for_raters.txt")

    # quick summary
    print(f"\nSaved: {manifest_csv}")
    print(f"Saved: {pairs_dir}")
    print(f"Saved: {ballots_dir} (template + {len(rater_names)} ballots)")
    print("Done.")

if __name__ == "__main__":
    main()
