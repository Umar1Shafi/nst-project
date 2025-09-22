
import os, json, argparse, csv, re
from pathlib import Path

def norm(p):
    if p is None: return None
    return str(p).replace("\\", "/")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    import json
    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    man_dir = Path(cfg["manifests_dir"])
    eval_dir = Path(cfg["eval_dir"]); (eval_dir / "tables").mkdir(parents=True, exist_ok=True)

    rows = []
    for jf in man_dir.glob("*.json"):
        try:
            obj = json.load(open(jf, "r", encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Skipping {jf}: {e}")
            continue

        row = {
            "manifest": norm(jf),
            "content": norm(obj.get("content")),
            "style": norm(obj.get("style")),
            "content_name": obj.get("content_name"),
            "style_name": obj.get("style_name"),
            "layers": obj.get("layers"),
            "style_weight": obj.get("style_weight"),
            "tv_weight": obj.get("tv_weight"),
            "edge_w": obj.get("edge_w"),
            "edge_face_down": obj.get("edge_face_down"),
            "face_preserve": obj.get("face_preserve"),
            "cpm": obj.get("cpm"),
            "is_portrait": obj.get("is_portrait"),
            "device": obj.get("device"),
            "sizes": obj.get("sizes"),
            "steps": obj.get("steps"),
            "optimizer": obj.get("optimizer"),
            "seed": obj.get("seed"),
            "out_img": norm(obj.get("out_img")),
            "out_final": norm(obj.get("out_final")),
        }
        # Also split sizes/steps into first and second stage if present "384,768"
        if row["sizes"] and isinstance(row["sizes"], str) and "," in row["sizes"]:
            a,b = row["sizes"].split(",")
            row["size_stage1"] = int(a)
            row["size_stage2"] = int(b)
        else:
            row["size_stage1"] = row["size_stage2"] = ""

        if row["steps"] and isinstance(row["steps"], str) and "," in row["steps"]:
            a,b = row["steps"].split(",")
            row["steps_stage1"] = int(a)
            row["steps_stage2"] = int(b)
        else:
            row["steps_stage1"] = row["steps_stage2"] = ""

        rows.append(row)

    out_csv = eval_dir / "tables" / "master_index.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"[OK] Wrote {out_csv} with {len(rows)} rows.")
        # Simple summary
        from collections import Counter
        sw = Counter([r["style_weight"] for r in rows])
        print("Style weights distribution:", sw)
    else:
        print("[WARN] No manifests parsed.")

if __name__ == "__main__":
    main()
