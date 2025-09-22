# scripts/eval/make_shortlist.py  (permissive version)
import pandas as pd
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IDX = ROOT / "reports" / "phase2_eval" / "index.csv"
OUT = ROOT / "reports" / "phase2_eval" / "shortlist.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Tunables
TARGET_PER_CONTENT = 12
MIN_PER_BUCKET = 2

STYLE_BINS = [-np.inf, 12000, 16000, 22000, np.inf]
STYLE_LABELS = ["<12k", "12–16k", "16–22k", ">22k"]
TV_BINS = [-np.inf, 0.0018, 0.0024, 0.0030, np.inf]
TV_LABELS = ["<0.0018", "0.0018–0.0024", "0.0024–0.0030", ">0.0030"]

def coerce_float(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    try:
        # If multiple values were logged (e.g., "0.0022,0.0026"), take first numeric
        if isinstance(s, str):
            for tok in s.replace(";",",").split(","):
                tok = tok.strip()
                try: return float(tok)
                except: pass
            return np.nan
        return float(s)
    except:
        return np.nan

def main():
    df = pd.read_csv(IDX)

    # If content_type is missing, default to 'other'
    if "content_type" not in df.columns:
        df["content_type"] = "other"
    df["content_type"] = df["content_type"].fillna("other")

    # Keep rows that have at least SOME evidence of an output/run
    keep_mask = (
        df["run_id"].notna() &
        (
            df.get("final_output", pd.Series([np.nan]*len(df))).notna()
            | df.get("manifest_file", pd.Series([np.nan]*len(df))).notna()
        )
    )
    df = df[keep_mask].copy()
    if df.empty:
        print("Index had no usable rows (even after permissive filter).")
        OUT.write_text("")
        return

    # Numeric buckets for variety sampling
    df["style_weight_f"] = df.get("style_weight", pd.Series([np.nan]*len(df))).apply(coerce_float)
    df["tv_weight_f"] = df.get("tv_weight", pd.Series([np.nan]*len(df))).apply(coerce_float)
    df["sw_bucket"] = pd.cut(df["style_weight_f"], bins=STYLE_BINS, labels=STYLE_LABELS)
    df["tv_bucket"] = pd.cut(df["tv_weight_f"], bins=TV_BINS, labels=TV_LABELS)

    # Prefer rows that have meta (helpful labels), but DO NOT require them
    meta_cols = [c for c in ["steps","sizes","optimizer","style_layers_bulk","style_layers_addon"] if c in df.columns]
    df["has_meta"] = df[meta_cols].notna().sum(axis=1) if meta_cols else 0

    shortlist_rows = []
    for ctype, g in df.groupby("content_type"):
        g = g.sort_values(["has_meta"], ascending=False)

        picks = []
        # ensure some diversity across buckets
        if "sw_bucket" in g.columns and "tv_bucket" in g.columns:
            for (swb, tvb), gg in g.groupby(["sw_bucket","tv_bucket"]):
                if len(picks) >= TARGET_PER_CONTENT: break
                if gg.empty: continue
                picks.extend(gg.head(MIN_PER_BUCKET).to_dict("records"))
                if len(picks) >= TARGET_PER_CONTENT: break

        # top-up if needed
        if len(picks) < TARGET_PER_CONTENT:
            extra = g[~g["run_id"].isin([r["run_id"] for r in picks])].head(TARGET_PER_CONTENT - len(picks))
            picks.extend(extra.to_dict("records"))

        shortlist_rows.extend(picks)

    out_df = pd.DataFrame(shortlist_rows).drop_duplicates(subset=["run_id"])
    out_df.to_csv(OUT, index=False)
    print(f"Wrote shortlist: {OUT} ({len(out_df)} rows)")

if __name__ == "__main__":
    main()
