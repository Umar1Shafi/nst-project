# scripts/human_ab_score.py
import os, argparse, json, math, glob
from pathlib import Path
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

CHOICES = ["A","B","T"]

def fleiss_kappa(counts: np.ndarray):
    """
    counts: [N_items, n_categories] integer votes per item (e.g., [A,B,T])
    Returns Fleiss' kappa for multiple raters, multiple categories.
    """
    N, k = counts.shape
    n = counts.sum(axis=1)
    if not np.all(n == n[0]):
        # allow variable raters: use per-item n_i in formula
        P_i = ( (counts*(counts-1)).sum(axis=1) / (n*(n-1) + 1e-12) )
    else:
        nn = float(n[0])
        P_i = ( (counts*(counts-1)).sum(axis=1) / (nn*(nn-1) + 1e-12) )

    P_bar = P_i.mean()
    p_j = counts.sum(axis=0) / counts.sum()
    P_e = (p_j**2).sum()
    kappa = (P_bar - P_e) / (1.0 - P_e + 1e-12)
    return float(kappa), float(P_bar), float(P_e)

def bootstrap_kappa(counts, B=2000, seed=123):
    rng = np.random.default_rng(seed)
    N = counts.shape[0]
    kappas = []
    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        k,_,_ = fleiss_kappa(counts[idx])
        kappas.append(k)
    arr = np.sort(np.asarray(kappas, dtype=float))
    lo = float(np.percentile(arr, 2.5))
    hi = float(np.percentile(arr,97.5))
    return lo, hi

def load_ballots(ballots_dir: Path):
    rows = []
    for p in sorted(ballots_dir.glob("*.csv")):
        rater = re.sub(r"\.csv$","",p.name)
        # accept "ballot_r3.csv" or "r3.csv"
        m = re.search(r"(r\d+)$", rater)
        rater = m.group(1) if m else rater
        df = pd.read_csv(p)
        # required cols
        need = {"pair_id","choice","score"}
        if not need.issubset(df.columns):
            raise ValueError(f"{p} missing columns {need - set(df.columns)}")
        df["rater"] = rater
        rows.append(df[["pair_id","rater","choice","score","notes"] if "notes" in df.columns else ["pair_id","rater","choice","score"]])
    if not rows:
        raise SystemExit(f"No CSV ballots in {ballots_dir}")
    allb = pd.concat(rows, ignore_index=True)
    # clean
    allb["choice"] = allb["choice"].str.strip().str.upper()
    allb = allb[allb["choice"].isin(CHOICES)]
    allb["score"] = pd.to_numeric(allb["score"], errors="coerce").clip(1,5)
    return allb

def decode_pairs_manifest(manifest_path: Path):
    if not manifest_path.exists():
        return None
    df = pd.read_csv(manifest_path)
    # Try to infer method/backbone columns if present
    cols = df.columns.str.lower().tolist()
    # normalize lowercase names
    lower = {c:c for c in df.columns}
    df.columns = [c.lower() for c in df.columns]
    # best-effort mapping
    for side in ["a","b"]:
        if f"{side}_method" not in df.columns:
            # try parse from path if available
            pcol = f"{side}_path" if f"{side}_path" in df.columns else (f"{side}_img" if f"{side}_img" in df.columns else None)
            if pcol and pcol in df.columns:
                df[f"{side}_method"] = df[pcol].fillna("").apply(lambda s: _parse_method_from_path(s))
            else:
                df[f"{side}_method"] = ""
        if f"{side}_backbone" not in df.columns:
            df[f"{side}_backbone"] = ""
    # restore original names dict (not needed further)
    return df

def _parse_method_from_path(path):
    # out/breadth/gatys_vgg19/… or breadth\adain_vgg16\…
    b = os.path.basename(os.path.dirname(path))
    return b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ballots", required=True, help="Folder with ballot CSVs (ballot_r1.csv, …)")
    ap.add_argument("--pairs_manifest", default="out/human_ab/pairs_manifest.csv")
    ap.add_argument("--out", default="out/human_ab/results")
    ap.add_argument("--make_plots", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    ballots = load_ballots(Path(args.ballots))
    ballots.to_csv(out_dir/"all_ballots.csv", index=False)

    # Per-pair tallies
    pivot = (ballots
             .groupby(["pair_id","choice"])
             .size().unstack(fill_value=0)
             .reindex(columns=CHOICES, fill_value=0))
    pivot["n"] = pivot.sum(axis=1)
    pivot["winner"] = pivot.apply(lambda r: "A" if r["A"]>r["B"] and r["A"]>r["T"] else ("B" if r["B"]>r["A"] and r["B"]>r["T"] else ("T" if r["T"]>=max(r["A"],r["B"]) else "—")), axis=1)
    scores = ballots.groupby("pair_id")["score"].mean().rename("mean_score").round(2)
    summary = pivot.join(scores).reset_index()
    summary = summary.sort_values("pair_id")
    summary.to_csv(out_dir/"summary.csv", index=False)

    # Agreement (Fleiss' kappa)
    C = summary[CHOICES].to_numpy(dtype=np.int64)
    kappa, Pbar, Pe = fleiss_kappa(C)
    lo, hi = bootstrap_kappa(C)
    with open(out_dir/"agreement.json","w") as f:
        json.dump({
            "fleiss_kappa": round(kappa,3),
            "kappa_ci95": [round(lo,3), round(hi,3)],
            "Pbar": round(Pbar,3), "Pe": round(Pe,3),
            "N_pairs": int(C.shape[0]), "raters_per_pair_mean": float(summary["n"].mean())
        }, f, indent=2)

    # Join with manifest (optional) to attribute winners to methods/backbones
    manifest = decode_pairs_manifest(Path(args.pairs_manifest))
    if manifest is not None and "pair_id" in manifest.columns:
        # Winner side metadata
        m = manifest.set_index("pair_id")
        def side_meta(row):
            side = row["winner"].lower()
            if side not in ("a", "b"):
                return pd.Series({
                    "pair_id": row["pair_id"],
                    "winner_method": "",
                    "winner_backbone": ""
                })
            return pd.Series({
                "pair_id": row["pair_id"],
                "winner_method": m.loc[row["pair_id"], f"{side}_method"] if f"{side}_method" in m.columns else "",
                "winner_backbone": m.loc[row["pair_id"], f"{side}_backbone"] if f"{side}_backbone" in m.columns else ""
            })

        summary = summary.merge(summary.apply(side_meta, axis=1), left_on="pair_id", right_on="pair_id", how="left")


        summary.to_csv(out_dir/"summary_with_methods.csv", index=False)

        # Method win table (if available)
        if "winner_method" in summary.columns:
            method_wins = (summary.assign(method_id=summary["winner_method"].fillna(""))
                           .groupby("method_id")["pair_id"].count().rename("wins").sort_values(ascending=False))
            method_wins.to_csv(out_dir/"method_wins.csv")
        if set(["winner_method","winner_backbone"]).issubset(summary.columns):
            combo = summary.assign(combo=(summary["winner_method"].fillna("")+"_"+summary["winner_backbone"].fillna("")))
            combo_wins = combo.groupby("combo")["pair_id"].count().rename("wins").sort_values(ascending=False)
            combo_wins.to_csv(out_dir/"method_backbone_wins.csv")

    # Optional quick plots
    if args.make_plots:
        (summary.set_index("pair_id")[["A","B","T"]]
                 .plot(kind="bar", stacked=True, figsize=(10,5)))
        plt.title("Votes per pair")
        plt.tight_layout(); plt.savefig(out_dir/"plots/votes_per_pair.png"); plt.close()

        totals = summary[["A","B","T"]].sum()
        totals.plot(kind="bar", figsize=(4,4))
        plt.title("Overall votes"); plt.tight_layout()
        plt.savefig(out_dir/"plots/overall_votes.png"); plt.close()

    print(f"Saved:\n- {out_dir/'all_ballots.csv'}\n- {out_dir/'summary.csv'}")
    if (out_dir/"summary_with_methods.csv").exists():
        print(f"- {out_dir/'summary_with_methods.csv'}")
    print(f"- {out_dir/'agreement.json'}")
    if args.make_plots:
        print(f"- {out_dir/'plots'}")
if __name__ == "__main__":
    main()
