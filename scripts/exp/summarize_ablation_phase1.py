# scripts/exp/summarize_ablation_phase1.py
#!/usr/bin/env python
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="out/ablation_phase1/ablation_metrics.csv")
    ap.add_argument("--out-md", default="out/ablation_phase1/summary.md")
    args = ap.parse_args()

    # pandas for grouping/markdown; we keep a simple fallback if markdown isn't available
    import pandas as pd

    df = pd.read_csv(args.csv)

    # ensure numeric cols are numeric
    for col in ["lpips", "psnr", "ssim", "elapsed_sec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # group and aggregate (mean), then sort by quality then speed
    g = (df.groupby(["backbone", "opt", "layers_tag"], as_index=False)
            .agg(lpips=("lpips","mean"),
                 psnr=("psnr","mean"),
                 ssim=("ssim","mean"),
                 elapsed_sec=("elapsed_sec","mean"))
            .sort_values(["lpips","elapsed_sec"], ascending=[True, True]))

    # write markdown table
    md_lines = []
    md_lines.append("# Phase-1 Ablation Summary")
    md_lines.append("")
    md_lines.append("Sorted by LPIPS (lower is better), then elapsed time:")
    md_lines.append("")

    try:
        table_md = g.to_markdown(index=False, floatfmt=".4f")
    except Exception:
        # Fallback if 'tabulate' isn't installed or older pandas
        table_md = "```\n" + g.to_string(index=False,
                                         formatters={
                                            "lpips": lambda x: f"{x:.4f}",
                                            "psnr": lambda x: f"{x:.3f}",
                                            "ssim": (lambda x: "" if pd.isna(x) else f"{x:.4f}"),
                                            "elapsed_sec": lambda x: f"{x:.3f}",
                                         }) + "\n```"

    md_lines.append(table_md)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[write] {out_md}")

    # print top-3 to console (pretty)
    print("\nTop-3 configs by LPIPS:")
    print(g.head(3).to_string(
        index=False,
        formatters={
            "lpips": lambda x: f"{x:.4f}",
            "psnr": lambda x: f"{x:.3f}",
            "ssim": (lambda x: "" if pd.isna(x) else f"{x:.4f}"),
            "elapsed_sec": lambda x: f"{x:.3f}",
        })
    )

if __name__ == "__main__":
    main()
