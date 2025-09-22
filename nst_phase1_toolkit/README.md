
# NST Phase 1 — Sweep Evaluation Toolkit

This toolkit turns your sweep outputs into **usable insights** by building a master index, making comparison grids, computing metrics, and suggesting a baseline.

## Folder Expectations

Your project (adjust paths if different):

```
out/
  phase2/
    finals/
    imgs/
    logs/
    manifests/
```

We'll write all evaluation artifacts to:

```
out/phase2_eval/
  tables/
  figures/
  grids/
  metrics/
```

## 1) Configure

Copy `config_example.json` to `config.json` and adjust paths if needed:

```bash
cp config_example.json config.json
```

Key fields:
- `phase2_dir`: the root for `finals`, `imgs`, `logs`, `manifests`.
- `eval_dir`: where we write evaluation artifacts.
- `content_root_override` (optional): if manifest `content` paths are relative to a different root.
- `style_root_override` (optional): similar for style images.

## 2) Build the Master Index

```bash
python 01_build_master_index.py --config config.json
```

Outputs:
- `out/phase2_eval/tables/master_index.csv` — one row per run with all hyperparams + file paths.
- Summary counts printed to console.

## 3) Make Comparison Grids

Pick a few representative (content, style) pairs and compare across hyperparameters.

```bash
python 02_make_grids.py --config config.json   --content animal,portrait2,city   --style Matisse,Monet   --by style_weight --topk 12
```

Outputs:
- `out/phase2_eval/grids/*png` — grids with captions (hyperparams as titles).

## 4) Compute Metrics (SSIM, optional LPIPS)

Install metrics libs first (once):

```bash
pip install scikit-image
# (Optional) LPIPS:
pip install lpips torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
```

Run:
```bash
python 03_compute_metrics.py --config config.json
```

Outputs:
- `out/phase2_eval/metrics/metrics.csv` — SSIM (and LPIPS if available) per output.
- `out/phase2_eval/tables/master_index_metrics.csv` — master index joined with metrics.

## 5) Plot Metric Curves

```bash
python 04_plots.py --config config.json --by style_weight
```

Outputs:
- `out/phase2_eval/figures/*png` — SSIM/LPIPS vs hyperparam plots.

## 6) Summarize & Suggest Baseline

```bash
python 05_summarize_baseline.py --config config.json
```

Outputs:
- `out/phase2_eval/tables/baseline_choices.csv` — best hyperparams per (content, style).
- Console report explaining choices and tradeoffs.

---

### Notes
- If images are on Windows paths in manifests, scripts normalize `\` to `/` and try both `finals` and `imgs`.
- If originals aren't found at the manifest paths, set `content_root_override` and `style_root_override` in `config.json`.
- Grids avoid seaborn and use matplotlib; one plot per figure; no explicit colors.

