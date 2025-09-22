#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make comparison grids from NST sweep results with wrapped captions and dynamic caption height.

Reads:
  - config.json -> paths for phase2, manifests, finals, imgs, eval_dir
  - out/phase2_eval/tables/master_index.csv -> one row per output

Writes:
  - out/phase2_eval/grids/grid__{content}__{style}__by-{hyperparam}.png

Usage (PowerShell example):
  python nst_phase1_toolkit/02_make_grids.py --config config.json `
    --content "animal,portrait2,city,Still_Life" `
    --style "Matisse,Monet,Ukiyo-e print" `
    --by style_weight --topk 16 --thumb 256 --cols 4
"""

import os
import math
import argparse
import json
import csv
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont


# ----------------------------
# Utilities
# ----------------------------
def load_csv(path: str) -> List[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def try_open(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def kfmt(sw) -> str:
    """14000 -> '14k' (best-effort)."""
    try:
        s = float(sw)
        if abs(s) >= 1000 and s % 1000 == 0:
            return f"{int(s/1000)}k"
        # keep as is if not clean multiple of 1000
        return str(int(s)) if s.is_integer() else str(s)
    except Exception:
        return str(sw)


def compress_layers(layers: Optional[str]) -> str:
    """
    'conv4_1,conv5_1' -> 'L=4,5'
    If missing/unknown -> 'L=?'
    """
    if not layers:
        return "L=?"
    parts = [p.strip() for p in layers.split(",")]
    nums = []
    import re
    for p in parts:
        m = re.search(r"conv(\d+)_", p)
        nums.append(m.group(1) if m else "?")
    return f"L={','.join(nums)}"


def text_width(draw: ImageDraw.ImageDraw, text: str, font: Optional[ImageFont.ImageFont]) -> int:
    """Robust text width: use textlength if available, else textbbox."""
    if hasattr(draw, "textlength"):
        return int(draw.textlength(text, font=font))
    # Fallback
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def wrap_caption(draw: ImageDraw.ImageDraw, text: str, max_width_px: int,
                 font: Optional[ImageFont.ImageFont]) -> List[str]:
    """
    Word-wrap caption so each line fits within max_width_px.
    Returns list of lines.
    """
    words = text.split()
    lines: List[str] = []
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        wlen = text_width(draw, test, font)
        if wlen <= max_width_px or not line:
            line = test
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.json")
    ap.add_argument("--content", default="", help="Comma-separated content_name list to include")
    ap.add_argument("--style", default="", help="Comma-separated style_name list to include")
    ap.add_argument("--by", default="style_weight", help="Hyperparam to sort by (e.g., style_weight, tv_weight, layers)")
    ap.add_argument("--topk", type=int, default=12, help="Max images per grid")
    ap.add_argument("--thumb", type=int, default=256, help="Thumbnail/tile width & height")
    ap.add_argument("--cols", type=int, default=4, help="Number of columns in the grid (max)")
    args = ap.parse_args()

    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    eval_dir = Path(cfg["eval_dir"])
    grids_dir = eval_dir / "grids"
    ensure_dir(grids_dir)

    tables_csv = eval_dir / "tables" / "master_index.csv"
    if not tables_csv.exists():
        raise FileNotFoundError(f"Missing table: {tables_csv}")

    rows = load_csv(str(tables_csv))

    sel_content = set([x.strip() for x in args.content.split(",") if x.strip()]) if args.content else None
    sel_style = set([x.strip() for x in args.style.split(",") if x.strip()]) if args.style else None

    # Filter by content/style if provided
    rows2 = []
    for r in rows:
        c_name = r.get("content_name")
        s_name = r.get("style_name")
        if sel_content and c_name not in sel_content:
            continue
        if sel_style and s_name not in sel_style:
            continue
        rows2.append(r)

    # Group by (content_name, style_name)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows2:
        groups[(r.get("content_name"), r.get("style_name"))].append(r)

    # Fonts + dummy drawer for measuring text
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    dummy = Image.new("RGB", (10, 10), (255, 255, 255))
    draw_dummy = ImageDraw.Draw(dummy)

    finals_dir = Path(cfg["finals_dir"])
    imgs_dir = Path(cfg["imgs_dir"])

    for (cname, sname), lst in groups.items():
        # Sort by chosen hyperparam; attempt numeric sorting
        key = args.by

        def key_fn(x):
            v = x.get(key, "")
            try:
                return float(v)
            except Exception:
                # for non-numeric fields like 'layers', try length as a rough proxy
                return float(len(str(v)))

        lst_sorted = sorted(lst, key=key_fn)
        sel = lst_sorted[: args.topk]

        # Load images and build short captions
        images: List[Image.Image] = []
        captions: List[str] = []
        rows_meta: List[dict] = []

        for r in sel:
            # Resolve image path: prefer out_final then out_img; try base name under finals/imgs if necessary
            imgp = (r.get("out_final") or r.get("out_img") or "").replace("\\", "/")
            im = try_open(imgp)
            if im is None:
                base = os.path.basename(imgp)
                for trial in [finals_dir / base, imgs_dir / base]:
                    im = try_open(str(trial))
                    if im:
                        break
            if im is None:
                # If we can't find the image, skip this row
                print(f"[WARN] Missing image for {cname} x {sname} -> {imgp}")
                continue

            images.append(im)
            rows_meta.append(r)

            # Short, readable caption
            cap = f"sw={kfmt(r.get('style_weight'))} | tv={r.get('tv_weight')} | {compress_layers(r.get('layers'))}"
            captions.append(cap)

        if not images:
            print(f"[WARN] No images for group {cname} — {sname}")
            continue

        # Prepare thumbnails
        TH = max(64, int(args.thumb))
        thumbs = []
        for im in images:
            im2 = im.copy()
            im2.thumbnail((TH, TH))
            thumbs.append(im2)

        # Grid layout
        n = len(thumbs)
        cols = max(1, min(int(args.cols), n))
        rows_n = math.ceil(n / cols)

        # Wrap captions and compute dynamic row heights
        wrapped_caps: List[List[str]] = []
        max_lines_per_row = [0] * rows_n

        for idx, cap in enumerate(captions):
            cap_lines = wrap_caption(draw_dummy, cap, max_width_px=TH - 10, font=font)
            wrapped_caps.append(cap_lines)
            r_i = idx // cols
            max_lines_per_row[r_i] = max(max_lines_per_row[r_i], len(cap_lines))

        # Estimate line height
        bbox = draw_dummy.textbbox((0, 0), "Ag", font=font)
        line_h = (bbox[3] - bbox[1]) if bbox else 12
        pad_y = 6  # extra space below caption block

        # Row heights = thumbnail + caption block
        row_heights = []
        for r_i in range(rows_n):
            cap_h = max(1, max_lines_per_row[r_i]) * line_h + pad_y
            row_heights.append(TH + cap_h)

        grid_w = cols * TH
        grid_h = sum(row_heights)
        grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
        draw = ImageDraw.Draw(grid)

        # Paste thumbs and draw wrapped captions
        y_cursor = 0
        for r_i in range(rows_n):
            cap_h = row_heights[r_i] - TH
            for c_i in range(cols):
                idx = r_i * cols + c_i
                if idx >= len(thumbs):
                    break
                x = c_i * TH
                grid.paste(thumbs[idx], (x, y_cursor))

                text_y = y_cursor + TH + 2
                for line in wrapped_caps[idx]:
                    draw.text((x + 4, text_y), line, fill=(0, 0, 0), font=font)
                    text_y += line_h
            y_cursor += row_heights[r_i]

        outp = grids_dir / f"grid__{cname}__{sname}__by-{key}.png"
        grid.save(outp)
        print(f"[OK] Wrote {outp}")


if __name__ == "__main__":
    main()
