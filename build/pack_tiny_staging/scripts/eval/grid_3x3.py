# scripts/grid_3x3.py
import os, sys, csv, json, time, argparse, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from skimage import color, metrics, feature
from skimage.transform import resize

def run_cmd(args_list):
    t0 = time.perf_counter()
    cp = subprocess.run(args_list, capture_output=True, text=True)
    t1 = time.perf_counter()
    if cp.returncode != 0:
        print("\n--- STDOUT ---\n", cp.stdout)
        print("\n--- STDERR ---\n", cp.stderr)
        raise subprocess.CalledProcessError(cp.returncode, cp.args)
    return t1 - t0

def parse_semilist(s):
    # semicolon-separated list; keeps paths with spaces intact
    return [x.strip() for x in s.split(";") if x.strip()]

def load_gray01(p):
    arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return color.rgb2gray(arr)

def edge_ratio(gray):
    return float(feature.canny(gray, sigma=1.2).mean())

def load_mask01_guess(mask_path, content_path, target_hw):
    m = None
    cand = None
    if mask_path:
        cand = mask_path
    else:
        stem = Path(content_path)
        # try "<name>_face_mask.png" next to content
        guess = stem.with_name(stem.stem + "_face_mask.png")
        if guess.exists():
            cand = str(guess)
    if cand and Path(cand).exists():
        m = np.asarray(Image.open(cand).convert("L"), dtype=np.float32) / 255.0
        m = resize(m, target_hw, mode="reflect", anti_aliasing=True)
        m = (m > 0.5).astype(np.float32)
    return m  # HxW in {0,1} or None

def ssim_full_and_face(content_gray, out_gray, mask01):
    # resize content if needed
    if content_gray.shape != out_gray.shape:
        content_gray = resize(content_gray, out_gray.shape, mode="reflect", anti_aliasing=True)
    ssim_full = float(metrics.structural_similarity(content_gray, out_gray, data_range=1.0))
    if mask01 is not None and mask01.sum() > 0:
        _, ssim_map = metrics.structural_similarity(content_gray, out_gray, data_range=1.0, full=True)
        ssim_face = float((ssim_map * mask01).sum() / (mask01.sum() + 1e-6))
    else:
        ssim_face = ssim_full
    return ssim_full, ssim_face

def build_cmd(method, backbone, content, style, out_img, sizes, steps, seed, device, face_mask):
    if method == "gatys":
        return [sys.executable, "scripts/core/nst_hybrid_ready.py",
                "--backbone", backbone,
                "--style_layers", "conv4_1,conv5_1",
                "--content_layer", "conv4_2",
                "--content", content,
                "--style", style,
                "--out", out_img,
                "--sizes", sizes,
                "--steps", steps,
                "--seed", str(seed),
                "--device", device] + (["--face_mask", face_mask] if face_mask else [])
    elif method == "adain":
        return [sys.executable, "scripts/core/baseline_adain_loss.py",
                "--backbone", backbone,
                "--content_layer", "conv4_2",
                "--style_layers", "conv4_1,conv5_1",
                "--content", content,
                "--style", style,
                "--out", out_img,
                "--sizes", sizes,
                "--steps", steps,
                "--seed", str(seed),
                "--device", device]
    elif method == "wct":
        return [sys.executable, "scripts/core/baseline_wct_loss.py",
                "--backbone", backbone,
                "--layer", "conv4_1",
                "--content", content,
                "--style", style,
                "--out", out_img,
                "--sizes", sizes,
                "--steps", steps,
                "--seed", str(seed),
                "--device", device]
    else:
        raise ValueError("method must be one of: gatys, adain, wct")

def draw_contact_sheet(grid_paths, content_labels, style_labels, save_path, tile=320, pad=16):
    """
    grid_paths: 3x3 list of output image paths (row = content, col = style)
    content_labels: list of 3 strings (row headers)
    style_labels:   list of 3 strings (col headers)
    """
    W = pad + (tile+pad)*3
    H = pad + (tile+pad)*3 + 40  # top header row
    sheet = Image.new("RGB", (W, H), (245,245,245))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    # draw column headers (styles)
    for j, lab in enumerate(style_labels):
        x = pad + j*(tile+pad) + tile//2
        draw.text((x- (len(lab)*3), 4), lab, fill=(10,10,10), font=font)

    # draw grid
    for i in range(3):
        for j in range(3):
            cell_x = pad + j*(tile+pad)
            cell_y = 40 + pad + i*(tile+pad)
            # row labels (contents)
            if j == 0:
                draw.text((4, cell_y + tile//2 - 6), content_labels[i], fill=(10,10,10), font=font)
            p = grid_paths[i][j]
            if not Path(p).exists():
                # empty placeholder
                Image.new("RGB", (tile, tile), (220,220,220)).save(p)
            im = Image.open(p).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            sheet.paste(im, (cell_x, cell_y))
    sheet.save(save_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["gatys","adain","wct"], default="gatys",
                    help="Which baseline to use for the 3x3 grid.")
    ap.add_argument("--backbone", choices=["vgg16","vgg19"], default="vgg16")
    ap.add_argument("--contents", required=True,
                    help='Semicolon-separated 3 content paths. Example: "data/content/a.jpg;data/content/b.jpg;data/content/c.jpg"')
    ap.add_argument("--styles", required=True,
                    help='Semicolon-separated 3 style paths. Example: "data/style/s1.jpg;data/style/s2.jpg;data/style/s3.jpg"')
    ap.add_argument("--face_masks", default="",
                    help='Optional semicolon-separated 3 face masks (one per content). Leave empty entries to auto-guess.')
    ap.add_argument("--out_root", default="out/grid")
    ap.add_argument("--sizes", default="384,768")
    ap.add_argument("--steps", default="250,350")      # a bit lighter than A3 to keep it quick
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    contents = parse_semilist(args.contents)
    styles   = parse_semilist(args.styles)
    masks_in = [s.strip() for s in args.face_masks.split(";")] if args.face_masks else ["","",""]

    assert len(contents) == 3 and len(styles) == 3, "Please provide exactly 3 contents and 3 styles."

    tag = f"{args.method}_{args.backbone}"
    root = Path(args.out_root) / tag
    root.mkdir(parents=True, exist_ok=True)

    # Run all 9 pairs
    rows = []
    out_grid = [[None]*3 for _ in range(3)]
    content_grays = [load_gray01(p) for p in contents]

    for i, cpath in enumerate(contents):
        for j, spath in enumerate(styles):
            pair_dir = root / f"c{i+1}_s{j+1}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            out_img = str(pair_dir / "out.jpg")

            # choose face mask (explicit or guessed)
            mask_path = masks_in[i].strip() if i < len(masks_in) and masks_in[i].strip() else None

            cmd = build_cmd(args.method, args.backbone, cpath, spath, out_img,
                            args.sizes, args.steps, args.seed, args.device, mask_path)

            print(f"[RUN] {args.method.upper()} {args.backbone}  C{i+1}×S{j+1}")
            secs = run_cmd(cmd)

            # metrics
            out_gray = load_gray01(out_img)
            mask01 = load_mask01_guess(mask_path, cpath, out_gray.shape)
            ssim_full, ssim_face = ssim_full_and_face(content_grays[i], out_gray, mask01)
            er = edge_ratio(out_gray)

            # prefer child's JSON VRAM if present
            meta_path = out_img.replace(".jpg", ".json")
            peak_gb = ""
            if os.path.exists(meta_path):
                try:
                    with open(meta_path,"r") as f:
                        meta = json.load(f)
                    if isinstance(meta.get("peak_vram_gb", None), (int,float)):
                        peak_gb = round(float(meta["peak_vram_gb"]), 3)
                except Exception:
                    pass

            rows.append({
                "method": args.method, "backbone": args.backbone,
                "content": cpath, "style": spath, "out": out_img,
                "runtime_sec": round(secs, 3), "peak_vram_gb": peak_gb,
                "sizes": args.sizes, "steps": args.steps, "seed": args.seed,
                "ssim_face": round(ssim_face, 4), "ssim_full": round(ssim_full, 4),
                "edge_ratio": round(er, 4)
            })
            out_grid[i][j] = out_img

    # CSV
    csv_path = root / "grid_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # Contact sheet
    content_labels = [Path(p).stem[:18] for p in contents]
    style_labels   = [Path(p).stem[:18] for p in styles]
    contact_path = root / "contact_sheet_3x3.jpg"
    draw_contact_sheet(out_grid, content_labels, style_labels, str(contact_path))
    print(f"Saved: {contact_path}")

if __name__ == "__main__":
    main()
