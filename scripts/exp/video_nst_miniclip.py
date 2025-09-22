#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tiny video NST (polished, multi-extractor):
- Input: a short video OR a folder of frames.
- Output: stylized frames + optional MP4/GIF.
- Frame extraction backends: imageio | ffmpeg | opencv  (auto-fallback).
- Stability:
  * frame 0 -> LBFGS; frames >0 -> Adam init from previous stylized
  * optional optical-flow warp of previous stylized
  * optional temporal EMA smoothing across stylized frames

Examples (PowerShell):

# Full run (stylize + MP4) using the default extractor (imageio):
python scripts\\exp\\video_nst_miniclip.py ^
  --video data\\video\\city_short.mp4 ^
  --style data\\style\\Monet.jpg ^
  --out_dir out\\video\\monet_city ^
  --mp4 out\\video\\monet_city.mp4 ^
  --max_frames 16 --size 512 ^
  --steps_first 260 --steps_rest 180 ^
  --style_layers layers45 ^
  --style_weight 17000 --tv_weight 0.003 ^
  --edge_w 0.02 --cpm_strength 0.25 ^
  --flow 1 --temporal_smooth 1 --fps 12 --device cuda

# If you prefer ffmpeg for extraction (requires ffmpeg in PATH):
python scripts\\exp\\video_nst_miniclip.py ^
  --video data\\video\\city_short.mp4 ^
  --style data\\style\\Monet.jpg ^
  --out_dir out\\video\\monet_city ^
  --mp4 out\\video\\monet_city.mp4 ^
  --extractor ffmpeg --max_frames 16 --size 512 --fps 12

# Assemble-only (no stylization) from existing stylized_*.png:
python scripts\\exp\\video_nst_miniclip.py ^
  --out_dir out\\video\\monet_city ^
  --mp4 out\\video\\monet_city.mp4 ^
  --fps 12 --assemble_only 1
"""

import argparse, os, subprocess, sys, shutil
from pathlib import Path
import numpy as np

# Optional deps (we try to stay robust if some are missing)
try:
    import cv2
except Exception:
    cv2 = None

from PIL import Image
import imageio.v2 as imageio

# -------------------------
# Small utilities
# -------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def center_crop_square_pil(pil: Image.Image) -> Image.Image:
    w, h = pil.size
    s = min(w, h)
    left = (w - s) // 2
    top  = (h - s) // 2
    return pil.crop((left, top, left + s, top + s))

def resize_square(pil: Image.Image, side: int) -> Image.Image:
    return pil.resize((side, side), Image.Resampling.LANCZOS)

# -------------------------
# Frame extraction (three backends)
# -------------------------
def extractor_imageio(video_path: Path, frames_dir: Path, max_frames: int, do_square: bool, target_side: int, every_k: int = 1):
    reader = imageio.get_reader(str(video_path))
    idx = 0
    written = []
    for n, frame in enumerate(reader):
        if n % every_k != 0:
            continue
        if idx >= max_frames:
            break
        pil = Image.fromarray(frame)  # RGB
        if do_square:
            pil = center_crop_square_pil(pil)
        if target_side:
            pil = resize_square(pil, target_side)
        outp = frames_dir / f"f{idx:03d}.png"
        pil.save(outp)
        written.append(outp)
        idx += 1
    reader.close()
    return written

def extractor_ffmpeg(video_path: Path, frames_dir: Path, max_frames: int, do_square: bool, target_side: int, fps: int | None):
    """
    Use ffmpeg CLI to dump processed frames directly.
    Requires ffmpeg in PATH. We'll build a filter chain:
      [fps] -> [optional crop to square] -> [scale target_side x target_side]
    Then we stop at max_frames with -frames:v.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")
    ensure_dir(frames_dir)
    filters = []
    if fps is not None and fps > 0:
        filters.append(f"fps={fps}")
    # crop to square: crop='min(iw,ih)':'min(iw,ih)'
    if do_square:
        filters.append("crop=min(iw\\,ih):min(iw\\,ih)")
    if target_side and target_side > 0:
        filters.append(f"scale={target_side}:{target_side}:flags=lanczos")
    vf = ",".join(filters) if filters else "null"
    out_pattern = str(frames_dir / "f%03d.png")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-frames:v", str(max_frames),
        out_pattern
    ]
    print("[ffmpeg extract]", " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({r.returncode}). Stderr:\n{r.stderr.decode(errors='ignore')}")
    # List what we wrote
    return sorted(frames_dir.glob("f*.png"))

def extractor_opencv(video_path: Path, frames_dir: Path, max_frames: int, do_square: bool, target_side: int, every_k: int = 1):
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    idx = 0
    n = 0
    written = []
    while idx < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if n % every_k != 0:
            n += 1
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        if do_square:
            pil = center_crop_square_pil(pil)
        if target_side:
            pil = resize_square(pil, target_side)
        outp = frames_dir / f"f{idx:03d}.png"
        pil.save(outp)
        written.append(outp)
        idx += 1
        n += 1
    cap.release()
    return written

def extract_frames(video_path: Path, frames_dir: Path, max_frames: int, do_square: bool, target_side: int,
                   extractor: str = "auto", src_fps: int | None = None, every_k: int = 1):
    """
    extractor options:
      - 'imageio' : use imageio.get_reader (DEFAULT)
      - 'ffmpeg'  : call ffmpeg (must be installed)
      - 'opencv'  : use cv2.VideoCapture
      - 'auto'    : try imageio -> ffmpeg -> opencv
    'src_fps' is used by ffmpeg; 'every_k' decimates frames for imageio/opencv.
    """
    ensure_dir(frames_dir)
    tried = []
    def _try(fn, label):
        tried.append(label)
        return fn(video_path, frames_dir, max_frames, do_square, target_side) if label != "ffmpeg" \
               else fn(video_path, frames_dir, max_frames, do_square, target_side, src_fps)

    # route
    if extractor == "imageio":
        return extractor_imageio(video_path, frames_dir, max_frames, do_square, target_side, every_k)
    if extractor == "ffmpeg":
        return extractor_ffmpeg(video_path, frames_dir, max_frames, do_square, target_side, src_fps)
    if extractor == "opencv":
        return extractor_opencv(video_path, frames_dir, max_frames, do_square, target_side, every_k)

    # auto
    for label, fn in [("imageio", extractor_imageio), ("ffmpeg", extractor_ffmpeg), ("opencv", extractor_opencv)]:
        try:
            print(f"[extract] trying {label}...")
            if label == "imageio":
                return extractor_imageio(video_path, frames_dir, max_frames, do_square, target_side, every_k)
            elif label == "ffmpeg":
                return extractor_ffmpeg(video_path, frames_dir, max_frames, do_square, target_side, src_fps)
            else:
                return extractor_opencv(video_path, frames_dir, max_frames, do_square, target_side, every_k)
        except Exception as e:
            print(f"[extract] {label} failed: {e}")
            continue
    raise SystemExit(f"[error] all extractors failed; tried in order: imageio, ffmpeg, opencv")

# -------------------------
# Optional flow warp (dimension-safe)
# -------------------------
def maybe_warp_prev(prev_stylized_path: Path, prev_frame_path: Path, cur_frame_path: Path, side: int):
    """
    Warp prev stylized toward current frame using Farneback optical flow
    computed on square grayscale versions at the same side.
    """
    if cv2 is None:
        return prev_stylized_path
    try:
        pr = Image.open(prev_frame_path).convert("L").resize((side, side), Image.Resampling.LANCZOS)
        cr = Image.open(cur_frame_path).convert("L").resize((side, side), Image.Resampling.LANCZOS)
        pr = np.array(pr); cr = np.array(cr)
        flow = cv2.calcOpticalFlowFarneback(pr, cr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        prev_st = imageio.imread(prev_stylized_path)  # RGB
        if prev_st.shape[0] != side or prev_st.shape[1] != side:
            prev_st = np.array(Image.fromarray(prev_st).resize((side, side), Image.Resampling.LANCZOS))

        H, W = side, side
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(prev_st, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        wp = prev_stylized_path.with_name(prev_stylized_path.stem + "_warp.png")
        imageio.imwrite(wp, warped)
        return wp
    except Exception:
        return prev_stylized_path

# -------------------------
# Call your NST core (Gatys)
# -------------------------
def run_nst(core_script: Path, python_exe: str, content_path: Path, style_path: Path, out_path: Path,
            device, seed, sizes, steps, backbone, style_layers,
            style_weight, tv_weight, edge_w, face_preserve, cpm_strength,
            opt=None, init_path=None):
    cmd = [python_exe, str(core_script),
           "--content", str(content_path),
           "--style",   str(style_path),
           "--out",     str(out_path),
           "--device",  device,
           "--seed",    str(seed),
           "--sizes",   sizes,
           "--steps",   steps,
           "--backbone", backbone,
           "--style_layers", style_layers,
           "--style_weight", str(style_weight),
           "--tv_weight", str(tv_weight),
           "--edge_w", str(edge_w),
           "--face_preserve", str(face_preserve),
           "--color_prematch_strength", str(cpm_strength)]
    if opt:
        cmd += ["--opt", opt]
    if init_path:
        cmd += ["--init", str(init_path)]
    print("[NST]", " ".join(map(str, cmd)))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"[error] NST returned {r.returncode}")
    return out_path

# -------------------------
# Assembling outputs
# -------------------------
def _even_hw(h, w):
    return (h - (h % 2)), (w - (w % 2))

def write_mp4_from_frames(frames, mp4_path, fps, crf=16, preset="slow", pixfmt="yuv420p", rgb=False):
    if not frames:
        raise SystemExit("[error] no frames to MP4")
    im0 = imageio.imread(frames[0])
    H, W = im0.shape[:2]
    H2, W2 = _even_hw(H, W)

    # Try ffmpeg backend first (via imageio-ffmpeg)
    try:
        params = ["-crf", str(crf), "-preset", preset, "-pix_fmt", ("rgb24" if rgb else pixfmt)]
        codec  = ("libx264rgb" if rgb else "libx264")
        writer = imageio.get_writer(mp4_path, fps=fps, codec=codec, quality=None, ffmpeg_params=params)
        try:
            for p in frames:
                fr = imageio.imread(p)
                if (H2, W2) != (H, W):
                    fr = fr[:H2, :W2]
                writer.append_data(fr)
        finally:
            writer.close()
        print(f"[ok] MP4 -> {mp4_path} (ffmpeg {codec}, CRF={crf}, preset={preset}, pixfmt={'rgb24' if rgb else pixfmt})")
        return
    except Exception as e:
        print(f"[warn] ffmpeg writer failed ({e}); trying OpenCV fallback...")

    # Fallback: OpenCV
    if cv2 is None:
        raise SystemExit("[error] OpenCV not available and ffmpeg writer failed.")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(mp4_path), fourcc, float(fps), (W2, H2))
    if not vw.isOpened():
        raise SystemExit("[error] OpenCV VideoWriter could not open output MP4")
    for p in frames:
        fr = cv2.imread(str(p))
        if (H2, W2) != (H, W):
            fr = fr[:H2, :W2]
        vw.write(fr)
    vw.release()
    print(f"[ok] MP4 -> {mp4_path} (OpenCV/mp4v)")

def write_gif_from_frames(frames, gif_path, fps):
    imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=256) for p in frames]
    imageio.mimsave(gif_path, imgs, duration=1.0/max(fps,1))
    print(f"[ok] GIF -> {gif_path}")

def temporal_ema(frames, out_dir, alpha=0.80):
    """Return paths to smoothed frames (writes alongside originals)."""
    smoothed = []
    prev = None
    for p in frames:
        cur = imageio.imread(p).astype(np.float32)
        out = cur if prev is None else alpha * cur + (1.0 - alpha) * prev
        out = np.clip(out, 0, 255).astype(np.uint8)
        q = Path(out_dir) / (Path(p).stem + "_smooth.png")
        imageio.imwrite(q, out)
        smoothed.append(q)
        prev = out
    return smoothed

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    # Inputs / outputs
    ap.add_argument("--video", default=None, help="Path to a short video")
    ap.add_argument("--frames_dir", default=None, help="Folder with raw frames (optional)")
    ap.add_argument("--style", required=False)  # not needed for assemble-only
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--mp4", default=None)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--mp4_rgb", type=int, default=0, help="1 = write MP4 with libx264rgb/rgb24 for truer colors")

    # NST knobs
    ap.add_argument("--max_frames", type=int, default=15)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--steps_first", type=int, default=230)
    ap.add_argument("--steps_rest", type=int, default=180)
    ap.add_argument("--backbone", default="vgg19")
    ap.add_argument("--style_layers", default="layers45")  # leaner set for video
    ap.add_argument("--style_weight", type=float, default=17000)
    ap.add_argument("--tv_weight", type=float, default=0.0028)
    ap.add_argument("--edge_w", type=float, default=0.04)
    ap.add_argument("--face_preserve", type=float, default=0.75)
    ap.add_argument("--cpm_strength", type=float, default=0.30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=77)

    ap.add_argument("--flow", type=int, default=1, help="1 = warp previous stylized by optical flow")
    ap.add_argument("--square_pre", type=int, default=1, help="1 = center-crop to square + resize before NST")
    ap.add_argument("--temporal_smooth", type=int, default=1, help="1 = EMA smoothing across frames (alpha=0.8)")
    ap.add_argument("--assemble_only", type=int, default=0, help="1 = skip NST, just MP4/GIF")  
    ap.add_argument("--ema_alpha", type=float, default=0.80)
    ap.add_argument("--pyramid", type=int, default=0, help="1 = do 384->size per-frame")

    # Extraction
    ap.add_argument("--extractor", choices=["auto","imageio","ffmpeg","opencv"], default="auto",
                    help="Which frame extractor to use.")
    ap.add_argument("--src_fps", type=int, default=None,
                    help="Decimate source to this FPS during extraction (ffmpeg only).")
    ap.add_argument("--every_k", type=int, default=1,
                    help="For imageio/opencv: keep every k-th frame (1 = keep all).")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    core_script = Path("scripts/core/nst_hybrid_ready.py").resolve()
    python_exe  = sys.executable

    # ---- stylize or reuse ----
    if args.assemble_only:
        stylized_paths = sorted(out_dir.glob("stylized_*.png"))
        if not stylized_paths:
            raise SystemExit(f"[error] assemble_only: no frames found in {out_dir}")
    else:
        # Prepare frames (square + resized so flow and NST agree)
        if args.frames_dir:
            src = sorted(Path(args.frames_dir).glob("*.png"))[:args.max_frames]
            frames = []
            for i, p in enumerate(src):
                pil = Image.open(p).convert("RGB")
                if args.square_pre:
                    pil = center_crop_square_pil(pil)
                pil = resize_square(pil, args.size)
                fp = out_dir / "frames_src" / f"f{i:03d}.png"
                ensure_dir(fp.parent)
                pil.save(fp)
                frames.append(fp)
        else:
            if not args.video:
                raise SystemExit("Provide --video or --frames_dir (or use --assemble_only).")
            frames = extract_frames(
                Path(args.video),
                out_dir / "frames_src",
                args.max_frames,
                do_square=bool(args.square_pre),
                target_side=args.size,
                extractor=args.extractor,
                src_fps=args.src_fps,
                every_k=max(1, int(args.every_k)),
            )

        if not args.style:
            raise SystemExit("Provide --style when stylizing frames (omit only with --assemble_only).")
        style_path = Path(args.style).resolve()

        stylized_paths = []
        prev_stylized = None

        # Frame 0 (LBFGS)
        if len(frames) > 0:
            outp0 = out_dir / "stylized_000.png"
            sizes0 = (f"384,{args.size}" if args.pyramid else f"{args.size}")
            steps0 = (f"{max(80, args.steps_first//2)},{args.steps_first}" if args.pyramid else f"{args.steps_first}")
            run_nst(core_script, python_exe,
                    content_path=frames[0], style_path=style_path, out_path=outp0,
                    device=args.device, seed=args.seed,
                    sizes=sizes0,steps=steps0,
                    backbone=args.backbone, style_layers=args.style_layers,
                    style_weight=args.style_weight, tv_weight=args.tv_weight, edge_w=args.edge_w,
                    face_preserve=args.face_preserve, cpm_strength=args.cpm_strength,
                    opt="lbfgs", init_path=None)
            stylized_paths.append(outp0)
            prev_stylized = outp0

        # Frames 1..N-1 (Adam, init from previous; optional flow warp)
        for i in range(1, len(frames)):
            outp = out_dir / f"stylized_{i:03d}.png"
            sizesr = (f"384,{args.size}" if args.pyramid else f"{args.size}")
            stepsr = (f"{max(60, args.steps_rest//2)},{args.steps_rest}" if args.pyramid else f"{args.steps_rest}")
            init = prev_stylized
            if args.flow:
                init = maybe_warp_prev(prev_stylized, frames[i-1], frames[i], side=args.size)
            run_nst(core_script, python_exe,
                    content_path=frames[i], style_path=style_path, out_path=outp,
                    device=args.device, seed=args.seed,
                    sizes=sizesr, steps=stepsr,
                    backbone=args.backbone, style_layers=args.style_layers,
                    style_weight=args.style_weight, tv_weight=args.tv_weight, edge_w=args.edge_w,
                    face_preserve=args.face_preserve, cpm_strength=args.cpm_strength,
                    opt="adam", init_path=init)
            stylized_paths.append(outp)
            prev_stylized = outp

    # ---- optional temporal smoothing ----
    to_pack = stylized_paths
    if args.temporal_smooth:
        to_pack = temporal_ema(stylized_paths, out_dir, alpha=float(args.ema_alpha))

    # ---- write outputs ----
    if args.mp4:
        write_mp4_from_frames(to_pack, args.mp4, args.fps, rgb=bool(args.mp4_rgb))
    if args.gif:
        write_gif_from_frames(to_pack, args.gif, args.fps)

    if not (args.mp4 or args.gif):
        print("[note] no --mp4/--gif specified; frames are at", out_dir)

if __name__ == "__main__":
    main()
