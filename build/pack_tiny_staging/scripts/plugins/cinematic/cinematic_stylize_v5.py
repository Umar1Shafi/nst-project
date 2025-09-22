#!/usr/bin/env python
# -*- coding: utf-8 -*-
# I turn photos into a teal–orange “cinematic” look (SD1.5 + ControlNet). I keep comments short.

import os, argparse, warnings
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageOps

import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- IO helpers ----------------

def load_image(path: str) -> Image.Image:
    # I always work in RGB.
    return Image.open(path).convert("RGB")

def resize_max_side(img: Image.Image, max_side=1280) -> Image.Image:
    # I keep aspect ratio and snap sizes to /8 so SD runs cleanly.
    w, h = img.size
    if max(w, h) <= max_side:
        w8, h8 = (w // 8) * 8, (h // 8) * 8
        return img.resize((max(8, w8), max(8, h8)), Image.LANCZOS)
    s = max_side / float(max(w, h))
    W = int((w * s) // 8) * 8
    H = int((h * s) // 8) * 8
    return img.resize((max(8, W), max(8, H)), Image.LANCZOS)

def ensure_dir(p: Path):
    # I make sure the output folder exists.
    p.parent.mkdir(parents=True, exist_ok=True)

def to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(np.float32) / 255.0

def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))

# ---------------- Annotators ----------------

def control_image_softedge(img: Image.Image) -> Image.Image|None:
    # I prefer HED edges for portraits.
    try:
        from controlnet_aux.hed import HEDdetector
        hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
        e = hed(img)
        return e.convert("RGB")
    except Exception as e:
        print(f"[warn] softedge annotator unavailable: {e}")
        return None

def control_image_depth(img: Image.Image) -> Image.Image|None:
    # I use depth for scenes to guide structure.
    try:
        from controlnet_aux.midas import MidasDetector
        midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
        dep = midas(img)
        return ImageOps.autocontrast(dep).convert("RGB")
    except Exception as e:
        print(f"[warn] depth annotator unavailable: {e}")
        return None

# ---------------- Utility ----------------

def luminance(arr: np.ndarray) -> np.ndarray:
    return 0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]

def sigmoid(x, k=10.0, x0=0.5):
    return 1.0/(1.0 + np.exp(-k*(x - x0)))

def saturate_pil(pil: Image.Image, sat_scale: float) -> Image.Image:
    # I nudge saturation at the end to keep colors lively.
    if abs(sat_scale - 1.0) < 1e-3:
        return pil
    hsv = pil.convert("HSV")
    h, s, v = hsv.split()
    s_np = np.array(s, dtype=np.float32)
    s_np = np.clip(s_np * sat_scale, 0, 255).astype(np.uint8)
    hsv = Image.merge("HSV", (h, Image.fromarray(s_np), v))
    return hsv.convert("RGB")

# ---------------- Refined dual-mode grade ----------------

def grade_v5(
    img: Image.Image,
    subject: str = "scene",
    tone_mix: float | None = None,
    bloom: float | None = None,
    contrast: float | None = None,
    skin_suppress: float | None = None,
    saturation: float = 1.05,
    add_dither: bool = True,
) -> Image.Image:
    # I do the teal–orange grade in numpy, with different settings for scene vs portrait.
    a = to_np(img)
    base = a.copy()
    h, w, _ = a.shape
    lum = luminance(a)

    if subject == "portrait":
        tone_mix = 0.22 if tone_mix is None else tone_mix
        bloom = 0.22 if bloom is None else bloom
        contrast = 0.18 if contrast is None else contrast
        skin_suppress = 0.80 if skin_suppress is None else skin_suppress
        shadow_w = sigmoid(0.35 - lum, k=12.0, x0=0.0)
        highlight_w = sigmoid(lum - 0.60, k=10.0, x0=0.0)
    else:
        tone_mix = 0.40 if tone_mix is None else tone_mix
        bloom = 0.42 if bloom is None else bloom
        contrast = 0.24 if contrast is None else contrast
        skin_suppress = 0.85 if skin_suppress is None else skin_suppress
        shadow_w = sigmoid(0.50 - lum, k=12.0, x0=0.0)
        highlight_w = sigmoid(lum - 0.55, k=12.0, x0=0.0)

    shadow_w = shadow_w[...,None].astype(np.float32)
    highlight_w = highlight_w[...,None].astype(np.float32)

    # I push teal in shadows and orange in highlights.
    if subject == "scene":
        teal_vec   = np.array([0.02, 0.58, 1.00], dtype=np.float32)
        orange_vec = np.array([1.05, 0.70, 0.05], dtype=np.float32)
    else:
        teal_vec   = np.array([0.02, 0.55, 0.95], dtype=np.float32)
        orange_vec = np.array([1.00, 0.62, 0.07], dtype=np.float32)

    teal_field = teal_vec[None,None,:] * shadow_w
    orange_field = orange_vec[None,None,:] * highlight_w
    mask = np.clip(shadow_w + highlight_w, 0.0, 1.0)
    a = np.clip(a*(1.0 - tone_mix*mask) + tone_mix*(teal_field + orange_field), 0.0, 1.0)

    # I warm midtones for scenes.
    if subject == "scene":
        mid = np.clip((lum - 0.45)/0.20, 0.0, 1.0) * np.clip((0.65 - lum)/0.20, 0.0, 1.0)
        mid = (mid * 0.06)[...,None]
        a[...,0] += mid[...,0]
        a[...,1] += (mid[...,0] * 0.5)

    # I protect skin for portraits.
    if subject == "portrait":
        r, g, b = a[...,0], a[...,1], a[...,2]
        skin = (r > g) & (r > b) & (r > 0.30) & (lum > 0.20) & (lum < 0.90)
        if skin.any():
            try:
                from scipy.ndimage import uniform_filter
                m = uniform_filter(skin.astype(np.float32), size=5)
            except Exception:
                m = skin.astype(np.float32)
            m = m[...,None]
            a = a*(1.0 - 0.40*m*skin_suppress) + base*(0.40*m*skin_suppress)
            mid_skin = ((lum > 0.35) & (lum < 0.65))[...,None].astype(np.float32) * m
            a[...,0] += 0.10 * mid_skin[...,0]
            a[...,1] += 0.05 * mid_skin[...,0]

    if contrast and contrast > 0.0:
        a = np.clip((a - 0.5) * (1.0 + 1.8*contrast) + 0.5, 0.0, 1.0)

    if bloom and bloom > 0.0:
        # I add soft glow on bright regions.
        try:
            from scipy.ndimage import gaussian_filter
            thr = np.clip((lum - 0.60)/0.40, 0.0, 1.0)[...,None]
            glow = gaussian_filter(a*thr, sigma=(6.0,6.0,0.0))
            a = np.clip(a + bloom*glow, 0.0, 1.0)
        except Exception:
            pass

    out = to_pil(a).filter(ImageFilter.UnsharpMask(radius=1, percent=115, threshold=5))
    out = saturate_pil(out, saturation)

    if add_dither:
        # I add tiny ordered dither to fight banding.
        arr = to_np(out); h, w, _ = arr.shape
        yy, xx = np.mgrid[0:h, 0:w]
        bayer = ((xx & 1) + ((yy & 1) << 1)) / 4.0 - 0.375
        arr += (1.0/255.0)*bayer[...,None]
        out = to_pil(np.clip(arr, 0.0, 1.0))

    return out

# ---------------- Runner ----------------

def run(
    model_id: str,
    device: str,
    src: Image.Image,
    control_img: Image.Image,
    controlnet_id: str,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    strength: float,
    control_scale: float,
    seed: int,
):
    # I set up the SD+ControlNet pipeline and render one image.
    if control_img.size != src.size:
        control_img = control_img.resize(src.size, Image.LANCZOS)

    controlnet = ControlNetModel.from_pretrained(
        controlnet_id,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        image=src,
        control_image=control_img,
        controlnet_conditioning_scale=float(control_scale),
        negative_prompt=negative_prompt,
        guidance_scale=guidance,
        strength=strength,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]
    return result

# ---------------- CLI ----------------

def main():
    # I expose simple flags; I tune defaults per subject.
    p = argparse.ArgumentParser("Cinematic Teal–Orange v5 (refined)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-i","--input", required=True)
    p.add_argument("-o","--output", required=True)
    p.add_argument("--subject", choices=["portrait","scene"], default="scene")
    p.add_argument("--model", default=os.environ.get("SD15_MODEL","runwayml/stable-diffusion-v1-5"))
    p.add_argument("--prompt", default="cinematic photograph, natural skin, warm key light, cool ambient shadows, filmic contrast, clean detail")
    p.add_argument("--negative-prompt", default="text, watermark, overprocessed skin, oversharpened halos, garish saturation, plastic skin, heavy vignette")
    p.add_argument("--steps", type=int, default=36)
    p.add_argument("--guidance", type=float, default=6.6)
    p.add_argument("--strength", type=float, default=0.38)
    p.add_argument("--control-scale", type=float, default=0.48)
    p.add_argument("--max-side", type=int, default=1280)
    p.add_argument("--seed", type=int, default=77)
    # Grade overrides
    p.add_argument("--tone-mix", type=float, default=None)
    p.add_argument("--bloom", type=float, default=None)
    p.add_argument("--contrast", type=float, default=None)
    p.add_argument("--skin-suppress", type=float, default=None)
    p.add_argument("--saturation", type=float, default=1.05, help="Final saturation multiplier (HSV S channel)")
    p.add_argument("--no-dither", action="store_true")
    p.add_argument("--low-vram", action="store_true")

    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    src = resize_max_side(load_image(args.input), args.max_side)

    # I pick annotator + ControlNet family based on subject.
    if args.subject == "portrait":
        args.steps = max(args.steps, 34)
        args.guidance = max(args.guidance, 6.2)
        args.strength = min(args.strength, 0.26)
        args.control_scale = min(max(args.control_scale, 0.24), 0.40)
        cimg = control_image_softedge(src); controlnet_id = "lllyasviel/sd-controlnet-hed"
    else:
        args.steps = max(args.steps, 36)
        args.guidance = max(args.guidance, 6.6)
        args.strength = min(args.strength, 0.42)
        args.control_scale = min(max(args.control_scale, 0.42), 0.55)
        cimg = control_image_depth(src); controlnet_id = "lllyasviel/sd-controlnet-depth"

    if cimg is None:
        print("[warn] Annotator missing — using RGB image as weak control.")
        cimg = src

    out = run(
        model_id=args.model,
        device=device,
        src=src,
        control_img=cimg,
        controlnet_id=controlnet_id,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        guidance=args.guidance,
        strength=args.strength,
        control_scale=args.control_scale,
        seed=args.seed,
    )

    # I apply the color grade and a tiny sharpen.
    graded = grade_v5(
        out,
        subject=args.subject,
        tone_mix=args.tone_mix,
        bloom=args.bloom,
        contrast=args.contrast,
        skin_suppress=args.skin_suppress,
        saturation=args.saturation,
        add_dither=not args.no_dither,
    ).filter(ImageFilter.UnsharpMask(radius=1, percent=115, threshold=5))

    ensure_dir(Path(args.output))
    graded.save(args.output)
    print(f"✅ Saved: {args.output}")

if __name__ == "__main__":
    main()
