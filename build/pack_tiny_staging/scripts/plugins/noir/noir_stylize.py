#!/usr/bin/env python
# -*- coding: utf-8 -*-
# I turn a photo into a noir/film look. I keep comments short and first-person.

import os, argparse, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import torch

# ---- CLI ----
def build_parser():
    # I collect simple flags so I can run this from the terminal.
    p = argparse.ArgumentParser("Photo → Noir Film")
    p.add_argument("-i", "--input", required=True, help="Input image path")
    p.add_argument("-o", "--output", required=True, help="Output image path (PNG recommended)")
    p.add_argument("--subject", choices=["portrait","scene"], default="portrait")
    p.add_argument("--prompt", default="high-contrast monochrome photo, dramatic lighting, cinematic composition, fine detail, sharp eyes, soft film halation")
    p.add_argument("--negative-prompt", default="color, oversharp halos, text, watermark, plastic skin, posterization")
    p.add_argument("--strength", type=float, default=0.18, help="Denoise 0..1 (I keep ≤0.28 for portraits)")
    p.add_argument("--guidance", type=float, default=6.0, help="Classifier-free guidance")
    p.add_argument("--steps", type=int, default=34, help="Sampling steps")
    p.add_argument("--scheduler", choices=["uni","dpmpp"], default="dpmpp")
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--control", choices=["none","canny"], default="none")
    p.add_argument("--control-scale", type=float, default=0.45, help="ControlNet weight")
    p.add_argument("--max-side", type=int, default=1280, help="Resize longest side (keeps aspect)")
    # Memory/attention
    p.add_argument("--mem-attn", choices=["auto","always","never"], default="auto",
                   help="xFormers on CUDA: auto tries, always requires, never skips")
    # Simple noir grade knobs
    p.add_argument("--noir-vignette", type=float, default=0.18)
    p.add_argument("--noir-halation", type=float, default=0.28)
    p.add_argument("--noir-bloom-sigma", type=float, default=3.0)
    p.add_argument("--noir-bloom-thresh", type=float, default=0.70)
    p.add_argument("--noir-dither", type=float, default=0.002, help="Grain stddev (0 disables)")
    p.add_argument("--noir-gamma", type=float, default=0.96)
    p.add_argument("--noir-gain",  type=float, default=1.02)
    p.add_argument("--noir-lift",  type=float, default=0.02)
    return p

# ---- Utils ----
def set_seed(seed: int):
    # I make runs repeatable.
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_image(path: str, max_side: int) -> Image.Image:
    # I load RGB and shrink if it's too big.
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side / float(max(w, h))
    if scale < 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return img

def pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(np.float32)/255.0

def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr*255.0).astype(np.uint8))

# ---- Noir grade (post) ----
def grade_noir(pil_img: Image.Image,
               vignette=0.18, halation=0.28,
               bloom_sigma=3.0, bloom_thresh=0.70,
               dither_std=0.002,
               filmic_lift=0.02, filmic_gamma=0.96, filmic_gain=1.02) -> Image.Image:
    # I do grayscale luma, local contrast, bloom/halation, vignette, and light grain.
    import cv2, numpy as np
    from PIL import ImageFilter

    arr = pil_to_numpy(pil_img)                      # HxWx3
    luma = (0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]).astype(np.float32)  # HxW

    # Local contrast on luma
    blur = cv2.GaussianBlur(luma, (0,0), 1.2)
    hp   = np.clip(luma - blur, 0.0, 1.0)
    l2   = np.clip(luma + 0.65*hp, 0.0, 1.0)

    # Bloom / halation
    bright = np.clip(l2 - float(bloom_thresh), 0.0, 1.0)
    halo   = cv2.GaussianBlur(bright, (0,0), float(bloom_sigma))
    out2d  = np.clip(l2 + float(halation)*halo, 0.0, 1.0)

    # Vignette
    h, w = out2d.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w/2)/(0.9*w))**2 + ((yy - h/2)/(0.9*h))**2)
    vig = np.clip(1.0 - float(vignette)*(r**1.5), 0.0, 1.0)
    out2d = np.clip(out2d * vig, 0.0, 1.0)

    # Filmic curve (lift/gamma/gain)
    def _filmic(x, lift=0.02, gamma=0.95, gain=1.04):
        x = np.clip(x, 0, 1)
        x = (x + lift) / (1.0 + lift)
        x = np.power(x, gamma)
        x = np.clip(x * gain, 0, 1)
        return x

    out2d = _filmic(out2d, lift=float(filmic_lift), gamma=float(filmic_gamma), gain=float(filmic_gain))

    # Back to RGB + tiny grain
    out = np.stack([out2d, out2d, out2d], axis=2)
    if dither_std and dither_std > 0:
        out = np.clip(out + np.random.normal(0, float(dither_std), out.shape).astype(np.float32), 0.0, 1.0)

    pil = numpy_to_pil(out).filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=6))
    return pil

# ---- Control preprocessors ----
def make_canny_cond(pil_img: Image.Image) -> Image.Image:
    # I build a simple Canny edge map for ControlNet.
    import cv2
    arr = (pil_to_numpy(pil_img)*255).astype(np.uint8)
    edges = cv2.Canny(arr, 100, 200)
    return Image.fromarray(edges).convert("RGB")

# ---- Pipeline factory ----
def build_pipeline(args):
    # I create SD 1.5 img2img, with optional Canny ControlNet.
    from diffusers import (
        StableDiffusionImg2ImgPipeline,
        StableDiffusionControlNetImg2ImgPipeline,
        ControlNetModel,
        DPMSolverMultistepScheduler,
        UniPCMultistepScheduler,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (device == "cuda") else torch.float32
    base = "runwayml/stable-diffusion-v1-5"

    if args.control == "canny":
        control_id = "lllyasviel/sd-controlnet-canny"
        controlnet = ControlNetModel.from_pretrained(control_id, torch_dtype=dtype)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(base, controlnet=controlnet, torch_dtype=dtype).to(device)
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(base, torch_dtype=dtype, safety_checker=None).to(device)

    # Scheduler
    if args.scheduler == "uni":
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # VAE slicing (saves memory)
    try: pipe.enable_vae_slicing()
    except Exception: pass

    # xFormers memory-efficient attention (CUDA only)
    if device == "cuda":
        want = args.mem_attn
        if want != "never":
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("[info] xFormers attention: ON")
            except Exception as e:
                if want == "always":
                    raise
                print(f"[info] xFormers not available; continuing without. ({e.__class__.__name__})")

    return pipe

# ---- Main ----
def main():
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    # I keep portrait strength modest so faces stay recognizable.
    if args.subject == "portrait" and args.strength > 0.28:
        print("[warn] For portraits, --strength > 0.28 may cause identity drift.")

    # Load & prep image
    inp = load_image(args.input, args.max_side)

    # Build pipeline
    pipe = build_pipeline(args)

    # Optional ControlNet input
    control_image = make_canny_cond(inp) if args.control == "canny" else None

    # Inference
    gen = torch.Generator(device=pipe.device).manual_seed(args.seed)
    common_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=inp,
        strength=args.strength,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        generator=gen
    )
    if control_image is not None:
        out = pipe(control_image=control_image, controlnet_conditioning_scale=float(args.control_scale), **common_kwargs)
    else:
        out = pipe(**common_kwargs)

    result = out.images[0]

    # Post grade to get the noir look
    noir = grade_noir(
        result,
        vignette=float(args.noir_vignette),
        halation=float(args.noir_halation),
        bloom_sigma=float(args.noir_bloom_sigma),
        bloom_thresh=float(args.noir_bloom_thresh),
        dither_std=float(args.noir_dither),
        filmic_lift=float(args.noir_lift),
        filmic_gamma=float(args.noir_gamma),
        filmic_gain=float(args.noir_gain),
    )

    Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)
    noir.save(args.output)
    print(f"[ok] Saved: {args.output}")

if __name__ == "__main__":
    main()
