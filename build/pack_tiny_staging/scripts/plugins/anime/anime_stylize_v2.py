#!/usr/bin/env python
# -*- coding: utf-8 -*-
# I turn a photo into an anime-style image (SD1.5/SD2.1), optionally using ControlNet.
# I keep comments short: what I pass in, how I pick models/control, and how I run seeds.

import os
import argparse
import warnings
from pathlib import Path
from typing import Optional, List

import numpy as np
from PIL import Image, ImageOps, ImageFilter

import torch

# --- OpenCV shims (I make sure basic cv2 ops exist before controlnet_aux imports) ---
try:
    import cv2
    if not hasattr(cv2, "INTER_AREA"):
        cv2.INTER_AREA = getattr(cv2, "INTER_LINEAR", 1)
    if not hasattr(cv2, "INTER_LANCZOS4"):
        cv2.INTER_LANCZOS4 = getattr(cv2, "INTER_CUBIC", 2)
    if not hasattr(cv2, "resize"):
        from PIL import Image as _PILImage
        import numpy as _np
        def _cv2_resize(arr, dsize, interpolation=None):
            pil = _PILImage.fromarray(arr)
            if interpolation == getattr(cv2, "INTER_LANCZOS4", None):
                resample = _PILImage.LANCZOS
            elif interpolation == getattr(cv2, "INTER_CUBIC", None):
                resample = _PILImage.BICUBIC
            elif interpolation == getattr(cv2, "INTER_NEAREST", None):
                resample = _PILImage.NEAREST
            else:
                resample = _PILImage.BILINEAR
            out = pil.resize(dsize, resample)
            return _np.array(out)
        cv2.resize = _cv2_resize
except Exception:
    cv2 = None
# -----------------------------------------------------------------------------------

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================== I/O & helpers ==============================
def load_image(path: str) -> Image.Image:
    # I always work in RGB.
    return Image.open(path).convert("RGB")

def resize_max_side(img: Image.Image, max_side: int = 640) -> Image.Image:
    # I keep aspect ratio and clamp the longest side.
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    s = max_side / float(max(w, h))
    return img.resize((int(w * s), int(h * s)), Image.LANCZOS)

def to_white_bg_black_lines(pil_img: Image.Image) -> Image.Image:
    # I normalize control maps to "white paper, black strokes".
    g = pil_img.convert("L")
    g = ImageOps.autocontrast(g)
    arr = np.array(g)
    if arr.mean() < 128:
        g = ImageOps.invert(g)
    g = ImageOps.autocontrast(g)
    return Image.merge("RGB", (g, g, g))

def ink_coverage(pil_img: Optional[Image.Image]) -> float:
    # I use this rough metric to avoid super-sparse line-art.
    if pil_img is None:
        return 0.0
    a = np.array(pil_img.convert("L"))
    return float((a < 240).mean())

# ============================== Model family utils ==============================
def attn_dim_for_base(repo_id: str) -> int:
    # I decide the cross-attention dim (768 for SD1.x; 1024 for SD2.1-like).
    if repo_id in {
        "cag/anything-v3-1",
        "runwayml/stable-diffusion-v1-5",
        "waifu-diffusion/wd-1-5-beta2",
    }:
        return 768
    return 768  # default

def pick_controlnet_repo(kind: str, attn_dim: int) -> Optional[str]:
    # I pick a ControlNet that matches the base model family.
    if attn_dim == 768:  # SD1.x
        mapping = {
            "lineart_anime": "lllyasviel/control_v11p_sd15s2_lineart_anime",
            "softedge":      "lllyasviel/control_v11p_sd15_softedge",
            "canny":         "lllyasviel/control_v11p_sd15_canny",
        }
    else:  # SD2.1-like
        mapping = {
            "lineart_anime": "thibaud/controlnet-sd21-lineart-diffusers",
            "softedge":      "thibaud/controlnet-sd21-hed-diffusers",
            "canny":         "thibaud/controlnet-sd21-canny-diffusers",
        }
    return mapping.get(kind)

# ============================== Control map builders ==============================
def build_control_map(init_image: Image.Image, control_kind: str, fast: bool) -> Optional[Image.Image]:
    # I generate a control image via controlnet_aux (lineart/hed) or OpenCV (canny).
    try:
        if control_kind == "lineart_anime":
            from controlnet_aux.lineart_anime import LineartAnimeDetector
            lad = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
            detect_res = 512 if fast else 768
            ctrl = lad(init_image, detect_resolution=detect_res, image_resolution=max(init_image.size))
            ctrl = to_white_bg_black_lines(ctrl)
        elif control_kind == "softedge":
            from controlnet_aux.hed import HEDdetector
            hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
            ctrl = hed(init_image)
            ctrl = to_white_bg_black_lines(ctrl)
        elif control_kind == "canny":
            if cv2 is None:
                from controlnet_aux.hed import HEDdetector
                hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
                ctrl = hed(init_image)
                ctrl = to_white_bg_black_lines(ctrl)
            else:
                g = cv2.cvtColor(np.array(init_image), cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(g, 80, 160)
                ctrl = Image.fromarray(edges)
                ctrl = to_white_bg_black_lines(ctrl)
        else:
            ctrl = None
    except Exception as e:
        print(f"[warn] controlnet_aux failed for {control_kind}: {e}")
        ctrl = None

    if ctrl is not None:
        ctrl = ctrl.resize(init_image.size, Image.LANCZOS)
    return ctrl

def ensure_compat(pipe) -> None:
    # I sanity-check text hidden size vs UNet cross-attn dim.
    txt = int(getattr(getattr(pipe, "text_encoder", object()), "config", object()).hidden_size
              if hasattr(getattr(pipe, "text_encoder", object()), "config") else 768)
    attn = int(getattr(getattr(pipe, "unet", object()), "config", object()).cross_attention_dim
               if hasattr(getattr(pipe, "unet", object()), "config") else txt)
    if txt != attn:
        raise ValueError(
            f"Incompatible model stack: text_encoder hidden_size={txt}, "
            f"unet cross_attention_dim={attn}. Use SD1.x base with SD1.5 ControlNets "
            f"or SD2.1 base with SD2.1 ControlNets."
        )

# ============================== Main ==============================
def parse_seed_list(s: Optional[str]) -> List[Optional[int]]:
    # I parse "7890,1234,none" into [7890,1234,None].
    if not s:
        return [None]
    out = []
    for part in s.split(","):
        part = part.strip()
        out.append(None if part == "" or part.lower() == "none" else int(part))
    return out

def main():
    p = argparse.ArgumentParser("Photo → Anime (SD1.5/SD2.1 + ControlNet)")
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", default="styled.png")
    p.add_argument("--model", "-m", choices=["primary", "trinart", "sd15"], default="primary",
                   help="primary=anything-v3-1, trinart=wd-1-5-beta2, sd15=runwayml/sd-v1-5")
    p.add_argument("--control", "-c", choices=["auto", "lineart_anime", "softedge", "canny", "none"], default="auto")
    p.add_argument("--lora", type=str, default=None, help="Only for --model sd15")

    # Core knobs
    p.add_argument("--strength", type=float, default=0.70)
    p.add_argument("--guidance", type=float, default=8.5)
    p.add_argument("--control-scale", type=float, default=None)
    p.add_argument("--steps", type=int, default=32)

    # Seeds and batching
    p.add_argument("--seed", type=int, default=None, help="Ignored if --seeds is given")
    p.add_argument("--seeds", type=str, default=None, help="Comma-separated list (e.g., 7890,1234)")
    p.add_argument("--out-prefix", type=str, default=None, help="Names outputs as <prefix>_s<seed>.png")

    # Runtime
    p.add_argument("--max-side", type=int, default=640)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--save-control", action="store_true")
    p.add_argument("--fast-detector", action="store_true")
    args = p.parse_args()

    # Load & resize input
    src = load_image(args.input)
    src = resize_max_side(src, args.max_side)

    # Choose base model
    if args.model == "primary":
        base = "cag/anything-v3-1"; base_attn = 768
    elif args.model == "trinart":
        base = "waifu-diffusion/wd-1-5-beta2"; base_attn = attn_dim_for_base(base)
    else:
        base = "runwayml/stable-diffusion-v1-5"; base_attn = 768

    # Pick control (AUTO tries lineart → softedge; fast mode flips order)
    requested = args.control
    controlnet_id: Optional[str] = None
    control_image: Optional[Image.Image] = None
    control_scale: Optional[float] = None

    if requested != "none":
        plan = [requested] if requested in {"lineart_anime", "softedge", "canny"} else ["lineart_anime", "softedge"]
        if requested == "auto" and args.fast_detector:
            plan = ["softedge", "lineart_anime"]

        for kind in plan:
            tmp_control_image = build_control_map(src, kind, fast=args.fast_detector)
            cov = ink_coverage(tmp_control_image)
            candidate_id = pick_controlnet_repo(kind, base_attn)
            if candidate_id is None:
                continue

            if kind == "lineart_anime":
                if cov < 0.015:
                    print(f"[auto] Anime-lineart too sparse ({cov:.3f}) → next")
                    continue
                control_scale = args.control_scale if args.control_scale is not None else 0.85
                controlnet_id, control_image = candidate_id, tmp_control_image
                break
            elif kind == "softedge":
                control_scale = args.control_scale if args.control_scale is not None else 1.05
                controlnet_id, control_image = candidate_id, tmp_control_image
                break
            elif kind == "canny":
                control_scale = args.control_scale if args.control_scale is not None else 1.0
                controlnet_id, control_image = candidate_id, tmp_control_image
                break
        else:
            print("[auto] No suitable control map; proceeding without ControlNet.")
            controlnet_id, control_image, control_scale = None, None, None

    # Build pipeline (I match dtype/device and reuse one pipe for all seeds)
    torch_dtype = torch.float16 if (torch.cuda.is_available() and not args.no_cuda) else torch.float32
    device = "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu"

    if controlnet_id:
        from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch_dtype)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            base, controlnet=controlnet, torch_dtype=torch_dtype, safety_checker=None, feature_extractor=None
        )
    else:
        from diffusers import StableDiffusionImg2ImgPipeline
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            base, torch_dtype=torch_dtype, safety_checker=None, feature_extractor=None
        )

    ensure_compat(pipe)

    # Memory/perf toggles
    pipe = pipe.to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[ok] xFormers attention enabled")
    except Exception as e:
        print(f"[warn] xFormers unavailable ({e}); using attention slicing")
        pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    # Optional LoRA for sd15
    if args.model == "sd15" and args.lora:
        try:
            print(f"[info] Loading LoRA: {args.lora}")
            pipe.load_lora_weights(args.lora)
        except Exception as e:
            print(f"[warn] Failed to load LoRA ({args.lora}): {e}")

    # Scheduler choice
    from diffusers import UniPCMultistepScheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Prompts (kept simple and safe for portraits)
    positive_prompt = (
        "masterpiece, best quality, anime illustration, clean bold lineart, cel shading, vivid colors, "
        "detailed anime eyes, smooth skin, studio portrait, soft background"
    )
    negative_prompt = (
        "lowres, blurry, bad anatomy, bad hands, extra fingers, text, watermark, "
        "photorealistic, photo, 3d, cgi, monochrome, grayscale, oversharp, noise"
    )

    # Common kwargs passed to the pipeline
    common_kwargs = dict(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        image=src,
        strength=float(args.strength),
        guidance_scale=float(args.guidance),
        num_inference_steps=int(args.steps),
    )
    if controlnet_id and control_image is not None:
        common_kwargs["control_image"] = control_image
        common_kwargs["controlnet_conditioning_scale"] = control_scale

    # Seed handling (single or batched)
    seed_list = parse_seed_list(args.seeds) if args.seeds else [args.seed]

    def out_name_for(seed_value: Optional[int]) -> str:
        # I decide the output name depending on --out-prefix and how many seeds we run.
        if args.out_prefix:
            return f"{args.out_prefix}_s{seed_value if seed_value is not None else 'rnd'}.png"
        if len(seed_list) == 1:
            return args.output
        stem = Path(args.output).with_suffix("").name
        return f"{stem}_s{seed_value if seed_value is not None else 'rnd'}.png"

    # Inference loop
    use_autocast = (device == "cuda" and torch_dtype == torch.float16)
    for s in seed_list:
        generator = None if s is None else torch.Generator(device=device).manual_seed(int(s))
        kwargs = dict(common_kwargs)
        kwargs["generator"] = generator

        out_path = out_name_for(s)
        print(
            f"Running… base={base} control={controlnet_id or 'none'} "
            f"strength={args.strength} cfg={args.guidance} c-scale={control_scale} "
            f"max-side={args.max_side} seed={s} → {out_path}"
        )

        with torch.inference_mode():
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = pipe(**kwargs).images[0]
            else:
                out = pipe(**kwargs).images[0]

        # I lightly sharpen the result to restore micro-contrast.
        out = out.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=8))

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.save(out_path)
        if args.save_control and control_image is not None:
            ctl_path = Path(out_path).with_name(Path(out_path).stem + "_control.png")
            control_image.save(ctl_path)

        print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    main()
