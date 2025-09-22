#!/usr/bin/env python
# -*- coding: utf-8 -*-
# I convert a photo into a neon cyberpunk look. I keep comments short and first-person.

import os, sys, argparse, traceback, warnings
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=FutureWarning)

from packaging import version
import diffusers as _df
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
)
import torch

# ---------- utils ----------
def load_image(path: str) -> Image.Image:
    # I always load as RGB.
    return Image.open(path).convert("RGB")

def resize_max_side(img: Image.Image, max_side=896) -> Image.Image:
    # I keep aspect ratio and cap the longest side.
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    s = max_side / float(max(w, h))
    return img.resize((int(w*s), int(h*s)), Image.LANCZOS)

def ensure_dir(p: Path):
    # I make sure the output folder exists.
    p.parent.mkdir(parents=True, exist_ok=True)

def round_to_multiple(x, m=8):
    return int(x // m * m)

def force_multiple_of_8(pil_img: Image.Image) -> Image.Image:
    # I snap size to /8 so SD is happy.
    w, h = pil_img.size
    w2, h2 = round_to_multiple(w), round_to_multiple(h)
    if (w2, h2) != (w, h):
        pil_img = pil_img.resize((max(8, w2), max(8, h2)), Image.LANCZOS)
    return pil_img

def to_3c(x: np.ndarray) -> np.ndarray:
    # I expand gray arrays to 3 channels.
    if x.ndim == 2: return np.repeat(x[..., None], 3, axis=2)
    if x.ndim == 3 and x.shape[2] == 1: return np.repeat(x, 3, axis=2)
    return x

def gaussian_blur_keepdims(x: np.ndarray, sigma: float) -> np.ndarray:
    # I blur without changing shape.
    import cv2
    y = cv2.GaussianBlur(x, (0,0), sigma)
    if y.ndim == 2: y = y[..., None]
    return y.astype(np.float32)

def collage_hstack(imgs, pad=8):
    # I tile multiple style refs side by side (nice for IP-Adapter).
    if not imgs: return None
    h = min(i.height for i in imgs)
    rs = [i.resize((int(i.width * h / i.height), h), Image.LANCZOS) for i in imgs]
    w_total = sum(i.width for i in rs) + pad*(len(rs)-1)
    out = Image.new("RGB", (w_total, h), (0,0,0))
    x = 0
    for k,i in enumerate(rs):
        out.paste(i, (x, 0))
        x += i.width + (pad if k < len(rs)-1 else 0)
    return out

# ------------- Masks -------------
def load_or_make_person_mask(img: Image.Image, mask_path: str|None, auto_person: bool) -> Image.Image|None:
    # I try user mask → rembg → None (later I fallback to an ellipse).
    if mask_path:
        try:
            m = Image.open(mask_path).convert("L").resize(img.size, Image.LANCZOS)
            return m
        except Exception:
            pass
    if auto_person:
        try:
            from rembg import remove
            rgba = remove(img)
            if isinstance(rgba, (bytes, bytearray)):
                from io import BytesIO
                rgba = Image.open(BytesIO(rgba)).convert("RGBA")
            elif not isinstance(rgba, Image.Image):
                rgba = Image.fromarray(np.array(rgba)).convert("RGBA")
            return rgba.split()[-1]
        except Exception:
            return None
    return None

def make_center_ellipse_mask(img: Image.Image) -> Image.Image:
    # I draw a simple centered ellipse as a safe portrait mask.
    w, h = img.size
    m = Image.new("L", (w, h), 0)
    pad_w, pad_h = int(0.14*w), int(0.10*h)
    box = (pad_w, pad_h, w - pad_w, int(h*0.82))
    ImageDraw.Draw(m).ellipse(box, fill=255)
    return m

def invert_mask(m: Image.Image) -> Image.Image:
    return ImageOps.invert(m)

# ----------- Control builders -----------
def control_image_softedge(img: Image.Image) -> Image.Image|None:
    # I prefer HED for people/background edges.
    try:
        from controlnet_aux.hed import HEDdetector
        hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
        hed_img = hed(img)
        return ImageOps.autocontrast(hed_img).convert("RGB")
    except Exception as e:
        print(f"[warn] HED annotator unavailable: {e}")
        return None

def control_image_depth(img: Image.Image) -> Image.Image|None:
    # I use depth for scene structure.
    try:
        from controlnet_aux.midas import MidasDetector
        midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
        dep = midas(img)
        return ImageOps.autocontrast(dep).convert("RGB")
    except Exception as e:
        print(f"[warn] MiDaS annotator unavailable: {e}")
        return None

def control_image_canny(img: Image.Image) -> Image.Image|None:
    # I fallback to OpenCV canny if needed.
    try:
        import cv2
        g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        e = cv2.Canny(g, 80, 160)
        return Image.merge("RGB", (Image.fromarray(e),)*3)
    except Exception as e:
        print(f"[warn] OpenCV canny failed: {e}")
        return None

def pick_controlnet_repo(kind: str) -> str|None:
    # I map friendly names to SD1.5 ControlNets.
    mapping = {
        "depth":   "lllyasviel/control_v11f1p_sd15_depth",
        "softedge":"lllyasviel/sd-controlnet-hed",
        "canny":   "lllyasviel/control_v11p_sd15_canny",
    }
    return mapping.get(kind)

def soft_erode_mask(m: np.ndarray, k=5, it=1):
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    return cv2.erode(m, kernel, iterations=it)

def get_skin_mask_rgb(img: Image.Image) -> Image.Image:
    # I estimate skin in HSV so I can protect it later.
    import cv2
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0,  20,  60], dtype=np.uint8)
    upper = np.array([35, 180, 255], dtype=np.uint8)
    m1 = cv2.inRange(hsv, lower, upper).astype(np.uint8)
    m1 = cv2.GaussianBlur(m1, (0,0), 3.0)
    m1 = (m1 > 32).astype(np.uint8)*255
    m1 = soft_erode_mask(m1, k=5, it=1)
    m1 = cv2.GaussianBlur(m1, (0,0), 2.0)
    return Image.fromarray(m1, mode="L")

def composite_with_mask(fg: Image.Image, bg: Image.Image, mask: Image.Image, alpha=0.7) -> Image.Image:
    # I blend two images with a soft mask.
    fg = np.asarray(fg).astype(np.float32)
    bg = np.asarray(bg).astype(np.float32)
    m = np.asarray(mask).astype(np.float32)[..., None] / 255.0
    m = np.clip(m*alpha, 0.0, 1.0)
    out = m*fg + (1.0-m)*bg
    return Image.fromarray(out.astype(np.uint8))

# ----------- Grading ----------
def _thin_edges(e: np.ndarray, hi_q=0.95) -> np.ndarray:
    # I keep only the thinnest, strongest edges for glow.
    import cv2
    t = float(np.quantile(e, hi_q))
    e = (e >= t).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    e = cv2.morphologyEx(e, cv2.MORPH_OPEN, k, iterations=1)
    e = cv2.dilate(e, k, iterations=1)
    e = cv2.GaussianBlur(e.astype(np.float32), (0,0), 1.2)
    return e

def grade_cyberpunk(
    pil_img: Image.Image,
    edges_for_glow: Image.Image|None,
    bg_mask_for_edges: Image.Image|None = None,
    neon: float = 0.30,
    bloom: float = 0.34,
    scanlines: float = 0.0,
    protect_skin: bool = True,
    tone_mix: float = 0.10,
    glow_sigma: float = 2.0,
    bloom_sigma: float = 5.0,
    bloom_thresh: float = 0.75,
    ca_px: int = 0,
    edge_q: float = 0.985,
    skin_suppress: float = 0.95,
    add_dither: bool = True,
) -> Image.Image:
    # I push teal/magenta, add edge glow + bloom, and keep skin safe.
    import cv2
    arr = np.asarray(pil_img).astype(np.float32) / 255.0
    h, w, _ = arr.shape

    gray = np.mean(arr, axis=2, keepdims=True)
    w_teal = np.clip(1.0 - 1.4*gray, 0.0, 1.0)
    w_mag  = np.clip(1.4*gray - 0.2, 0.0, 1.0)
    teal    = np.array([0.0,1.0,1.0], dtype=np.float32)[None,None,:]
    magenta = np.array([1.0,0.0,0.9], dtype=np.float32)[None,None,:]
    toned = np.clip(arr*(1.0 - tone_mix) + (w_teal*teal + w_mag*magenta)*tone_mix, 0.0, 1.0)

    if edges_for_glow is not None:
        e_img = edges_for_glow.resize((w, h), Image.LANCZOS).convert("L")
        e = np.asarray(e_img).astype(np.float32)/255.0
    else:
        g = cv2.cvtColor((toned*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        e = cv2.Canny(g, 80, 160).astype(np.float32)/255.0

    if bg_mask_for_edges is not None:
        bg = ImageOps.invert(bg_mask_for_edges).resize((w,h), Image.NEAREST)
        bg = np.asarray(bg).astype(np.float32)/255.0
        e = e * (bg > 0.5).astype(np.float32)

    e = _thin_edges(e, hi_q=edge_q)
    e = gaussian_blur_keepdims(e, 1.2)

    if protect_skin:
        hsv = cv2.cvtColor((arr*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        H, S, V = hsv[...,0], hsv[...,1]/255.0, hsv[...,2]/255.0
        skin = ((H >= 0) & (H <= 50) & (S > 0.10) & (S < 0.68) & (V > 0.2) & (V < 0.95)).astype(np.float32)
        skin = gaussian_blur_keepdims(skin, 3.0)
        e = e * (1.0 - skin_suppress * skin)

    y = (np.arange(h, dtype=np.float32) / max(h - 1, 1))[:, None]
    grad_mag  = np.clip(1.2*y - 0.1, 0.0, 1.0)
    grad_teal = 1.0 - grad_mag
    glow_color = np.stack([
        1.0*grad_mag + 0.0*grad_teal,
        0.10*grad_mag + 1.0*grad_teal,
        0.90*grad_mag + 1.0*grad_teal,
    ], axis=2)
    glow = gaussian_blur_keepdims(e * glow_color, glow_sigma)

    lum = np.max(toned, axis=2, keepdims=True)
    bloom_map = np.clip(lum - bloom_thresh, 0.0, 1.0)
    bloom_map = gaussian_blur_keepdims(bloom_map, bloom_sigma)
    bloom_rgb = to_3c(bloom_map)

    def screen(a, b, k): return 1.0 - (1.0 - a) * (1.0 - np.clip(k,0,1)*np.clip(b,0,1))
    out = screen(toned, glow,  neon)
    out = screen(out,  bloom_rgb, bloom)

    if scanlines > 0:
        yy = np.arange(h, dtype=np.float32)[:, None]
        lines = 0.5*(1.0 + np.sin(2.0*np.pi*yy/3.0))
        mask = 1.0 - float(scanlines)*(1.0 - lines)
        mask = np.broadcast_to(mask[:, :, None], (h, w, 1)).astype(np.float32)
        out = np.clip(out * mask, 0.0, 1.0)

    if ca_px > 0:
        out_ca = out.copy()
        out_ca[...,0] = np.roll(out[...,0], -ca_px, axis=1)
        out_ca[...,2] = np.roll(out[...,2],  ca_px, axis=1)
        out = out_ca

    blur = cv2.GaussianBlur(out, (0,0), 0.8)
    hp = np.clip(out - blur, 0.0, 1.0)
    weight = np.clip(np.max(hp, axis=2, keepdims=True)*1.5, 0.0, 1.0)
    out = np.clip(out + weight*0.30*(out - blur), 0.0, 1.0)

    if add_dither:
        noise = np.random.normal(0, 0.002, out.shape).astype(np.float32)
        out = np.clip(out + noise, 0.0, 1.0)

    return Image.fromarray((out*255).astype(np.uint8))

# ------------- main -------------
def main():
    # I expose simple flags; portraits do a background inpaint first.
    p = argparse.ArgumentParser("Photo → Cyberpunk (v4)")
    p.add_argument("-i","--input", required=True)
    p.add_argument("-o","--output", default="styled.png")
    p.add_argument("--subject", choices=["portrait","scene"], default="portrait")
    p.add_argument("--base", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--scheduler", choices=["unipc","dpmpp"], default="dpmpp")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-side", type=int, default=1024)

    # Style refs
    p.add_argument("--style-image", type=str, default=None, help="One or more paths, comma-separated")
    p.add_argument("--style-strength", type=float, default=None)

    # Control / strength
    p.add_argument("--control", choices=["auto","depth","softedge","canny","none"], default="auto")
    p.add_argument("--control-scale", type=float, default=None)
    p.add_argument("--strength", type=float, default=None)
    p.add_argument("--refine", action="store_true")
    p.add_argument("--refine-strength", type=float, default=None)

    # Masks & inpaint (portraits)
    p.add_argument("--mask", type=str, default=None, help="Subject mask (white=subject)")
    p.add_argument("--auto-mask-person", action="store_true")
    p.add_argument("--force-inpaint", action="store_true")

    # Grade opts
    p.add_argument("--grade-only", action="store_true")
    p.add_argument("--neon", type=float, default=None)
    p.add_argument("--bloom", type=float, default=None)
    p.add_argument("--edge-q", type=float, default=None)
    p.add_argument("--skin-suppress", type=float, default=None)
    p.add_argument("--scanlines", type=float, default=0.0)
    p.add_argument("--rim-boost", type=float, default=0.22, help="Extra rim glow for portraits (0..1)")
    p.add_argument("--skin-keep", type=float, default=0.65)

    args = p.parse_args()

    # I/O
    src = resize_max_side(load_image(args.input), args.max_side)

    style_imgs = None
    if args.style_image:
        paths = [s.strip() for s in args.style_image.split(",") if s.strip()]
        style_imgs = [resize_max_side(load_image(pth), args.max_side) for pth in paths]
    style_collage = collage_hstack(style_imgs) if style_imgs and len(style_imgs) > 1 else (style_imgs[0] if style_imgs else None)

    # Grade-only path
    if args.grade_only:
        graded = grade_cyberpunk(
            src, edges_for_glow=None, bg_mask_for_edges=None,
            neon=args.neon or 0.30, bloom=args.bloom or 0.34,
            edge_q=args.edge_q or 0.985, skin_suppress=args.skin_suppress or 0.95,
            scanlines=args.scanlines
        )
        ensure_dir(Path(args.output))
        graded.save(args.output); print(f"✅ Saved (grade-only): {args.output}")
        return

    # Device / dtype
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Defaults per subject
    if args.subject == "portrait":
        steps    = args.steps    if args.steps    is not None else 34
        guidance = args.guidance if args.guidance is not None else 6.0
        strength = args.strength if args.strength is not None else 0.18
        style_strength = args.style_strength if args.style_strength is not None else 0.50
        edge_q = args.edge_q if args.edge_q is not None else 0.985
        neon = args.neon if args.neon is not None else 0.28
        bloom = args.bloom if args.bloom is not None else 0.34
        skin_suppress = args.skin_suppress if args.skin_suppress is not None else 0.95
        control_policy = "softedge"
    else:
        steps    = args.steps    if args.steps    is not None else 38
        guidance = args.guidance if args.guidance is not None else 6.5
        strength = args.strength if args.strength is not None else 0.82
        style_strength = args.style_strength if args.style_strength is not None else 0.70
        edge_q = args.edge_q if args.edge_q is not None else 0.95
        neon = args.neon if args.neon is not None else 0.26
        bloom = args.bloom if args.bloom is not None else 0.30
        skin_suppress = args.skin_suppress if args.skin_suppress is not None else 0.96
        control_policy = "depth"

    # Prompts
    if args.subject == "portrait":
        ppos = ("cyberpunk portrait, neon rim light, teal/magenta colored fog and bokeh, photographic, "
                "realistic skin texture, soft volumetric haze, filmic contrast, sharp eyes, natural skin, "
                "cinematic background")
        pneg = ("text, watermark, overprocessed skin, plastic/waxy skin, harsh sharpening, lowres, blurry, "
                "extra fingers, 3d render, cartoon, anime, toy look, heavy artifacts")
    else:
        ppos = ("rainy neon megacity at night, wet asphalt mirror reflections, dense holographic billboards, "
                "backlit fog shafts, layered depth, busy signage, cinematic framing, glassy highlights, rim-lit silhouettes")
        pneg = ("text, watermark, garish oversaturation, flat lighting, low detail, heavy vignette, blurry")

    # Stage-1 (portraits): background inpaint
    bg_mask = None
    stage1_img = src
    subj_mask = None

    if args.subject == "portrait":
        subj_mask = load_or_make_person_mask(src, args.mask, True)
        if subj_mask is None:
            subj_mask = make_center_ellipse_mask(src)
            print("[info] rembg not available; using dummy ellipse subject mask.")
        bg_mask = invert_mask(subj_mask)

        try:
            inpaint = StableDiffusionInpaintPipeline.from_pretrained(
                args.base, torch_dtype=torch_dtype, safety_checker=None
            ).to(device)
            inpaint.scheduler = (
                DPMSolverMultistepScheduler if args.scheduler == "dpmpp" else UniPCMultistepScheduler
            ).from_config(inpaint.scheduler.config)

            s1_strength = 0.70
            s1_steps = max(32, int(steps))
            stage1_img = inpaint(
                prompt=("neon cyberpunk city backdrop, magenta and teal signage, rain bokeh, colored fog, cinematic depth"),
                negative_prompt="text, watermark, heavy vignette, plastic look",
                image=src,
                mask_image=bg_mask,
                strength=s1_strength,
                guidance_scale=5.8,
                num_inference_steps=s1_steps,
            ).images[0]
            print("Stage-1 background inpaint done.")
            stage1_img = force_multiple_of_8(stage1_img)
        except Exception:
            traceback.print_exc()
            print("[warn] Inpaint stage failed; continuing with original background.")
            stage1_img = src

    # ControlNet (auto or chosen)
    control_choice = args.control
    if control_choice == "auto":
        control_choice = (control_policy if args.subject=="scene" else "softedge")
    controlnet_id, control_img = None, None
    if control_choice != "none":
        if control_choice == "depth":
            control_img = control_image_depth(stage1_img)
        elif control_choice == "softedge":
            control_img = control_image_softedge(stage1_img)
        elif control_choice == "canny":
            control_img = control_image_canny(stage1_img)
        controlnet_id = pick_controlnet_repo(control_choice) if control_img is not None else None

        if control_img is not None:
            control_img = force_multiple_of_8(control_img.resize(stage1_img.size, Image.LANCZOS))
            stage1_img   = force_multiple_of_8(stage1_img)

    # Build pipeline
    try:
        if controlnet_id:
            controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch_dtype)
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                args.base, controlnet=controlnet, torch_dtype=torch_dtype,
                safety_checker=None, feature_extractor=None,
            )
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                args.base, torch_dtype=torch_dtype, safety_checker=None, feature_extractor=None
            )
        pipe.scheduler = (DPMSolverMultistepScheduler if args.scheduler=="dpmpp" else UniPCMultistepScheduler).from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
    except Exception:
        traceback.print_exc(); sys.exit(2)

    # IP-Adapter (style refs)
    DIFF_OK_FOR_IP = version.parse(_df.__version__) >= version.parse("0.35.0")
    use_style = False
    if style_collage is not None and DIFF_OK_FOR_IP:
        try:
            try: pipe.disable_attention_slicing()
            except Exception: pass
            pipe.load_ip_adapter(
                "h94/IP-Adapter", subfolder="models",
                weight_name="ip-adapter_sd15.safetensors",
                torch_dtype=torch_dtype,
            )
            pipe.set_ip_adapter_scale([float(style_strength)])
            use_style = True
            print(f"IP-Adapter loaded, style_strength={style_strength}")
        except Exception as e:
            print(f"[warn] IP-Adapter load failed: {e}")
    if not use_style:
        try: pipe.enable_attention_slicing("auto")
        except Exception: pass

    # Seed
    generator = (torch.Generator(device=device).manual_seed(args.seed) if args.seed else None)

    # Control scale defaults
    if args.subject == "scene":
        control_scale = args.control_scale if args.control_scale is not None else 0.45
    else:
        control_scale = args.control_scale if args.control_scale is not None else 0.22

    # Stage-2 (main stylization)
    kwargs = dict(
        prompt=ppos, negative_prompt=pneg,
        image=stage1_img, strength=float(strength),
        guidance_scale=float(guidance), num_inference_steps=int(steps),
        generator=generator,
    )
    if controlnet_id and control_img is not None:
        kwargs["control_image"] = control_img
        kwargs["controlnet_conditioning_scale"] = float(control_scale)
    if use_style:
        kwargs["ip_adapter_image"] = [style_collage]
    print(f"Running Stage-2… subject={args.subject} strength={strength} cfg={guidance} steps={steps} control={control_choice}/{bool(controlnet_id)} style={use_style}")
    try:
        result = pipe(**kwargs).images[0]
    except Exception:
        traceback.print_exc(); sys.exit(3)

    # Optional refine pass
    if args.refine:
        r_strength = args.refine_strength if args.refine_strength is not None else (0.22 if args.subject=="portrait" else 0.28)
        ref_kwargs = dict(
            prompt=ppos, negative_prompt=pneg, image=result,
            strength=float(r_strength), guidance_scale=float(guidance),
            num_inference_steps=20, generator=generator
        )
        if controlnet_id and control_img is not None:
            ref_kwargs["control_image"] = control_img
            ref_kwargs["controlnet_conditioning_scale"] = float(control_scale*0.9)
        if use_style and style_collage is not None:
            ref_kwargs["ip_adapter_image"] = [style_collage]
        result = pipe(**ref_kwargs).images[0]

    # Skin keep (blend some original skin back)
    if args.subject == "portrait" and subj_mask is not None and args.skin_keep > 0:
        skin_mask = get_skin_mask_rgb(src).resize(result.size, Image.BILINEAR)
        subj_mask_res = subj_mask.resize(result.size, Image.NEAREST)
        m = Image.fromarray(((np.array(skin_mask) > 80) & (np.array(subj_mask_res) > 128)).astype(np.uint8) * 255, mode="L")
        m = m.filter(ImageFilter.GaussianBlur(radius=2.0))
        result = composite_with_mask(src.resize(result.size, Image.LANCZOS), result, m, alpha=float(args.skin_keep))

    # Edges/masks for final glow
    edges_for_glow = None
    bg_mask_for_edges = None
    if args.subject == "scene":
        edges_for_glow = control_img if (control_img is not None and control_choice != "canny") else None
    else:
        bg_mask_for_edges = bg_mask if bg_mask is not None else None

    # Optional colored rim glow (portraits)
    rim_rgb = None
    if args.subject == "portrait" and subj_mask is not None and args.rim_boost > 0:
        import cv2
        rim = np.array(subj_mask.resize(result.size, Image.NEAREST))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        outer = cv2.dilate(rim, k, iterations=1).astype(np.float32)
        inner = cv2.erode(rim,  k, iterations=1).astype(np.float32)
        rim_band = np.clip((outer - inner)/255.0, 0, 1)
        h, w = rim_band.shape
        y = (np.arange(h, dtype=np.float32) / max(h-1,1))[:, None]
        grad_mag  = np.clip(1.2*y - 0.1, 0.0, 1.0)
        grad_teal = 1.0 - grad_mag
        glow_color = np.stack([
            1.0*grad_mag + 0.0*grad_teal,
            0.10*grad_mag + 1.0*grad_teal,
            0.90*grad_mag + 1.0*grad_teal,
        ], axis=2)
        rim_rgb = gaussian_blur_keepdims(rim_band[...,None]*glow_color, 2.0) * float(args.rim_boost)

    if rim_rgb is not None:
        base = np.asarray(result).astype(np.float32)/255.0
        base = np.clip(base + rim_rgb, 0, 1)
        result = Image.fromarray((base*255).astype(np.uint8))

    # Final grade + light sharpen
    graded = grade_cyberpunk(
        result,
        edges_for_glow=edges_for_glow,
        bg_mask_for_edges=bg_mask_for_edges,
        neon=float(neon), bloom=float(bloom), scanlines=float(args.scanlines),
        edge_q=float(edge_q), skin_suppress=float(skin_suppress),
        ca_px=(0 if args.subject=="portrait" else 1)
    ).filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=6))

    ensure_dir(Path(args.output))
    graded.save(args.output)
    print(f"✅ Saved: {args.output}")

if __name__ == "__main__":
    main()
