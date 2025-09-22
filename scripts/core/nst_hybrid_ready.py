#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- Phase-1 Determinism Guard (kept, de-duped a bit) ---
import os, random
import numpy as np
from pathlib import Path

import torch
SEED = int(os.environ.get("NST_SEED", "77"))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
try:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
except Exception:
    pass
# --- end determinism guard ---

import argparse, json, time
import cv2  # pip install opencv-contrib-python==4.12.0.88 (optional; fallback ok)
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
import torchvision.transforms as T
from torchvision import models
import torch.nn.functional as F
from skimage import color  # pip install scikit-image

# -----------------------
# Normalization for VGG
# -----------------------
IMNORM = T.Normalize([0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225])

# === Backbone layer maps (names -> feature indices) ===
VGG19_MAP = {
    "conv1_1":0, "relu1_1":1, "conv1_2":2, "relu1_2":3, "pool1":4,
    "conv2_1":5, "relu2_1":6, "conv2_2":7, "relu2_2":8, "pool2":9,
    "conv3_1":10,"relu3_1":11,"conv3_2":12,"relu3_2":13,"conv3_3":14,"relu3_3":15,"conv3_4":16,"relu3_4":17,"pool3":18,
    "conv4_1":19,"relu4_1":20,"conv4_2":21,"relu4_2":22,"conv4_3":23,"relu4_3":24,"conv4_4":25,"relu4_4":26,"pool4":27,
    "conv5_1":28,"relu5_1":29,"conv5_2":30,"relu5_2":31,"conv5_3":32,"relu5_3":33,"conv5_4":34,"relu5_4":35,"pool5":36
}
VGG16_MAP = {
    "conv1_1":0, "relu1_1":1, "conv1_2":2, "relu1_2":3, "pool1":4,
    "conv2_1":5, "relu2_1":6, "conv2_2":7, "relu2_2":8, "pool2":9,
    "conv3_1":10,"relu3_1":11,"conv3_2":12,"relu3_2":13,"conv3_3":14,"relu3_3":15,"pool3":16,
    "conv4_1":17,"relu4_1":18,"conv4_2":19,"relu4_2":20,"conv4_3":21,"relu4_3":22,"pool4":23,
    "conv5_1":24,"relu5_1":25,"conv5_2":26,"relu5_2":27,"conv5_3":28,"relu5_3":29,"pool5":30
}

# -----------------------
# I/O helpers
# -----------------------
def load_rgb(path):  # robust open
    return Image.open(path).convert("RGB")

def resize_keep_aspect_crop(pil, size):
    # center-crop to square, then resize
    w, h = pil.size
    side = min(w, h)
    left = (w - side)//2
    top  = (h - side)//2
    pil = pil.crop((left, top, left+side, top+side))
    return pil.resize((size, size), Image.Resampling.LANCZOS)

def parse_layer_list(s):
    """
    Accept comma-separated names (e.g., 'conv4_1,conv5_1') or numeric indices '19,28' (vgg19 only).
    Returns list[str] OR list[int].
    """
    s = s.strip().replace(" ", "")
    items = [x for x in s.split(",") if x]
    if all(x.isdigit() for x in items):
        return [int(x) for x in items]
    return items

def build_backbone(backbone: str):
    if backbone == "vgg19":
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        layer_map = VGG19_MAP
    else:
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        layer_map = VGG16_MAP
    for p in vgg.parameters():
        p.requires_grad_(False)
    return vgg, layer_map

def pil_to_01_keep(pil, size=None):
    if size:
        pil = resize_keep_aspect_crop(pil, size)
    return T.ToTensor()(pil).unsqueeze(0)

def save_01(t, path):
    im = t.detach().clamp(0,1).cpu().squeeze(0)
    T.ToPILImage()(im).save(path)

# -----------------------
# Face-mask helpers (auto)
# -----------------------
def center_crop_square(pil):
    w, h = pil.size
    side = min(w, h)
    left = (w - side)//2
    top  = (h - side)//2
    return pil.crop((left, top, left+side, top+side))

def resize_mask_crop(pil_mask, size, soften=True):
    m = center_crop_square(pil_mask).convert("L")
    m = m.resize((size, size), Image.Resampling.LANCZOS)
    if soften:
        from PIL import ImageFilter
        m = m.filter(ImageFilter.GaussianBlur(radius=size/256))
    return m

def auto_face_mask_for_image(content_path, save_path=None, verbose=True):
    try:
        import cv2
    except Exception:
        print("[mask] OpenCV not available; skipping auto face mask.")
        return None
    pil = load_rgb(content_path)
    sq  = center_crop_square(pil)
    arr = np.asarray(sq.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64)
    )
    if len(faces) == 0:
        if verbose: print("[mask] no face detected; proceeding without mask.")
        return None
    x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
    mask = np.zeros((sq.height, sq.width), dtype=np.uint8)
    cx, cy = x + w//2, y + h//2
    ax, ay = int(w * 0.62), int(h * 0.82)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, thickness=-1)
    k = max(3, (min(mask.shape)//128)|1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    m_pil = Image.fromarray(mask, mode="L")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        m_pil.save(save_path)
        if verbose: print(f"[mask] saved auto mask -> {save_path}")
    return m_pil

# -----------------------
# Optional colour pre-match (LAB)
# -----------------------
def color_transfer_lab(src_pil, ref_p_pil, strength=0.5):
    src = np.asarray(src_pil) / 255.0
    ref = np.asarray(ref_p_pil) / 255.0
    src_lab = color.rgb2lab(src)
    ref_lab = color.rgb2lab(ref)
    for i in range(3):
        s = src_lab[..., i]; r = ref_lab[..., i]
        s_std = s.std() + 1e-6
        src_lab[..., i] = (s - s.mean()) * ((r.std()+1e-6) / s_std) + r.mean()
    out = np.clip(color.lab2rgb(src_lab), 0, 1)
    out = strength*out + (1-strength)*(np.asarray(src_pil)/255.0)
    return Image.fromarray((out*255).astype(np.uint8))

# -----------------------
# Laplacian edge helpers
# -----------------------
def laplacian3x3(x):
    k = torch.tensor([[0, 1, 0],
                      [1,-4, 1],
                      [0, 1, 0]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    k = k.repeat(x.shape[1], 1, 1, 1)                # per-channel
    x_pad = F.pad(x, (1,1,1,1), mode="replicate")
    return F.conv2d(x_pad, k, groups=x.shape[1])

def edge_loss_l1(x01, c01, face_mask_01=None, face_down=0.5):
    Lx = laplacian3x3(x01)
    Lc = laplacian3x3(c01)
    diff = (Lx - Lc).abs()
    if face_mask_01 is not None:
        w = 1.0 - (1.0 - float(face_down)) * face_mask_01  # 1 outside, face_down inside
        diff = diff * w
    return diff.mean()

# -----------------------
# Losses
# -----------------------
def tv_loss(x):
    dx = (x[:,:,1:,:] - x[:,:,:-1,:]).abs().mean()
    dy = (x[:,:,:,1:] - x[:,:,:,:-1]).abs().mean()
    return dx + dy

def gram_matrix(fm):
    b,c,h,w = fm.shape
    X = fm.view(b, c, h*w)
    return (X @ X.transpose(1,2)) / (c*h*w)

# -----------------------
# VGG features
# -----------------------
class VGGFeatures(nn.Module):
    def __init__(self, vgg, content_layer_id: int, style_layer_ids):
        super().__init__()
        self.vgg = vgg
        self.content_layer_id = int(content_layer_id)
        self.style_layer_ids = set(int(i) for i in style_layer_ids)

    def forward(self, x01):  # x01 in [0,1]
        x = IMNORM(x01.squeeze(0)).unsqueeze(0)
        content_feat = None
        style_feats = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i == self.content_layer_id:
                content_feat = x
            if i in self.style_layer_ids:
                style_feats[i] = x
        return content_feat, style_feats

# -----------------------
# One optimization stage
# -----------------------
def run_stage(x01, c01, s01, steps, opt_name, lr,
              vgg, content_layer_id, style_layer_ids,
              w_c, w_s, w_tv, device,
              face_mask_01=None, w_face=0.0,
              edge_w=0.0, edge_face_down=0.5):
    model = VGGFeatures(vgg, content_layer_id, style_layer_ids).to(device).eval()
    mse = nn.MSELoss()

    with torch.no_grad():
        c_feat_c, _   = model(c01)
        _,   s_feats_s = model(s01)
        grams_s        = {l: gram_matrix(s_feats_s[l]) for l in style_layer_ids}

    x01 = nn.Parameter(x01)

    if opt_name == "lbfgs":
        optimizer = optim.LBFGS([x01], max_iter=steps, line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad(set_to_none=True)
            x_cl = x01.clamp(0,1)
            c_feat, s_feats = model(x_cl)
            c_loss = mse(c_feat, c_feat_c)
            s_loss = sum(mse(gram_matrix(s_feats[l]), grams_s[l]) for l in style_layer_ids)
            tv     = tv_loss(x_cl)
            e_loss = edge_loss_l1(x_cl, c01, face_mask_01, edge_face_down)
            loss = w_c*c_loss + w_s*s_loss + w_tv*tv + edge_w*e_loss
            if (w_face>0) and (face_mask_01 is not None):
                pix = ((x_cl - c01) * face_mask_01).pow(2).mean()
                loss = loss + w_face*pix
            loss.backward()
            return loss
        optimizer.step(closure)
    else:
        optimizer = optim.Adam([x01], lr=lr)
        for t in range(steps):
            optimizer.zero_grad(set_to_none=True)
            x_cl = x01.clamp(0,1)
            c_feat, s_feats = model(x_cl)
            c_loss = mse(c_feat, c_feat_c)
            s_loss = sum(mse(gram_matrix(s_feats[l]), grams_s[l]) for l in style_layer_ids)
            k  = 1.0 - (t+1)/steps
            tv = tv_loss(x_cl) * (0.6*k + 0.4)
            e_loss = edge_loss_l1(x_cl, c01, face_mask_01, edge_face_down)
            loss = w_c*c_loss + w_s*s_loss + w_tv*tv + edge_w*e_loss
            if (w_face>0) and (face_mask_01 is not None):
                pix = ((x_cl - c01) * face_mask_01).pow(2).mean()
                loss = loss + w_face*pix
            loss.backward()
            optimizer.step()

    return x01.clamp(0,1).detach()

# -----------------------
# Style-layer alias support
# -----------------------
ALIAS_STYLE_LAYERS = {
    # convenience aliases used in your experiments / README
    "layers45":  "conv4_1,conv5_1",
    "layers345": "conv3_1,conv4_1,conv5_1",
    "layersA":   "conv1_1,conv2_1,conv3_1,conv4_1,conv5_1",  # Gatys classic
    "layersB":   "conv3_1,conv4_1",                          # lean
    "layersC":   "conv3_1,conv4_1,conv5_1",                  # mid
}
def normalize_style_layers(spec: str) -> str:
    s = (spec or "").strip()
    return ALIAS_STYLE_LAYERS.get(s, s)

# -----------------------
# Full pipeline (multi-size / multi-step) + timing
# -----------------------
def run_nst(content_path, style_path, out_path,
            sizes=(384,768), steps=(400,500),
            backbone="vgg19",
            content_layer="conv4_2",
            style_layers="conv4_1,conv5_1",
            style_weight=1.2e4, content_weight=1.0, tv_weight=2.5e-3,
            opt="adam", lr=0.01, device="cuda",
            face_mask_path=None, face_preserve=0.65,
            color_prematch_strength=0.5, seed=123,
            edge_w=0.0, edge_face_down=0.5,
            init_path=None):

    torch.manual_seed(seed)
    device = torch.device(device if (device=="cpu" or torch.cuda.is_available()) else "cpu")
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Build backbone and pick layer map
    vgg, LMAP = build_backbone(backbone)
    vgg = vgg.to(device).eval()

    # Parse layer specs (accept names, numeric, or aliases)
    def to_index(item):
        if isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
            if backbone != "vgg19":
                raise ValueError("Numeric layer indices only supported for vgg19. Use names for vgg16.")
            return int(item)
        if item not in LMAP:
            raise ValueError(f"Unknown layer name '{item}' for backbone {backbone}.")
        return LMAP[item]

    style_spec = normalize_style_layers(style_layers)
    style_items = parse_layer_list(style_spec)
    style_layer_ids = [to_index(x) for x in style_items]
    content_item = normalize_style_layers(content_layer)  # allow aliases, though typically a single name
    content_layer_id = to_index(content_item if isinstance(content_item, str) else content_layer)

    # Load images (+ optional color pre-match)
    c_pil = load_rgb(content_path)
    s_pil = load_rgb(style_path)
    if color_prematch_strength > 0:
        c_pil = color_transfer_lab(c_pil, s_pil, strength=color_prematch_strength)
    init_pil = load_rgb(init_path) if (init_path and os.path.exists(init_path)) else None

    # Face mask
    face_mask_img = None
    auto_requested = (face_mask_path is None) or (str(face_mask_path).strip().lower() in ["", "auto"])
    if auto_requested:
        default_mask_path = os.path.splitext(content_path)[0] + "_face_mask.png"
        if os.path.exists(default_mask_path):
            face_mask_img = Image.open(default_mask_path)
            print(f"[mask] loaded existing mask: {default_mask_path}")
        else:
            face_mask_img = auto_face_mask_for_image(content_path, save_path=default_mask_path, verbose=True)
    else:
        if os.path.exists(face_mask_path):
            face_mask_img = Image.open(face_mask_path)
            print(f"[mask] loaded provided mask: {face_mask_path}")
        else:
            print(f"[mask] provided mask path not found: {face_mask_path} -> ignoring.")

    # --- run multi-stage with timing ---
    stage_times = []
    t0_total = time.time()
    x01 = None

    for i,(sz,nst) in enumerate(zip(sizes, steps), 1):
        c01 = pil_to_01_keep(c_pil, sz).to(device)
        s01 = pil_to_01_keep(s_pil, sz).to(device)
        if x01 is None:
            x01 = pil_to_01_keep(init_pil if init_pil else c_pil, sz).to(device)
        else:
            x01 = T.functional.resize(x01, [sz,sz], antialias=True)

        face_mask_01 = None
        if face_mask_img is not None:
            m = resize_mask_crop(face_mask_img, sz)
            m01 = T.ToTensor()(m).unsqueeze(0).to(device)
            if m01.shape[1] == 3:
                m01 = (0.299*m01[:,0:1]+0.587*m01[:,1:2]+0.114*m01[:,2:3])
            face_mask_01 = m01

        print(f"[Stage {i}] size={sz} steps={nst} opt={opt} "
              f"style_layers={style_spec} content_layer={content_layer} "
              f"sw={style_weight} cw={content_weight} tv={tv_weight} "
              f"face_preserve={face_preserve} backbone={backbone}")

        t0 = time.time()
        x01 = run_stage(x01, c01, s01, nst, opt, lr,
                        vgg, content_layer_id, style_layer_ids,
                        content_weight, style_weight, tv_weight, device,
                        face_mask_01=face_mask_01, w_face=face_preserve,
                        edge_w=edge_w, edge_face_down=edge_face_down)
        stage_times.append(time.time() - t0)

        outp = Path(out_path)
        stage_path = outp.with_stem(outp.stem + f"_{sz}")
        save_01(x01, str(stage_path))

    total_time = time.time() - t0_total
    save_01(x01, out_path)

    # Meta (includes resolved layers + timing)
    meta = {
        "content": content_path, "style": style_path, "out": out_path,
        "sizes": list(sizes), "steps": list(steps),
        "backbone": backbone,
        "content_layer": content_layer,
        "style_layers": style_layers,
        "content_layer_resolved": int(content_layer_id),
        "style_layers_resolved": [int(i) for i in style_layer_ids],
        "style_weight": float(style_weight), "content_weight": float(content_weight),
        "tv_weight": float(tv_weight), "opt": opt, "lr": float(lr),
        "device": str(device), "face_mask_path": face_mask_path,
        "face_preserve": float(face_preserve),
        "color_prematch_strength": float(color_prematch_strength),
        "seed": int(seed), "init_path": init_path,
        "edge_w": float(edge_w), "edge_face_down": float(edge_face_down),
        "runtime_sec": float(total_time),
        "stage_times_sec": [float(t) for t in stage_times]
    }
    try:
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            meta["peak_vram_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    except Exception:
        pass

    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--out", default="out/nst_hybrid_base.jpg")

    ap.add_argument("--opt", choices=["adam","lbfgs"], default="adam")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--style_weight", type=float, default=1.2e4)
    ap.add_argument("--content_weight", type=float, default=1.0,  # <<< NEW flag is live
                    help="Multiplier for content loss.")
    ap.add_argument("--tv_weight", type=float, default=2.5e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--init", default=None)

    ap.add_argument("--face_mask", default=None,
        help='Path to mask, or "auto" (default/empty) to detect & cache per content.')
    ap.add_argument("--face_preserve", type=float, default=0.65)
    ap.add_argument("--color_prematch_strength", type=float, default=0.5)

    ap.add_argument("--sizes", type=str, default="384,768",
                    help="Comma-separated sizes (e.g., 384,768).")
    ap.add_argument("--steps", type=str, default="400,500",
                    help="Comma-separated steps matching --sizes.")

    ap.add_argument("--backbone", choices=["vgg19","vgg16"], default="vgg19",
                    help="Feature extractor backbone.")
    ap.add_argument("--style_layers", default="conv4_1,conv5_1",
                    help="Comma-separated layers or aliases: layers45, layers345, layersA/B/C.")
    ap.add_argument("--content_layer", default="conv4_2",
                    help="Content layer (name preferred; numeric allowed for vgg19).")
    ap.add_argument("--edge_w", type=float, default=0.0,
                help="Strength of Laplacian edge loss (0 disables).")
    ap.add_argument("--edge_face_down", type=float, default=0.5,
                help="Multiply edge loss inside face mask (0..1).")

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    sizes = tuple(int(x) for x in str(args.sizes).split(","))
    steps = tuple(int(x) for x in str(args.steps).split(","))
    if len(sizes) != len(steps):
        raise SystemExit(f"--sizes and --steps must have same length, got {len(sizes)} vs {len(steps)}")

    run_nst(
        content_path=args.content,
        style_path=args.style,
        out_path=args.out,
        sizes=sizes, steps=steps,
        backbone=args.backbone,
        content_layer=args.content_layer,
        style_layers=args.style_layers,                # aliases OK
        style_weight=args.style_weight,
        content_weight=args.content_weight,            # <<< NOW PASSED THROUGH
        tv_weight=args.tv_weight,
        opt=args.opt, lr=args.lr, device=args.device,
        face_mask_path=args.face_mask, face_preserve=args.face_preserve,
        color_prematch_strength=args.color_prematch_strength,
        seed=args.seed, init_path=args.init,
        edge_w=args.edge_w, edge_face_down=args.edge_face_down
    )
