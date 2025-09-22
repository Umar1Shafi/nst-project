# scripts/nst_hybrid_ready.py
# --- Phase-1 Determinism Guard (tiny, safe) ---
import os, random
import numpy as np

# Read from env if set; default to 77 (your project seed)
SEED = int(os.environ.get("NST_SEED", "77"))

random.seed(SEED)
np.random.seed(SEED)

try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch >=1.12: opt into deterministic algorithms (warn_only avoids hard errors)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
except Exception:
    pass

# For some CUDA BLAS paths; harmless on CPU
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# --- end determinism guard ---

import os, argparse, json
import numpy as np
import cv2  # pip install opencv-contrib-python==4.12.0.88
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
import torchvision.transforms as T
from torchvision import models
import torch.nn.functional as F
from skimage import color  # pip install scikit-image

# -----------------------
# Determinism (nice-to-have)
# -----------------------
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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
    Accepts either:
      - names: 'conv4_1,conv5_1'
      - numbers: '19,28' (VGG19 only; kept for backward-compat)
    Returns: list[str] of names OR list[int] if numeric.
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
    # same crop/resize policy as images (keeps alignment)
    m = center_crop_square(pil_mask).convert("L")
    m = m.resize((size, size), Image.Resampling.LANCZOS)
    if soften:
        from PIL import ImageFilter
        m = m.filter(ImageFilter.GaussianBlur(radius=size/256))
    return m

def auto_face_mask_for_image(content_path, save_path=None, verbose=True):
    """
    1) Center-crop the content to a square.
    2) Detect the largest face with OpenCV Haar cascade.
    3) Draw a filled ellipse roughly covering the face, soften edges, save.
    Returns: PIL 'L' mask or None if no face found.
    """
    try:
        import cv2
    except Exception:
        print("[mask] OpenCV not available; skipping auto face mask.")
        return None
    pil = load_rgb(content_path)
    sq  = center_crop_square(pil)                 # align with pipeline
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

    # Largest face
    x, y, w, h = max(faces, key=lambda b: b[2]*b[3])

    mask = np.zeros((sq.height, sq.width), dtype=np.uint8)
    cx, cy = x + w//2, y + h//2
    ax, ay = int(w * 0.62), int(h * 0.82)  # ellipse axes scale
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, thickness=-1)

    k = max(3, (min(mask.shape)//128)|1)  # odd kernel size
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
def color_transfer_lab(src_pil, ref_pil, strength=0.5):
    """Match src mean/std to ref in LAB; blend by 'strength'."""
    src = np.asarray(src_pil) / 255.0
    ref = np.asarray(ref_pil) / 255.0
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
    """
    x: [B,C,H,W] in [0,1]. Returns Laplacian per channel via 4-neighbour kernel.
    """
    B, C, H, W = x.shape
    k = torch.tensor([[0, 1, 0],
                      [1,-4, 1],
                      [0, 1, 0]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    k = k.repeat(C, 1, 1, 1)                # one kernel per channel
    x_pad = F.pad(x, (1,1,1,1), mode="reflect")
    return F.conv2d(x_pad, k, groups=C)

def edge_loss_l1(x01, c01, face_mask_01=None, face_down=0.5):
    """
    L_edge = mean( |Lap(x) - Lap(c)| * weight )
    weight = 1 outside face; = face_down inside face (0..1).
    """
    Lx = laplacian3x3(x01)
    Lc = laplacian3x3(c01)
    diff = (Lx - Lc).abs()                   # [B,C,H,W]
    if face_mask_01 is not None:
        w = 1.0 - (1.0 - float(face_down)) * face_mask_01  # 1 outside, face_down inside
        diff = diff * w
    return diff.mean()

# -----------------------
# Losses
# -----------------------
def tv_loss(x):  # mean, not sum
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
    """
    Wrap a chosen VGG (16 or 19). We feed normalized input and
    collect features at integer layer indices.
    """
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
# One optimization stage (Adam default, TV schedule)
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
            # TV schedule: stronger early, gentler late
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
# Full pipeline (384 -> 768 by default)
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

    # Parse layer specs (allow names or numeric indices)
    def parse_spec_list(s):
        s = (s or "").replace(" ", "")
        return [x for x in s.split(",") if x]

    def to_index(item):
        # Numeric? Only valid for vgg19 (backward compat)
        if item.isdigit():
            if backbone != "vgg19":
                raise ValueError("Numeric layer indices are only supported for --backbone vgg19. "
                                 "Use names like conv4_1 for vgg16.")
            return int(item)
        # Named layer
        if item not in LMAP:
            raise ValueError(f"Unknown layer name '{item}' for backbone {backbone}.")
        return LMAP[item]

    style_items = parse_spec_list(style_layers)
    style_layer_ids = [to_index(x) for x in style_items]
    content_layer_id = to_index(content_layer)

    c_pil = load_rgb(content_path)
    s_pil = load_rgb(style_path)

    if color_prematch_strength > 0:
        c_pil = color_transfer_lab(c_pil, s_pil, strength=color_prematch_strength)

    init_pil = load_rgb(init_path) if (init_path and os.path.exists(init_path)) else None

    # --- Face mask handling (auto / reuse / provided) ---
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
              f"style_layers={style_layers} content_layer={content_layer} "
              f"sw={style_weight} tv={tv_weight} face_preserve={face_preserve} backbone={backbone}")

        x01 = run_stage(x01, c01, s01, nst, opt, lr,
                        vgg, content_layer_id, style_layer_ids,
                        content_weight, style_weight, tv_weight, device,
                        face_mask_01=face_mask_01, w_face=face_preserve,
                        edge_w=edge_w, edge_face_down=edge_face_down)

        save_01(x01, out_path.replace(".jpg", f"_{sz}.jpg"))

    save_01(x01, out_path)

    # Build meta first
    meta = {
        "content": content_path, "style": style_path, "out": out_path,
        "sizes": sizes, "steps": steps,
        "backbone": backbone,
        "content_layer": content_layer,
        "style_layers": style_layers,
        "content_layer_resolved": int(content_layer_id),
        "style_layers_resolved": [int(i) for i in style_layer_ids],
        "style_weight": style_weight, "content_weight": content_weight,
        "tv_weight": tv_weight, "opt": opt, "lr": lr,
        "device": str(device), "face_mask_path": face_mask_path,
        "face_preserve": face_preserve, "color_prematch_strength": color_prematch_strength,
        "seed": seed, "init_path": init_path,
        "edge_w": edge_w, "edge_face_down": edge_face_down
    }

    # Then add VRAM (optional) into meta
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
def parse_style_layers(s): return tuple(x.strip() for x in s.split(","))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--out", default="out/nst_hybrid_base.jpg")

    ap.add_argument("--opt", choices=["adam","lbfgs"], default="adam")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--style_weight", type=float, default=1.2e4)
    ap.add_argument("--tv_weight", type=float, default=2.5e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--init", default=None)

    ap.add_argument("--face_mask", default=None,
        help='Path to mask, or "auto" (default/empty) to detect & cache per content.')
    ap.add_argument("--face_preserve", type=float, default=0.65)
    ap.add_argument("--color_prematch_strength", type=float, default=0.5)
    ap.add_argument("--sizes", type=str, default="384,768")
    ap.add_argument("--steps", type=str, default="400,500")

    ap.add_argument("--backbone", choices=["vgg19","vgg16"], default="vgg19",
                    help="Feature extractor backbone.")
    ap.add_argument("--style_layers", default="conv4_1,conv5_1",
                    help="Comma-separated style layers. Prefer names (e.g., conv4_1,conv5_1). "
                         "Numeric indices are supported for vgg19 only.")
    ap.add_argument("--content_layer", default="conv4_2",
                    help="Content layer (name preferred; numeric allowed for vgg19).")
    ap.add_argument("--edge_w", type=float, default=0.0,
                help="Strength of Laplacian edge loss (0 disables).")
    ap.add_argument("--edge_face_down", type=float, default=0.5,
                help="Multiply edge loss by this factor inside face mask (0..1).")

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    sizes = tuple(int(x) for x in args.sizes.split(","))
    steps = tuple(int(x) for x in args.steps.split(","))

    run_nst(
        content_path=args.content,
        style_path=args.style,
        out_path=args.out,
        sizes=sizes, steps=steps,
        backbone=args.backbone,
        content_layer=args.content_layer,
        style_layers=args.style_layers,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        opt=args.opt, lr=args.lr, device=args.device,
        face_mask_path=args.face_mask, face_preserve=args.face_preserve,
        color_prematch_strength=args.color_prematch_strength,
        seed=args.seed, init_path=args.init,
        edge_w=args.edge_w, edge_face_down=args.edge_face_down
    )
