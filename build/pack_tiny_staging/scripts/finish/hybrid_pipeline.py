# scripts/hybrid_pipeline.py
# Hybrid stylisation: palette quantisation + XDoG edges + face-aware blending + paper grain
import argparse, os, random
import numpy as np, cv2
from PIL import Image
from skimage import color

# ---------- determinism helpers ----------
def set_determinism(seed: int):
    """Lock all RNGs and avoid non-deterministic OpenCV codepaths."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        cv2.setRNGSeed(seed)
    except Exception:
        pass
    # Avoid OpenCL/GPU paths and thread scheduling nondeterminism
    try:
        cv2.ocl.setUseOpenCL(False)
        os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
    except Exception:
        pass
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

# ---------- basic I/O ----------
def load_rgb(p): return Image.open(p).convert("RGB")
def pil2np(pil): return np.array(pil).astype(np.float32)/255.0
def np2pil(arr): return Image.fromarray(np.clip(arr*255,0,255).astype(np.uint8))

# ---------- guided smoothing (optional, makes flat fills) ----------
def guided_flatten(img_rgb, radius=5, eps=1e-3):
    if not hasattr(cv2, "ximgproc"):
        return img_rgb  # fallback if contrib not present
    guide = img_rgb
    src   = img_rgb
    out = np.zeros_like(img_rgb)
    for c in range(3):
        out[...,c] = cv2.ximgproc.guidedFilter(guide=guide[...,c], src=src[...,c], radius=radius, eps=eps)
    return out

# ---------- palettes ----------
def kmeans_palette_from_style(style_pil, k=12, iters=20, seed=123, sample=120_000):
    """
    Deterministic K-means palette in LAB space.
    - Seeds OpenCV RNG.
    - Deterministically subsamples to 'sample' pixels for speed & stability.
    """
    try:
        cv2.setRNGSeed(seed)
    except Exception:
        pass

    sty = pil2np(style_pil)
    lab = color.rgb2lab(sty).reshape(-1,3).astype(np.float32)

    # reduce very-bright paper pixels so blues dominate more
    L = lab[:,0]
    keep = L < 95
    if keep.sum() > k*10:
        lab = lab[keep]

    # deterministic downsample (if too many pixels)
    if lab.shape[0] > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(lab.shape[0], size=sample, replace=False)
        lab = lab[idx]

    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, iters, 1e-4)
    _ret, _labels, centers = cv2.kmeans(
        lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    return centers  # LAB (k,3)

def preset_ukiyoe_palette():
    hexes = ["#142b4a","#1f4f7a","#6fa0c9","#f0e6c8","#d6c19d","#c95a3a","#6e5a44","#202020"]
    pal = []
    for hx in hexes:
        rgb = np.array([int(hx[i:i+2],16) for i in (1,3,5)], dtype=np.float32)[None,None,:]/255.0
        pal.append(color.rgb2lab(rgb).reshape(3,).astype(np.float32))
    return np.stack(pal, axis=0)

def quantize_to_palette(img_pil, palette_lab):
    img = pil2np(img_pil); H,W,_ = img.shape
    lab = color.rgb2lab(img).reshape(-1,3)
    d2 = ((lab[:,None,:]-palette_lab[None,:,:])**2).sum(axis=2)  # [N,k]
    idx = d2.argmin(axis=1)
    q_lab = palette_lab[idx].reshape(H,W,3)
    q_rgb = color.lab2rgb(q_lab)
    return np2pil(np.clip(q_rgb,0,1))

# ---------- XDoG edges ----------
def xdog_edges(pil, sigma=0.6, k=1.6, gamma=0.98, epsilon=-0.1, phi=10.0, thresh=0.0):
    gray = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    g1 = cv2.GaussianBlur(gray, (0,0), sigma)
    g2 = cv2.GaussianBlur(gray, (0,0), sigma*k)
    D  = g1 - gamma*g2
    X  = 1.0 + np.tanh( phi*(D - epsilon) )
    X  = X/2.0
    if thresh>0: X = (X>thresh).astype(np.float32)
    ink = 1.0 - X
    return np.clip(ink,0,1)

def thicken(ink01, px=1):
    if px<=0: return ink01
    k = np.ones((px*2+1, px*2+1), np.uint8)
    return cv2.dilate(ink01, k, iterations=1)

# ---------- paper grain ----------
def add_paper_grain(pil, strength=0.06, rng=None):
    """
    Add subtle paper tint + grain using a *local* RNG to keep results reproducible.
    """
    if rng is None:
        rng = np.random.default_rng(123)
    base = pil2np(pil); h,w,_ = base.shape
    noise = rng.normal(0.0, 1.0, (h,w,1)).astype(np.float32)
    tint  = np.array([0.98,0.96,0.92], dtype=np.float32)
    out   = base*(1-strength) + strength*(tint + 0.06*noise)
    return np2pil(np.clip(out,0,1))

# ---------- core ----------
def run_hybrid(inp_nst, style_path, out_path,
               colors=12, use_preset_palette=False,
               xdog_sigma=0.6, xdog_phi=11.0, xdog_epsilon=-0.1, xdog_k=1.6, xdog_gamma=0.98,
               edge_gain=0.38, edge_thickness=1,
               face_mask=None, edge_gain_face=0.22, mask_feather=9,
               use_guided=True, guided_radius=6, guided_eps=1e-3,
               paper=True, paper_strength=0.06,
               face_colors=16, bg_colors=10,
               seed=123,
               verbose=False):

    # lock randomness
    set_determinism(seed)
    rng = np.random.default_rng(seed)

    nst = load_rgb(inp_nst); sty = load_rgb(style_path)

    # palettes
    if use_preset_palette:
        pal_bg = preset_ukiyoe_palette()
        pal_face = pal_bg
        if verbose: print(f"[palette] preset ukiyo-e: {pal_bg.shape[0]} colors")
    else:
        pal_bg   = kmeans_palette_from_style(sty, k=bg_colors, seed=seed)
        pal_face = kmeans_palette_from_style(sty, k=face_colors, seed=seed)
        if verbose: print(f"[palette] k-means: bg={bg_colors} face={face_colors} (seed={seed})")

    # quantise whole image to bg palette
    flat_bg = quantize_to_palette(nst, pal_bg)
    if use_guided:
        flat_bg = np2pil(guided_flatten(pil2np(flat_bg), radius=guided_radius, eps=guided_eps))
        if verbose: print(f"[guided] radius={guided_radius} eps={guided_eps}")

    base_np = pil2np(flat_bg)

    # optional face re-quantise to richer palette
    if face_mask and os.path.exists(face_mask):
        mask_pil = load_rgb(face_mask).convert("L")
        # resize mask to match NST image size
        W, H = nst.size
        mask_pil = mask_pil.resize((W, H), Image.Resampling.LANCZOS)

        m = (np.array(mask_pil).astype(np.float32)/255.0)
        if mask_feather > 0:
            m = cv2.GaussianBlur(m, (0, 0), mask_feather)
        m3 = m[..., None]  # (H,W,1)

        flat_face = quantize_to_palette(nst, pal_face)
        if use_guided:
            flat_face = np2pil(guided_flatten(pil2np(flat_face), radius=guided_radius, eps=guided_eps))
        face_np = pil2np(flat_face)

        # blend: more detailed face, flat background
        base_np = face_np * m3 + base_np * (1.0 - m3)
        if verbose: print("[face] blended richer face palette")

    # edges (kept on NST image for parity; you can switch to flat_bg if preferred)
    ink = xdog_edges(nst, sigma=xdog_sigma, k=xdog_k, gamma=xdog_gamma, epsilon=xdog_epsilon, phi=xdog_phi)
    ink = thicken(ink, edge_thickness)
    edges_rgb = np.stack([ink,ink,ink], axis=2)

    # face-aware edge gain
    if face_mask and os.path.exists(face_mask):
        mask_pil = load_rgb(face_mask).convert("L")
        W, H = nst.size
        mask_pil = mask_pil.resize((W, H), Image.Resampling.LANCZOS)

        m = (np.array(mask_pil).astype(np.float32)/255.0)
        if mask_feather > 0:
            m = cv2.GaussianBlur(m, (0, 0), mask_feather)
        m3 = m[..., None]
        gain_map = edge_gain * (1.0 - m3) + edge_gain_face * m3
    else:
        gain_map = edge_gain

    out = base_np * (1.0 - gain_map*edges_rgb)
    out_pil = np2pil(out)
    if paper:
        out_pil = add_paper_grain(out_pil, strength=paper_strength, rng=rng)
        if verbose: print(f"[paper] strength={paper_strength} (seed={seed})")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_pil.save(out_path)
    print("Saved:", out_path)

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", required=True, help="NST image (jpg/png)")
    p.add_argument("--style", required=True, help="style image (for palette)")
    p.add_argument("--out", required=True)
    p.add_argument("--colors", type=int, default=12, help="legacy shared colors (ignored if face/bg provided)")
    p.add_argument("--bg_colors", type=int, default=10)
    p.add_argument("--face_colors", type=int, default=16)
    p.add_argument("--use_preset_palette", action="store_true")

    p.add_argument("--xdog_sigma", type=float, default=0.6)
    p.add_argument("--xdog_phi", type=float, default=11.0)
    p.add_argument("--xdog_epsilon", type=float, default=-0.10)
    p.add_argument("--xdog_k", type=float, default=1.6)
    p.add_argument("--xdog_gamma", type=float, default=0.98)

    p.add_argument("--edge_gain", type=float, default=0.38)
    p.add_argument("--edge_gain_face", type=float, default=0.22)
    p.add_argument("--edge_thickness", type=int, default=1)

    p.add_argument("--face_mask", default=None)
    p.add_argument("--mask_feather", type=int, default=9)

    p.add_argument("--use_guided", action="store_true")
    p.add_argument("--guided_radius", type=int, default=6)
    p.add_argument("--guided_eps", type=float, default=1e-3)

    p.add_argument("--paper", action="store_true")
    p.add_argument("--paper_strength", type=float, default=0.06)

    p.add_argument("--seed", type=int, default=123)  # <-- new
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # keep compatibility with --colors if user sets only that
    bg_cols   = args.bg_colors if args.bg_colors else args.colors
    face_cols = args.face_colors if args.face_colors else max(args.colors, 12)
    run_hybrid(
        inp_nst=args.inp, style_path=args.style, out_path=args.out,
        colors=args.colors, use_preset_palette=args.use_preset_palette,
        xdog_sigma=args.xdog_sigma, xdog_phi=args.xdog_phi, xdog_epsilon=args.xdog_epsilon,
        xdog_k=args.xdog_k, xdog_gamma=args.xdog_gamma,
        edge_gain=args.edge_gain, edge_thickness=args.edge_thickness,
        face_mask=args.face_mask, edge_gain_face=args.edge_gain_face, mask_feather=args.mask_feather,
        use_guided=args.use_guided, guided_radius=args.guided_radius, guided_eps=args.guided_eps,
        paper=args.paper, paper_strength=args.paper_strength,
        face_colors=face_cols, bg_colors=bg_cols,
        seed=args.seed,
        verbose=args.verbose
    )
