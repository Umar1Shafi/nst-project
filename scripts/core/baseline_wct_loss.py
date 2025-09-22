# scripts/baseline_wct_loss.py
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
import torch, torch.nn as nn, torch.optim as optim
from torchvision import models, transforms as T
from PIL import Image

IMNORM = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

VGG19_MAP = {
    "conv1_1":0,"relu1_1":1,"conv1_2":2,"relu1_2":3,"pool1":4,
    "conv2_1":5,"relu2_1":6,"conv2_2":7,"relu2_2":8,"pool2":9,
    "conv3_1":10,"relu3_1":11,"conv3_2":12,"relu3_2":13,"conv3_3":14,"relu3_3":15,"conv3_4":16,"relu3_4":17,"pool3":18,
    "conv4_1":19,"relu4_1":20,"conv4_2":21,"relu4_2":22,"conv4_3":23,"relu4_3":24,"conv4_4":25,"relu4_4":26,"pool4":27,
    "conv5_1":28,"relu5_1":29,"conv5_2":30,"relu5_2":31,"conv5_3":32,"relu5_3":33,"conv5_4":34,"relu5_4":35,"pool5":36
}
VGG16_MAP = {
    "conv1_1":0,"relu1_1":1,"conv1_2":2,"relu1_2":3,"pool1":4,
    "conv2_1":5,"relu2_1":6,"conv2_2":7,"relu2_2":8,"pool2":9,
    "conv3_1":10,"relu3_1":11,"conv3_2":12,"relu3_2":13,"conv3_3":14,"relu3_3":15,"pool3":16,
    "conv4_1":17,"relu4_1":18,"conv4_2":19,"relu4_2":20,"conv4_3":21,"relu4_3":22,"pool4":23,
    "conv5_1":24,"relu5_1":25,"conv5_2":26,"relu5_2":27,"conv5_3":28,"relu5_3":29,"pool5":30
}

def build_backbone(name):
    if name=="vgg19":
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        lmap = VGG19_MAP
    else:
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        lmap = VGG16_MAP
    for p in vgg.parameters(): p.requires_grad_(False)
    return vgg, lmap

def load_rgb(p): return Image.open(p).convert("RGB")

def center_crop_resize(pil, size):
    w,h = pil.size
    s = min(w,h); left=(w-s)//2; top=(h-s)//2
    pil = pil.crop((left,top,left+s,top+s))
    return pil.resize((size,size), Image.Resampling.LANCZOS)

def to01(pil, size=None):
    if size: pil = center_crop_resize(pil, size)
    return T.ToTensor()(pil).unsqueeze(0)

def save01(t, path): T.ToPILImage()(t.clamp(0,1).cpu().squeeze(0)).save(path)

def tv_loss(x):
    dx = (x[:,:,1:,:]-x[:,:,:-1,:]).abs().mean()
    dy = (x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean()
    return dx+dy

class VGGFeat(nn.Module):
    def __init__(self, vgg, layers_collect):
        super().__init__()
        self.vgg=vgg; self.collect=set(layers_collect)
    def forward(self, x01):
        x = IMNORM(x01.squeeze(0)).unsqueeze(0)
        feats={}
        for i,layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.collect: feats[i]=x
        return feats

def parse_list(s): return [x for x in (s or "").replace(" ","").split(",") if x]

def resolve(specs, lmap, backbone):
    out=[]
    for it in specs:
        if it.isdigit():
            if backbone!="vgg19":
                raise ValueError("Numeric indices supported only for vgg19. Use names like conv4_1.")
            out.append(int(it))
        else:
            if it not in lmap: raise ValueError(f"Unknown layer '{it}' for {backbone}")
            out.append(lmap[it])
    return out

def wct_transform(c_feat, s_feat, eps=1e-5):
    """
    WCT on features. Accepts 1xCxHxW or CxHxW; returns 1xCxHxW.
    """
    # Ensure we work on channel-first, no batch
    if c_feat.dim() == 4:  # 1xCxHxW
        c_feat = c_feat.squeeze(0)
    if s_feat.dim() == 4:
        s_feat = s_feat.squeeze(0)

    C, H, W = c_feat.shape
    c = c_feat.view(C, -1)
    s = s_feat.view(C, -1)

    # zero-mean
    c_mean = c.mean(dim=1, keepdim=True)
    s_mean = s.mean(dim=1, keepdim=True)
    c = c - c_mean
    s = s - s_mean

    # covariances
    eps = float(eps)
    c_cov = (c @ c.t()) / (c.shape[1] - 1 + eps)
    s_cov = (s @ s.t()) / (s.shape[1] - 1 + eps)

    # eigendecomp (symmetric)
    Dc, Uc = torch.linalg.eigh(c_cov)   # ascending eigenvalues
    Ds, Us = torch.linalg.eigh(s_cov)

    # clamp tiny/negative eigenvalues
    Dc = torch.clamp(Dc, min=eps)
    Ds = torch.clamp(Ds, min=eps)

    # whiten content, then color with style
    Wh = Uc @ torch.diag(Dc.pow(-0.5)) @ Uc.t()
    cw = Wh @ c
    Co = Us @ torch.diag(Ds.pow(0.5)) @ Us.t()
    cs = Co @ cw + s_mean  # add style mean back

    return cs.view(1, C, H, W)

def run_stage(x01, c01, s01, steps, lr,
              vgg, layer_id, w_c, w_s, w_tv, device):
    net = VGGFeat(vgg, [layer_id]).to(device).eval()
    with torch.no_grad():
        c_feat = net(c01)[layer_id]
        s_feat = net(s01)[layer_id]
        target = wct_transform(c_feat.to(device), s_feat.to(device))  # returns 1xCxHxW

    x01 = nn.Parameter(x01)
    opt = optim.Adam([x01], lr=lr)
    mse = nn.MSELoss()

    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        x_cl = x01.clamp(0,1)
        f = net(x_cl)[layer_id]
        c_loss = mse(f, c_feat) * w_c
        s_loss = mse(f, target) * w_s
        k = 1.0 - (t+1)/steps
        tv = tv_loss(x_cl) * (0.5*k + 0.5) * w_tv
        loss = c_loss + s_loss + tv
        loss.backward(); opt.step()
    return x01.clamp(0,1).detach()

def run_wct_loss(content, style, out,
                 backbone="vgg19",
                 layer="conv4_1",    # WCT usually done at one or a few layers; start here
                 sizes=(384,768), steps=(300,400),
                 lr=0.01, w_c=1.0, w_s=1.0, w_tv=2.5e-3,
                 seed=123, device="cuda"):
    torch.manual_seed(seed)
    device = torch.device(device if (device=="cpu" or torch.cuda.is_available()) else "cpu")
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    vgg, lmap = build_backbone(backbone); vgg=vgg.to(device)

    layer_id = resolve([layer], lmap, backbone)[0]
    c_pil = load_rgb(content); s_pil = load_rgb(style)

    x01=None
    for i,(sz,nst) in enumerate(zip(sizes,steps),1):
        c01 = to01(c_pil, sz).to(device)
        s01 = to01(s_pil, sz).to(device)
        if x01 is None:
            x01 = to01(c_pil, sz).to(device)
        else:
            x01 = T.functional.resize(x01, [sz,sz], antialias=True)

        print(f"[WCT] Stage {i} size={sz} steps={nst} backbone={backbone} layer={layer}")
        x01 = run_stage(x01, c01, s01, nst, lr, vgg, layer_id, w_c, w_s, w_tv, device)
        save01(x01, out.replace(".jpg", f"_{sz}.jpg"))
    save01(x01, out)

    # Build meta first
    meta = {"method":"wct_loss","backbone":backbone,"layer":layer,
            "sizes":sizes,"steps":steps,"lr":lr,"w_c":w_c,"w_s":w_s,"w_tv":w_tv,
            "content":content,"style":style,"out":out}

    # Then add VRAM
    try:
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            meta["peak_vram_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    except Exception:
        pass

    with open(out.replace(".jpg",".json"),"w") as f:
        json.dump(meta,f,indent=2)


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--out", default="out/wct_loss/out.jpg")
    ap.add_argument("--backbone", choices=["vgg19","vgg16"], default="vgg19")
    ap.add_argument("--layer", default="conv4_1")
    ap.add_argument("--sizes", default="384,768")
    ap.add_argument("--steps", default="300,400")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--w_c", type=float, default=1.0)
    ap.add_argument("--w_s", type=float, default=1.0)
    ap.add_argument("--w_tv", type=float, default=2.5e-3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sizes = tuple(int(x) for x in args.sizes.split(","))
    steps = tuple(int(x) for x in args.steps.split(","))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    run_wct_loss(args.content,args.style,args.out,
                 backbone=args.backbone, layer=args.layer,
                 sizes=sizes, steps=steps,
                 lr=args.lr, w_c=args.w_c, w_s=args.w_s, w_tv=args.w_tv,
                 seed=args.seed, device=args.device)
    
    
