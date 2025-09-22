import argparse, subprocess, sys, yaml, cv2, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "runners" / "run_stylizer.py"
REGISTRY = ROOT / "configs" / "phase3" / "preset_registry.yaml"

def detect_subject(img_path: Path) -> str:
    """
    Heuristic:
    1) If a sibling face-mask exists and covers >1.5% → portrait
    2) Else Haar face detection → if any face area ratio >0.8% or faces>=1 → portrait
    3) Else scene
    """
    # 1) Face-mask heuristic (e.g., portrait2_face_mask.png next to content)
    mask_candidate = img_path.with_name(img_path.stem + "_face_mask.png")
    if mask_candidate.exists():
        m = cv2.imread(str(mask_candidate), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            ratio = float(np.count_nonzero(m > 127)) / float(m.size)
            if ratio > 0.015:
                return "portrait"

    # 2) Haar face detection (fallback)
    img = cv2.imread(str(img_path))
    if img is None:
        return "scene"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = str(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          flags=cv2.CASCADE_SCALE_IMAGE, minSize=(40, 40))
    if len(faces) > 0:
        h, w = gray.shape[:2]
        # area ratio of largest face
        fx, fy, fw, fh = max(faces, key=lambda b: b[2]*b[3])
        ratio = (fw*fh) / float(w*h)
        if ratio > 0.008 or len(faces) >= 1:
            return "portrait"

    return "scene"

def choose_preset(style: str, subject: str, variant: str, registry: dict) -> str:
    """
    - For anime:
        * default = portrait→portrait_faithful, scene→scene_stylized
        * override with --variant faithful|stylized
    - For others:
        * portrait|scene keys
    """
    r = registry.get(style, {})
    if style == "anime":
        if variant in ("faithful", "stylized"):
            key = "portrait_faithful" if variant == "faithful" else "scene_stylized"
        else:
            key = "portrait_faithful" if subject == "portrait" else "scene_stylized"
    else:
        key = subject  # "portrait" or "scene"
    if key not in r:
        raise KeyError(f"No preset found for style={style}, key={key}")
    return r[key]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True, choices=["anime","cinematic","cyberpunk","noir"])
    ap.add_argument("-i","--input", required=True)
    ap.add_argument("-o","--output", required=True)
    ap.add_argument("--subject", default="auto", choices=["auto","portrait","scene"])
    ap.add_argument("--variant", default=None, choices=[None,"faithful","stylized"])
    ap.add_argument("--extra", default="", help="extra args appended at the end (optional)")
    args = ap.parse_args()

    reg = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))

    inp = Path(args.input)
    out = Path(args.output)
    if "phase3" not in out.parts:
        out = ROOT / "out" / "phase3" / args.style / out.name
    out.parent.mkdir(parents=True, exist_ok=True)

    # Subject routing
    subject = args.subject
    if subject == "auto":
        subject = detect_subject(inp)
        print(f"[router] auto-detected subject: {subject}")

    preset_args = choose_preset(args.style, subject, args.variant, reg)

    cmd = [sys.executable, str(RUNNER), "--stylizer", args.style, "-i", str(inp), "-o", str(out),
           "--stylizer-args", preset_args + ((" " + args.extra) if args.extra else "")]
    print(">>>", " ".join(cmd))
    ret = subprocess.call(cmd)
    sys.exit(ret)

if __name__ == "__main__":
    main()


