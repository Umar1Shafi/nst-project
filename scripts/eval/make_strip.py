import argparse
from pathlib import Path
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="List of images in order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", type=int, default=6)
    args = ap.parse_args()

    imgs = [Image.open(p).convert("RGB") for p in args.images]
    h = min(im.height for im in imgs)
    imgs = [im.resize((int(im.width*h/im.height), h), Image.LANCZOS) for im in imgs]
    w = sum(im.width for im in imgs) + args.pad * (len(imgs)-1)

    strip = Image.new("RGB", (w, h), (18,18,18))
    x = 0
    for i,im in enumerate(imgs):
        strip.paste(im, (x,0))
        x += im.width + args.pad
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    strip.save(args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
