#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def load_sq(p):
    im = Image.open(p).convert("RGB")
    # If not square, center-crop to square
    w,h = im.size; s=min(w,h); L=(w-s)//2; T=(h-s)//2
    return im.crop((L,T,L+s,T+s))

def label(im, text):
    im = im.copy()
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    d.rectangle([0,0,im.width,28], fill=(0,0,0))
    d.text((8,5), text, fill=(255,255,255), font=font)
    return im

def make_triptych(content, gatys, hybrid, out, side=512):
    c = load_sq(content).resize((side,side))
    g = Image.open(gatys).convert("RGB").resize((side,side))
    h = Image.open(hybrid).convert("RGB").resize((side,side))
    c = label(c, "Original")
    g = label(g, "Gatys baseline")
    h = label(h, "Hybrid (edge+face+LAB)")
    w = side*3; trip = Image.new("RGB", (w, side))
    trip.paste(c,(0,0)); trip.paste(g,(side,0)); trip.paste(h,(2*side,0))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    trip.save(out)
    print("[write]", out)

if __name__ == "__main__":
    # Example usage:
    make_triptych(
        "data/content/portrait2.jpg",
        "figs/wall/portrait2__Matisse__gatys.png",
        "figs/wall/portrait2__Matisse__hybrid.png",
        "figs/wall/Fportrait2__Matisse__triptych.png"
    )
