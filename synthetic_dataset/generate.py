"""Generate YOLO audiogram crops from 16 user-provided symbol sprites.
"""
from pathlib import Path
import argparse
import random
import math
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

ROOT = Path(__file__).resolve().parent
SPRITES = ROOT / 'symbols'
NAMES = ['<','<-os','>','>-os','O','O-os','X','X-os','[','[-os',']',']-os','□','□-os','△','△-os']
# Symbol center offsets relative to frequency grid line; brackets often sit beside the line.
OFFSETS = {0:-10, 1:-10, 2:10, 3:10, 8:-10, 9:-10, 10:10, 11:10}


def sprite_to_rgba(path):
    """Remove nearly white background while preserving antialiased red/blue strokes.

    A pixel's opacity is based on its color saturation and distance from white;
    this prevents a rectangular white patch from covering grid lines.
    """
    rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # Red: R much larger than G and B. Blue: B much larger than R and G.
    red = np.maximum(0, r - (g + b) / 2)
    blue = np.maximum(0, b - (r + g) / 2)
    strength = np.maximum(red, blue)
    alpha = np.clip((strength - 8) * (255 / 95), 0, 255).astype('uint8')
    # Convert to clean colored strokes; alpha retains anti-aliased edges.
    is_red = red >= blue
    out = np.zeros((*alpha.shape, 4), dtype='uint8')
    out[..., 0] = np.where(is_red, 210, 30)
    out[..., 1] = np.where(is_red, 35, 50)
    out[..., 2] = np.where(is_red, 55, 215)
    out[..., 3] = alpha
    im = Image.fromarray(out, 'RGBA')
    bbox = im.getchannel('A').getbbox()
    return im.crop(bbox) if bbox else im


def load_sprites():
    return {
        i: Image.open(SPRITES / f"{i}.png").convert("RGBA")
        for i in range(16)
    }


def dashed(draw, a, b, fill, width, dash, gap):
    x1,y1=a; x2,y2=b
    dist=math.hypot(x2-x1,y2-y1)
    if not dist: return
    ux,uy=(x2-x1)/dist,(y2-y1)/dist
    pos=0
    while pos<dist:
        end=min(pos+dash,dist)
        draw.line([(x1+ux*pos,y1+uy*pos),(x1+ux*end,y1+uy*end)],fill=fill,width=width)
        pos+=dash+gap


def draw_grid(w,h,rng):
    """Grid: solid vertical major lines, dashed intermediate lines, dashed horizontals."""
    im=Image.new('RGB',(w,h),'white')
    draw=ImageDraw.Draw(im)
    spacing=w/rng.uniform(7.2,8.5)
    first=rng.uniform(0.25,0.65)*spacing
    major=[]; x=first
    while x<w-15:
        major.append(round(x));x+=spacing
    gray=rng.randint(145,178)
    for x in major:
        draw.line((x,0,x,h),fill=(gray,gray,gray),width=rng.choice([2,3]))
    for a,b in zip(major,major[1:]):
        x=(a+b)//2
        dashed(draw,(x,0),(x,h),fill=(177,177,177),width=2,dash=7,gap=7)
    y_spacing=h/rng.uniform(12.5,15)
    horizontal=[];y=rng.uniform(0.5,1.0)*y_spacing
    while y<h-12:
        horizontal.append(round(y))
        dashed(draw,(0,y),(w,y),fill=(168,168,168),width=2,dash=8,gap=7)
        y+=y_spacing
    im=im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25,0.6)))
    return im,major,horizontal


def add_overlapping_pair(
    image, labels, boxes, sprites,
    x, y, rng, w, h, classes
):
    pair = []

    for cls in classes:
        sprite = sprites[cls]

        scale = rng.uniform(0.85, 1.18)
        sw = max(5, round(sprite.width * scale))
        sh = max(5, round(sprite.height * scale))

        glyph = sprite.resize(
            (sw, sh),
            Image.Resampling.LANCZOS,
        )

        # 輕微偏移，讓兩個符號部分重疊
        offset_x = rng.randint(-3, 3)
        offset_y = rng.randint(-3, 3)

        dx = OFFSETS.get(cls, 0) * scale

        left = round(x + dx + offset_x - sw / 2)
        top = round(y + offset_y - sh / 2)

        right = left + sw
        bottom = top + sh

        if (
            left < 1 or top < 1
            or right >= w - 1
            or bottom >= h - 1
        ):
            return False

        pair.append((cls, glyph, left, top, right, bottom))

    # 確認兩個符號都能放入圖片後才貼上
    for cls, glyph, left, top, right, bottom in pair:
        image.paste(glyph, (left, top), glyph)

        boxes.append((left, top, right, bottom))

        labels.append(
            f"{cls} "
            f"{(left + right) / (2 * w):.6f} "
            f"{(top + bottom) / (2 * h):.6f} "
            f"{(right - left) / w:.6f} "
            f"{(bottom - top) / h:.6f}"
        )

    return True


def generate_one(idx, out, sprites, rng, w, h, n_symbols, os_ratio):
    image, major, horizontals = draw_grid(w, h, rng)

    if len(major) < 3 or len(horizontals) < 3:
        return None

    labels = []
    boxes = []

    # 1. 決定 OS 數量
    n_os = round(n_symbols * os_ratio)

    os_flags = (
        [True] * n_os
        + [False] * (n_symbols - n_os)
    )

    rng.shuffle(os_flags)

    # 2. 建立格線位置
    positions = [
        (x, y)
        for x in major[1:-1]
        for y in horizontals[1:-1]
    ]

    rng.shuffle(positions)

    # 3. 產生一組重疊符號
    overlap_x, overlap_y = positions.pop()

    # 紅色與藍色的 base index
    red_bases = [0, 2, 4, 7]
    blue_bases = [1, 3, 5, 6]

    red_base = rng.choice(red_bases)
    blue_base = rng.choice(blue_bases)

    # 從原本 OS flags 取出兩個
    red_os = os_flags.pop()
    blue_os = os_flags.pop()

    red_cls = 2 * red_base + int(red_os)
    blue_cls = 2 * blue_base + int(blue_os)

    success = add_overlapping_pair(
        image=image,
        labels=labels,
        boxes=boxes,
        sprites=sprites,
        x=overlap_x,
        y=overlap_y,
        rng=rng,
        w=w,
        h=h,
        classes=[red_cls, blue_cls],
    )

    if not success:
        print(f"Warning: image {idx}: overlapping pair failed")
        return None

    # 4. 生成其餘符號
    for is_os in os_flags:

        base = rng.randrange(8)
        cls = 2 * base + int(is_os)

        sprite = sprites[cls]

        scale = rng.uniform(0.85, 1.18)

        sw = max(5, round(sprite.width * scale))
        sh = max(5, round(sprite.height * scale))

        glyph = sprite.resize(
            (sw, sh),
            Image.Resampling.LANCZOS,
        )

        placed = False

        for x, y in positions:

            dx = OFFSETS.get(cls, 0) * scale

            left = round(x + dx - sw / 2)
            top = round(y - sh / 2)

            right = left + sw
            bottom = top + sh

            if (
                left < 1 or top < 1
                or right >= w - 1
                or bottom >= h - 1
            ):
                continue

            # 避免和其他符號重疊
            if any(
                not (
                    right + 8 < a
                    or left - 8 > c
                    or bottom + 8 < b
                    or top - 8 > d
                )
                for a, b, c, d in boxes
            ):
                continue

            image.paste(glyph, (left, top), glyph)

            boxes.append(
                (left, top, right, bottom)
            )

            labels.append(
                f"{cls} "
                f"{(left + right) / (2 * w):.6f} "
                f"{(top + bottom) / (2 * h):.6f} "
                f"{(right - left) / w:.6f} "
                f"{(bottom - top) / h:.6f}"
            )

            positions.remove((x, y))
            placed = True
            break

        if not placed:
            print(
                f"Warning: image {idx}: "
                "insufficient nonoverlapping grid positions"
            )

    # 5. 儲存圖片與 Label
    stem = f"synth_{idx:05d}"

    image.save(
        out / "images" / f"{stem}.png"
    )

    (
        out / "labels" / f"{stem}.txt"
    ).write_text(
        "\n".join(labels) + "\n",
        encoding="utf-8",
    )

    return image, boxes


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--count',type=int,default=120)
    p.add_argument('--out',type=Path,default=ROOT)
    p.add_argument('--width',type=int,default=590)
    p.add_argument('--height',type=int,default=630)
    p.add_argument("--min-symbols", type=int, default=8)
    p.add_argument("--max-symbols", type=int, default=16)
    p.add_argument('--os-ratio',type=float,default=0.9)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if not 0<=args.os_ratio<=1: p.error('--os-ratio must be between 0 and 1')
    if args.min_symbols < 2 or args.max_symbols < 2:
        p.error('--min-symbols and --max-symbols must be at least 2')
    if args.min_symbols > args.max_symbols:
        p.error('--min-symbols must not exceed --max-symbols')
    rng=random.Random(args.seed)
    for sub in ['images','labels']:(args.out/sub).mkdir(parents=True,exist_ok=True)
    sprites=load_sprites()
    for idx in range(args.count):
        n_symbols = rng.randint(
            args.min_symbols,
            args.max_symbols,
        )
        result=generate_one(idx,args.out,sprites,rng,args.width,args.height,n_symbols,args.os_ratio)
        if idx==0 and result:
            im,boxes=result
            preview=im.copy();d=ImageDraw.Draw(preview)
            for box in boxes:d.rectangle(box,outline=(0,160,0),width=2)
            preview.save(ROOT/'preview_labeled.png')
    print(f'Generated {args.count} images at {args.out}')

if __name__=='__main__':main()
