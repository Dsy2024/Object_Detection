from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random

def random_id():
    return f"{random.randint(100000, 999999)}"

def random_name():
    names = ["王小明", "陳志強", "林美玲", "張雅婷", "李俊傑"]
    return random.choice(names)

def draw_info_on_image(image_path, output_path):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # download font
    font = ImageFont.truetype("fonts/NotoSansCJKtc-VF.otf", 30)

    text = f"Name: {random_name()}\nID: {random_id()}"

    # random location
    x = random.randint(50, 200)
    y = random.randint(50, 100)

    draw.text((x, y), text, fill=(255, 0, 0), font=font)

    img.save(output_path)

if __name__ == "__main__":
    p = Path("images/")
    for img_file in p.glob("*.jpg"):
        output_file = f"temp/{img_file.name}"
        draw_info_on_image(img_file, output_file)