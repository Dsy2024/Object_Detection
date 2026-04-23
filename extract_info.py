import cv2
import easyocr
import re
import numpy as np
from pathlib import Path
from database import insert_detected_info

reader = easyocr.Reader(['en', 'ch_tra'], gpu=True)


def crop_top_left(img, width_ratio=0.35, height_ratio=0.2):
    h, w = img.shape[:2]
    x1, y1 = 0, 0
    x2 = int(w * width_ratio)
    y2 = int(h * height_ratio)
    return img[y1:y2, x1:x2]


def extract_name_and_id(raw_text):
    """
    Default format:
    Name: 王小明
    ID: 12345678
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    values = []
    for line in lines:
        if ":" in line:
            values.append(line.split(":", 1)[1].strip())
        elif "：" in line:
            values.append(line.split("：", 1)[1].strip())

    name = values[0] if len(values) > 0 else None
    mrn = values[1] if len(values) > 1 else None

    return name, mrn


def ocr_detect_top_left(image_path, save_crop=False):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    roi = crop_top_left(img)
    roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # preprocess for better OCR results
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]

    # morphological closing to connect text parts
    # kernel = np.ones((2, 2), np.uint8)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Save the cropped area for debugging
    if save_crop:
        cv2.imwrite("outputs/debug_top_left_crop.png", binary)

    results = reader.readtext(binary, detail=0)
    raw_text = "\n".join(results)

    name, mrn = extract_name_and_id(raw_text)
    return name, mrn, raw_text

def process_patient_info(image_path, db_path="patient_info.db"):
    image_path = Path(image_path)
    name, mrn, raw_text = ocr_detect_top_left(image_path, save_crop=True)

    insert_detected_info(
        db_path=db_path,
        image_filename=image_path.name,
        patient_name=name,
        mrn=mrn
    )

    print("Detected:")
    print("  Name:", name)
    print("  MRN :", mrn)
    print("  Raw OCR:")
    print(raw_text)

if __name__ == "__main__":
    process_patient_info("temp/1.jpg")