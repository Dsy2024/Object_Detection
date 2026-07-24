import os
import re

import opencc
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

from .db import upsert_record


def add_fake_data(filename, image_path, output_path):
    fake = Faker("zh_TW")
    os.makedirs(output_path, exist_ok=True)

    img_path = os.path.join(image_path, filename)
    doctor_name = fake.name()
    patient_name = fake.name()
    letters = "".join(fake.random_letters(length=5))
    numbers = fake.random_number(digits=6, fix_len=True)
    serial_number = f"SN-{letters}-{numbers}"

    print(filename, doctor_name, patient_name, serial_number)

    img = Image.open(img_path).convert("RGB")
    x, y = img.width // 3, img.height // 6
    draw = ImageDraw.Draw(img)

    # font = ImageFont.truetype("msjh.ttc", 48)  # Microsoft JhengHei (Windows)
    # font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 36)
    font = ImageFont.truetype("fonts/NotoSansCJKtc-VF.otf", 30)


    draw.text(
        (x, y),
        f"醫生: {doctor_name}   姓名: {patient_name}",
        fill=(75, 75, 75),
        font=font,
    )
    draw.text((x, y + 50), f"序號: {serial_number}", fill=(75, 75, 75), font=font)

    output_file = os.path.join(output_path, f"f{filename}")
    img.save(output_file)

    data = {
        "path": output_file,
        "doctor": doctor_name,
        "patient": patient_name,
        "serial": serial_number,
    }

    return output_file, data


def extract_data(result, file_path):
    if isinstance(result, list):
        text = "\n".join(map(str, result))
    else:
        text = str(result)

    text = text.replace(" ", "").replace("\n", "")
    print(f"  → OCR Result:\n {text}")
    # clean_text = re.sub(r"\s+", "", full_text)
    # clean_text = re.sub(r"\*", "", clean_text)
    # clean_text = re.sub(r"\#", "", clean_text)
    # print(f"  → Cleaned Text: {clean_text}")

    # converter = opencc.OpenCC("s2t.json")
    # text = converter.convert(clean_text)
    # print(f"  → Converted Text: {text}")

    doctor_match = re.search(r"醫生[:：]([\u4e00-\u9fff]+)", text)
    patient_match = re.search(r"姓名[:：]([\u4e00-\u9fff]+)", text)
    serial_match = re.search(r"(SN-[A-Za-z]{5}-\d{6})", text)

    doctor_name = doctor_match.group(1) if doctor_match else "N/A"
    patient_name = patient_match.group(1) if patient_match else "N/A"
    serial_number = serial_match.group(1) if serial_match else "N/A"
    for word in ["姓名", "醫生", "序號"]:
        if word in doctor_name:
            doctor_name = doctor_name.replace(word, "")
        if word in patient_name:
            patient_name = patient_name.replace(word, "")

    print(f"  → 醫生: {doctor_name}, 姓名: {patient_name}, 序號: {serial_number}")

    data = {
        "path": file_path,
        "doctor": doctor_name,
        "patient": patient_name,
        "serial": serial_number,
    }

    return data


def save_extracted_record(data, file_path):
    if data["doctor"] == "N/A":
        print(f"  → [DB] Skipped (doctor not found): {file_path}")
    elif data["patient"] == "N/A":
        print(f"  → [DB] Skipped (patient not found): {file_path}")
    elif data["serial"] == "N/A":
        print(f"  → [DB] Skipped (serial not found): {file_path}")
    else:
        upsert_record(
            serial_number=data["serial"],
            patient_name=data["patient"],
            doctor_name=data["doctor"],
            audiogram=file_path,
        )
        print(f"  → [DB] Saved record for serial {data['serial']}")


def validation(org, new):
    print("\n=== Validation Results ===")

    total_name = 0      # doctor + patient
    correct_name = 0

    total_serial = 0
    correct_serial = 0

    org_map = {data["path"]: data for data in org}

    for data in new:
        path = data["path"]

        if path not in org_map:
            print(f"⚠️ No original data for {path}")
            continue

        original_data = org_map[path]

        doctor_ok = data["doctor"] == original_data["doctor"]
        patient_ok = data["patient"] == original_data["patient"]
        serial_ok = data["serial"] == original_data["serial"]

        total_name += 2
        total_serial += 1

        if doctor_ok:
            correct_name += 1
        if patient_ok:
            correct_name += 1

        if serial_ok:
            correct_serial += 1

        if doctor_ok and patient_ok and serial_ok:
            print(f"✅ Data matches for {path}")
        else:
            print(f"❌ {path}")
            if not doctor_ok:
                print(f"  Doctor: {data['doctor']} != {original_data['doctor']}")
            if not patient_ok:
                print(f"  Patient: {data['patient']} != {original_data['patient']}")
            if not serial_ok:
                print(f"  Serial: {data['serial']} != {original_data['serial']}")

    name_acc = correct_name / total_name if total_name else 0
    serial_acc = correct_serial / total_serial if total_serial else 0

    print("\n=== Accuracy ===")
    print(f"Doctor + Patient Accuracy: {correct_name}/{total_name} = {name_acc:.2%}")
    print(f"Serial Accuracy: {correct_serial}/{total_serial} = {serial_acc:.2%}")


def crop_top(img, save_crop=False):
    crop = img.crop((
        img.width // 4,
        img.height // 6 - 20,
        img.width * 3 // 4,
        img.height // 6 + 120
    ))

    # Save the cropped area for debugging
    if save_crop:
        crop.save("outputs/debug/crop_top.png")

    return crop