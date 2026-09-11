import os
import re
import hashlib
import random
from pathlib import Path

import opencc
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

from .db import upsert_record


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}


def safe_filename(path, default="audiogram.png"):
    name = re.split(r"[\\/]+", str(path))[-1]
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    if name in {"", ".", ".."}:
        return default
    return name


def resolve_input_path(file_path, allowed_exts=None):
    path_text = str(file_path).strip().strip('"')
    candidates = [line.strip().strip('"') for line in path_text.splitlines() if line.strip()]
    candidates.append(path_text)

    allowed_exts = allowed_exts or (IMAGE_EXTS | PDF_EXTS)
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file() and path.suffix.lower() in allowed_exts:
            return path
        if path.is_dir():
            for child in path.iterdir():
                if child.is_file() and child.suffix.lower() in allowed_exts:
                    return child

    return Path(candidates[-1])


def pdf_first_page_to_image(pdf_path, output_dir="outputs/pdf_pages", scale=2):
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ImportError(
            "PDF uploads require `pypdfium2`. Install it with `uv sync` after "
            "updating dependencies."
        ) from exc

    pdf_path = Path(pdf_path) # Ensure pdf_path is a Path object
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha1(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:10]
    image_path = output_dir / f"{pdf_path.stem}_{digest}_page1.png"

    pdf = pdfium.PdfDocument(str(pdf_path))
    page = None
    try:
        if len(pdf) == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")

        page = pdf[0]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil().convert("RGB")
        image.save(image_path)
    finally:
        if page is not None:
            page.close()
        pdf.close()

    return image_path


def prepare_input_image(file_path):
    path = resolve_input_path(file_path)
    if path.suffix.lower() in PDF_EXTS:
        return pdf_first_page_to_image(path)
    return path


def pil_to_ocr_image(img):
    import numpy as np

    return np.array(img.convert("RGB"))


def save_crop_image(img, output_path="outputs/debug/crop_top.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    return output_path


def add_fake_data(filename, image_path, output_path):
    fake = Faker("zh_TW")
    os.makedirs(output_path, exist_ok=True)

    def format_date(date):
        return f"{date.year}/{date.month}/{date.day}"

    img_path = os.path.join(image_path, filename)
    doctor_name = fake.name()
    patient_name = fake.name()
    patient_number = fake.random_number(digits=7, fix_len=True)
    serial_number = f"{fake.random_int(min=10, max=99)}-{fake.random_int(min=1000, max=9999)}"
    gender = random.choice(["Male", "Female"])
    birthdate = format_date(fake.date_of_birth(minimum_age=18, maximum_age=90))
    test_date = format_date(fake.date_between(start_date="-1y", end_date="today"))
    test_time = fake.time(pattern="%p %I:%M").replace("AM", "上午").replace("PM", "下午")
    reliability = random.choice(["Good", "Fair"])

    print(filename, doctor_name, patient_name, serial_number, gender, birthdate, test_date)

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # font = ImageFont.truetype("msjh.ttc", 48)  # Microsoft JhengHei (Windows)
    # font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 36)
    font = ImageFont.truetype("fonts/NotoSansCJKtc-VF.otf", 28)

    left_label_x = max(20, img.width // 60)
    left_value_x = max(left_label_x + 115, img.width // 10)
    right_label_x = max(img.width // 2, left_value_x + 280)
    right_value_x = right_label_x + 150
    right_edge_x = img.width - 160
    top_y = max(30, img.height // 18)
    line_gap = 38
    text_color = (0, 0, 0)

    rows = [
        ((left_label_x, top_y), "Name", (left_value_x, top_y), f"{patient_number}，{patient_name}"),
        ((left_label_x, top_y + line_gap), "聽檢編號:", (left_value_x, top_y + line_gap), serial_number),
        ((left_label_x, top_y + line_gap * 2), "性別:", (left_value_x, top_y + line_gap * 2), gender),
        ((left_label_x, top_y + line_gap * 3), "Birthdate", (left_value_x, top_y + line_gap * 3), birthdate),
        ((right_label_x, top_y), "Test Date", (right_value_x, top_y), test_date),
        ((right_label_x, top_y + line_gap), "Test Time", (right_value_x, top_y + line_gap), test_time),
        ((right_label_x, top_y + line_gap * 2), "Reliability", (right_edge_x, top_y + line_gap * 2), reliability),
        ((right_label_x + 45, top_y + line_gap * 3), "醫師:", (right_edge_x, top_y + line_gap * 3), doctor_name),
    ]

    for label_pos, label, value_pos, value in rows:
        draw.text(label_pos, label, fill=text_color, font=font)
        draw.text(value_pos, value, fill=text_color, font=font)

    output_file = os.path.join(output_path, f"f{filename}")
    img.save(output_file)

    data = {
        "path": output_file,
        "doctor": doctor_name,
        "patient": patient_name,
        "serial": serial_number,
        "gender": gender,
        "birthdate": birthdate,
        "test_date": test_date,
    }

    return output_file, data


def _flatten_ocr_tokens(result):
    if result is None:
        return []

    if isinstance(result, str):
        return [result]

    if isinstance(result, (list, tuple)):
        tokens = []
        for item in result:
            tokens.extend(_flatten_ocr_tokens(item))
        return tokens

    return [str(result)]


def _clean_ocr_text(result):
    raw_text = "\n".join(_flatten_ocr_tokens(result))

    return raw_text, re.sub(r"\s+", "", raw_text)


def _extract_between(text, label_patterns, stop_patterns, value_pattern=".+?"):
    labels = "|".join(label_patterns)
    stops = "|".join(stop_patterns)
    match = re.search(rf"(?:{labels})[:：]?({value_pattern})(?=(?:{stops})|$)", text, re.IGNORECASE)
    return match.group(1).strip(" ,，:：") if match else "N/A"


def _path_lookup_keys(path):
    if path is None:
        return []

    raw = str(path)
    path_obj = Path(raw)
    keys = [
        raw,
        str(path_obj),
        str(path_obj.resolve(strict=False)),
        path_obj.name,
    ]

    return list(dict.fromkeys(key for key in keys if key))


def _extract_doctor(text):
    match = re.search(
        r"(?:醫生|醫師|医生|医師)"
        r"\s*[:：]?\s*"
        r"([\u4e00-\u9fff]{2,4})",
        text,
    )

    if match:
        return match.group(1)

    # 英文 Doctor fallback
    match = re.search(
        r"(?:Doctor|Dr\.?)\s*[:：]?\s*"
        r"([A-Za-z][A-Za-z .'-]{1,40})",
        text,
        re.IGNORECASE,
    )

    return match.group(1).strip() if match else "N/A"


def _extract_serial(text, field_stops):
    serial_match = re.search(r"(SN-[A-Za-z]{5}-\d{6})", text)
    if serial_match:
        return serial_match.group(1)

    serial_labels = [
        r"聽檢編號",
        r"听检编号",
        r"檢號",
        r"检號",
        r"序號",
        r"號",
        r"号",
        r"MRN",
        r"Serial",
    ]
    labels = "|".join(serial_labels)
    stops = "|".join(field_stops)

    labeled_match = re.search(
        rf"(?:{labels})[:：]?([A-Za-z0-9]+-[A-Za-z0-9]+)(?=(?:{stops})|$)",
        text,
        re.IGNORECASE,
    )
    if labeled_match:
        return labeled_match.group(1).strip(" ,，:：")

    return _extract_between(
        text,
        serial_labels,
        field_stops,
        r"[A-Za-z0-9\-]+?",
    )


def _clean_value(value):
    if value == "N/A":
        return value

    for word in ["姓名", "醫生", "醫師", "医師", "序號", "聽檢編號", "听检编号", "檢號", "检號", "號", "号"]:
        value = value.replace(word, "")

    if "," in value:
        value = value.split(",")[-1]
    if "，" in value:
        value = value.split("，")[-1]

    return value.strip(" ,，:：") or "N/A"


def _normalize_gender(value):
    if value == "N/A":
        return value

    gender = value.strip(" ,，:：")
    gender_map = {
        "M": "Male",
        "MALE": "Male",
        "男": "Male",
        "F": "Female",
        "FEMALE": "Female",
        "女": "Female",
    }
    return gender_map.get(gender.upper(), gender)


def _extract_date(text, labels):
    label_pattern = "|".join(labels)

    match = re.search(
        rf"(?:{label_pattern})\s*[:：]?\s*"
        rf"(\d{{4}}[/-]\d{{1,2}}[/-]\d{{1,2}})",
        text,
        re.IGNORECASE,
    )

    return match.group(1) if match else "N/A"


def _extract_gender(text):
    match = re.search(
        r"(?:性別|性别|Gender|Sex)\s*[:：]?\s*"
        r"(Male|Female|M|F|男|女)",
        text,
        re.IGNORECASE,
    )

    return match.group(1) if match else "N/A"


def _extract_field(items, labels, pattern=None, max_lookahead=4):
    # 1. label/value is different OCR block
    for i, item in enumerate(items):
        clean = item.strip().rstrip(":：")

        if clean in labels:
            end = min(len(items), i + 1 + max_lookahead)

            for j in range(i + 1, end):
                value = _clean_value(items[j])

                if pattern is None:
                    return value

                match = re.fullmatch(pattern, value, re.IGNORECASE)
                if match:
                    return match.group(0)

    # 2. label/value inside the same OCR block
    for item in items:
        for label in labels:
            match = re.search(
                rf"{re.escape(label)}\s*[:：]?\s*(.+)",
                item,
                re.IGNORECASE,
            )

            if match:
                value = _clean_value(match.group(1))

                if pattern is None:
                    return value

                value_match = re.search(pattern, value, re.IGNORECASE)
                if value_match:
                    return value_match.group(0)

    return "N/A"


def extract_data(result, file_path):
    print(f"  → Raw Result:\n {result}")
    # _, text = _clean_ocr_text(result)
    # print(f"  → OCR Result:\n {text}")

    # field_stops = [
    #     "Name",
    #     "姓名",
    #     "聽檢編號",
    #     "听检编号",
    #     "檢號",
    #     "检號",
    #     "號",
    #     "号",
    #     "性別",
    #     "性别",
    #     "Gender",
    #     "Sex",
    #     "Birthdate",
    #     "BirthDate",
    #     "DOB",
    #     "出生日期",
    #     "生日",
    #     "TestDate",
    #     "測試日期",
    #     "檢查日期",
    #     "测试日期",
    #     "检查日期",
    #     "TestTime",
    #     "Reliability",
    #     "醫生",
    #     "醫師",
    #     "医師",
    #     "医生",
    #     "Doctor",
    #     "Dr",
    #     "序號",
    #     r"IP\d*",
    #     "Testedby",
    #     "125",
    #     "250",
    #     "500",
    #     "1K",
    #     "2K",
    #     "4K",
    #     "8K",
    # ]

    # doctor_name = _extract_doctor(text)
    # patient_name = _extract_between(
    #     text,
    #     [r"姓名"],
    #     field_stops,
    #     r"[\u4e00-\u9fffA-Za-z.\-]+?",
    # )
    # if patient_name == "N/A":
    #     patient_name = _extract_between(
    #         text,
    #         [r"Name"],
    #         field_stops,
    #     )

    # serial_number = _extract_serial(text, field_stops)

    # birthdate = _extract_date(
    #     text,
    #     [r"Birthdate", r"BirthDate", r"DOB", r"出生日期", r"生日"]
    # )
    # test_date = _extract_date(
    #     text,
    #     [r"TestDate", r"測試日期", r"檢查日期", r"测试日期", r"检查日期"]
    # )

    # doctor_name = _clean_value(doctor_name)
    # patient_name = _clean_value(patient_name)
    # gender = _normalize_gender(_extract_gender(text))

    patient_name = _extract_field(
        result,
        ["Name", "姓名"],
        r"[\u4e00-\u9fff]{2,4}",
    )

    test_date = _extract_field(
        result,
        ["Test Date", "TestDate"],
        r"\d{4}[/-]\d{1,2}[/-]\d{1,2}"
    )

    serial_number = _extract_field(
        result,
        ["編號", "编号", "编號", "序號", "序号", "機编號", "聽檢編號"],
        r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+"
    )

    gender = _extract_field(
        result,
        ["性別", "性别", "Gender", "Sex"],
        r"Male|Female|M|F|男|女"
    )

    birthdate = _extract_field(
        result,
        ["Birthdate", "BirthDate", "DOB"],
        r"\d{4}[/-]\d{1,2}[/-]\d{1,2}"
    )

    doctor_name = _extract_field(
        result,
        ["醫師", "医師", "醫生", "医生", "Doctor"]
    )

    print(
        "  → "
        f"姓名: {patient_name}, Test Date: {test_date}, 序號: {serial_number},"
        f"性別: {gender}, Birthdate: {birthdate}, 醫生: {doctor_name}"
    )

    data = {
        "path": file_path,
        "patient": patient_name,
        "test_date": test_date,
        "serial": serial_number,
        "gender": gender,
        "birthdate": birthdate,
        "doctor": doctor_name,
    }

    return data


def clean_patient_name(value):
    parts = re.split(r"[,，]", value)

    if len(parts) >= 2:
        return parts[-1].strip()

    return value.strip()


def extract_hunyuan_data(text, file_path):
    data = {
        "path": file_path,
        "patient": "N/A",
        "test_date": "N/A",
        "serial": "N/A",
        "gender": "N/A",
        "birthdate": "N/A",
        "doctor": "N/A",
    }

    field_map = {
        "Name": "patient",
        "姓名": "patient",

        "聽檢編號": "serial",
        "听检编号": "serial",

        "性別": "gender",
        "性别": "gender",

        "生日": "birthdate",
        "Birthdate": "birthdate",

        "檢查日期": "test_date",
        "检查日期": "test_date",
        "Test Date": "test_date",

        "醫生": "doctor",
        "醫師": "doctor",
        "医生": "doctor",
        "医师": "doctor",
        "Doctor": "doctor",
    }

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        match = re.match(
            r"^(.+?)\s*[:：]\s*(.+)$",
            line,
        )

        if not match:
            continue

        key = match.group(1).strip()
        value = match.group(2).strip().rstrip("。,.，")

        if key not in field_map:
            continue

        field = field_map[key]

        if field == "patient":
            value = clean_patient_name(value)

        data[field] = value

    return data


def save_extracted_record(data, file_path, audiogram_df=None):
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
            gender=None if data.get("gender") == "N/A" else data.get("gender"),
            birthdate=None if data.get("birthdate") == "N/A" else data.get("birthdate"),
            test_date=None if data.get("test_date") == "N/A" else data.get("test_date"),
            audiogram=file_path,
            audiogram_df=audiogram_df,
        )
        print(f"  → [DB] Saved record for serial {data['serial']}")


def validation(org, new):
    print("\n=== Validation Results ===")

    fields = [
        ("doctor", "Doctor"),
        ("patient", "Patient"),
        ("serial", "Serial"),
        ("gender", "Gender"),
        ("birthdate", "Birthdate"),
        ("test_date", "Test Date"),
    ]
    correct_counts = {key: 0 for key, _ in fields}
    total_counts = {key: 0 for key, _ in fields}

    org_map = {}
    for original_data in org:
        for key in _path_lookup_keys(original_data.get("path")):
            org_map.setdefault(key, original_data)

    for data in new:
        path = data.get("path")

        original_data = None
        for key in _path_lookup_keys(path):
            original_data = org_map.get(key)
            if original_data is not None:
                break

        if original_data is None:
            print(f"⚠️ No original data for {path}")
            continue

        mismatches = []
        for key, label in fields:
            if key not in original_data:
                continue

            expected = original_data.get(key, "N/A")
            actual = data.get(key, "N/A")
            is_match = actual == expected

            total_counts[key] += 1
            if is_match:
                correct_counts[key] += 1
            else:
                mismatches.append((label, actual, expected))

        if not mismatches:
            print(f"✅ Data matches for {path}")
        else:
            print(f"❌ {path}")
            for label, actual, expected in mismatches:
                print(f"  {label}: {actual} != {expected}")

    print("\n=== Accuracy ===")
    for key, label in fields:
        total = total_counts[key]
        correct = correct_counts[key]
        accuracy = correct / total if total else 0
        print(f"{label} Accuracy: {correct}/{total} = {accuracy:.2%}")

    name_keys = ["doctor", "patient"]
    name_total = sum(total_counts[key] for key in name_keys)
    name_correct = sum(correct_counts[key] for key in name_keys)
    name_accuracy = name_correct / name_total if name_total else 0
    print(
        "Doctor + Patient Name Accuracy: "
        f"{name_correct}/{name_total} = {name_accuracy:.2%}"
    )


def crop_top(img, save_crop=False):
    crop = img.crop((
        0,
        0,
        img.width,
        img.height // 3
    ))

    # Save the cropped area for debugging
    if save_crop:
        output_path = "outputs/debug/crop_top.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        crop.save(output_path)

    return crop
