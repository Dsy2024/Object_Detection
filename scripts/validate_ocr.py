
from PIL import Image

import time
import pandas as pd
from pathlib import Path
from .utils import (
    crop_top,
    extract_data,
    pil_to_ocr_image,
    prepare_input_image,
    save_extracted_record,
    validation,
)


GT_CSV = "outputs/ocr/fake_ground_truth.csv"
DATASET_DIR = Path("scripts/ocr/TC-STR")
LABEL_FILE = DATASET_DIR / "test_labels.txt"


def _as_file_list(file_paths):
    if isinstance(file_paths, (str, Path)):
        return [file_paths]

    return list(file_paths)


def paddle_ocr(file_paths):
    from .ocr.paddle_ocr import paddle_ocr as run_paddle_ocr

    return run_paddle_ocr(_as_file_list(file_paths))


def paddle_ocrvl(file_paths):
    from .ocr.paddle_ocrvl import paddle_ocrvl as run_paddle_ocrvl

    return run_paddle_ocrvl(_as_file_list(file_paths))


def hunyuanocr(file_paths):
    from .ocr.hunyuan_ocr import hunyuanocr as run_hunyuanocr

    return run_hunyuanocr(_as_file_list(file_paths))


def rapidocr(file_paths):
    file_paths = _as_file_list(file_paths)

    from rapidocr import LangDet, RapidOCR
    engine = RapidOCR(
        params={
            "Det.lang_type": LangDet.CH,
        }
    )

    data_list = []

    for file_path in file_paths[:30]:
        file_path = prepare_input_image(file_path)
        print(f"\nProcessing: {file_path}")
        img = Image.open(file_path)
        img = crop_top(img)
        result = engine(pil_to_ocr_image(img))
        data = extract_data(result.txts, file_path)

        data_list.append(data)

    # for file_path in file_paths:
    #     print(f"\nProcessing: {file_path}")
    #     result = engine(file_path)
    #     if result.txts:
    #         pred = "".join(result.txts).strip()
    #     else:
    #         pred = ""
    #     data_list.append(pred)

    return data_list


def easy_ocr(file_paths):
    file_paths = _as_file_list(file_paths)

    import easyocr
    # reader = easyocr.Reader(['en'], gpu=True)
    reader = easyocr.Reader(['ch_tra', 'en'], gpu=True)

    data_list = []

    for file_path in file_paths[:30]:
        file_path = prepare_input_image(file_path)
        print(f"\nProcessing: {file_path}")
        img = Image.open(file_path)
        img = crop_top(img)
        results = reader.readtext(pil_to_ocr_image(img), detail=0)
        data = extract_data(results, file_path)

        data_list.append(data)

    # for file_path in file_paths:
    #     print(f"\nProcessing: {file_path}")
    #     results = reader.readtext(file_path, detail=0)
    #     pred = "".join(results).strip()
    #     data_list.append(pred)

    return data_list


OCR_MODELS = {
    "rapidocr": rapidocr,
    "easyocr": easy_ocr,
    "paddleocr": paddle_ocr,
    "paddleocrvl": paddle_ocrvl,
    "hunyuanocr": hunyuanocr,
}


OCR_MODEL_LABELS = {
    "RapidOCR": "rapidocr",
    "EasyOCR": "easyocr",
    "PaddleOCR": "paddleocr",
    "PaddleOCR-VL": "paddleocrvl",
    "HunyuanOCR": "hunyuanocr",
}


def normalize_ocr_model(model_name):
    model_name = (model_name or "hunyuanocr").strip()
    model_key = OCR_MODEL_LABELS.get(model_name, model_name).lower().replace("_", "").replace("-", "")

    if model_key not in OCR_MODELS:
        choices = ", ".join(OCR_MODELS)
        raise ValueError(f"Unknown OCR model '{model_name}'. Choose one of: {choices}")

    return model_key


def ocr(file_path, model_name="hunyuanocr", audiogram_df=None, save_record=False):
    model_key = normalize_ocr_model(model_name)
    data_list = OCR_MODELS[model_key]([file_path])

    if not data_list:
        data = {
            "path": file_path,
            "patient": "N/A",
            "test_date": "N/A",
            "serial": "N/A",
            "gender": "N/A",
            "birthdate": "N/A",
            "doctor": "N/A",
        }
    else:
        data = data_list[0]

    if save_record:
        save_extracted_record(data, file_path, audiogram_df=audiogram_df)

    return data


def main():
    gt_df = pd.read_csv(GT_CSV)

    fake_file_paths = gt_df["path"].tolist()
    fake_data = gt_df.to_dict("records")

    # extracted_data = easy_ocr(fake_file_paths)
    # extracted_data = rapidocr(fake_file_paths)
    # extracted_data = paddle_ocr(fake_file_paths)
    # extracted_data = paddle_ocrvl(fake_file_paths)
    extracted_data = hunyuanocr(fake_file_paths)

    blank_df = pd.DataFrame()
    for data in extracted_data:
        save_extracted_record(data, data.get("path"), audiogram_df=blank_df)

    validation(fake_data, extracted_data)


def validate():
    file_paths = []
    ground_truths = []

    with open(LABEL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            image_rel_path, gt = line.split(maxsplit=1)

            file_paths.append(str(DATASET_DIR / image_rel_path))
            ground_truths.append(gt.strip())

    print(f"Total images: {len(file_paths)}")

    extracted_data = easy_ocr(file_paths)
    # extracted_data = rapidocr(file_paths)
    # extracted_data = paddle_ocr(file_paths)
    # extracted_data = paddle_ocrvl(file_paths)
    # extracted_data = hunyuanocr(file_paths)

    correct = 0

    for pred, gt in zip(extracted_data, ground_truths):
        if pred == gt:
            print(f"✅ Correct: {pred}")
            correct += 1
        else:
            print(f"❌ Incorrect: Predicted: {pred} | Ground Truth: {gt}")
            
    accuracy = correct / len(ground_truths)
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    # start = time.perf_counter()
    main()
    # print(f"\nTotal execution time: {time.perf_counter() - start:.2f}s")
