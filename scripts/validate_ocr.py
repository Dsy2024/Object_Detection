
from PIL import Image

import easyocr
import time
import pandas as pd
from pathlib import Path
from rapidocr import LangDet, RapidOCR
from .utils import crop_top, extract_data, save_extracted_record, validation


GT_CSV = "outputs/ocr/fake_ground_truth.csv"
DATASET_DIR = Path("scripts/ocr/TC-STR")
LABEL_FILE = DATASET_DIR / "test_labels.txt"


def paddle_ocr(file_paths):
    from .ocr.paddle_ocr import paddle_ocr as run_paddle_ocr

    return run_paddle_ocr(file_paths)


def paddle_ocrvl(file_paths):
    from .ocr.paddle_ocrvl import paddle_ocrvl as run_paddle_ocrvl

    return run_paddle_ocrvl(file_paths)


def hunyuanocr(file_paths):
    from .ocr.hunyuan_ocr import hunyuanocr as run_hunyuanocr

    return run_hunyuanocr(file_paths)


def rapidocr(file_paths):
    engine = RapidOCR(
        params={
            "Det.lang_type": LangDet.CH,
        }
    )

    data_list = []
    tmp_path = "outputs/debug/crop_top.png"

    for file_path in file_paths[:30]:
        print(f"\nProcessing: {file_path}")
        img = Image.open(file_path)
        img = crop_top(img)
        result = engine(tmp_path)
        data = extract_data(result.txts, file_path)

        data_list.append(data)
        save_extracted_record(data, file_path)

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
    # reader = easyocr.Reader(['en'], gpu=True)
    reader = easyocr.Reader(['ch_tra', 'en'], gpu=True)

    data_list = []
    tmp_path = "outputs/debug/crop_top.png"

    for file_path in file_paths[:30]:
        print(f"\nProcessing: {file_path}")
        img = Image.open(file_path)
        img = crop_top(img)
        results = reader.readtext(tmp_path, detail=0)
        data = extract_data(results, file_path)

        data_list.append(data)
        save_extracted_record(data, file_path)

    # for file_path in file_paths:
    #     print(f"\nProcessing: {file_path}")
    #     results = reader.readtext(file_path, detail=0)
    #     pred = "".join(results).strip()
    #     data_list.append(pred)

    return data_list


def ocr(file_path):
    
    engine = RapidOCR(
        params={
            "Det.lang_type": LangDet.CH,
        }
    )

    tmp_path = "outputs/debug/crop_top.png"

    print(f"\nProcessing: {file_path}")
    img = Image.open(file_path)
    img = crop_top(img)
    result = engine(tmp_path)
    data = extract_data(result.txts, file_path)

    save_extracted_record(data, file_path)

    return data


def main():
    gt_df = pd.read_csv(GT_CSV)

    fake_file_paths = gt_df["path"].tolist()
    fake_data = gt_df.to_dict("records")

    # extracted_data = easy_ocr(fake_file_paths)
    extracted_data = rapidocr(fake_file_paths)
    # extracted_data = paddle_ocr(fake_file_paths)
    # extracted_data = paddle_ocrvl(fake_file_paths)
    # extracted_data = hunyuanocr(fake_file_paths)

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
