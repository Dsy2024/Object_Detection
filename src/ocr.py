import os
import easyocr
from rapidocr import LangDet, RapidOCR

from scripts.utils import add_fake_data, extract_data, save_extracted_record, validation


IMAGE_PATH = "data/raw/"
OUTPUT_PATH = "outputs/ocr/"


reader = easyocr.Reader(['en', 'ch_tra'], gpu=True)


def ocr(file_paths):
    engine = RapidOCR(
        params={
            # "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": LangDet.CH,
            # "Det.model_type": ModelType.MOBILE,
            # "Det.ocr_version": OCRVersion.PPOCRV5,
            # "Rec.engine_type": EngineType.ONNXRUNTIME,
            # "Rec.lang_type": LangRec.CH,
            # "Rec.model_type": ModelType.MOBILE,
            # "Rec.ocr_version": OCRVersion.PPOCRV5,
            # "EngineConfig.onnxruntime.use_coreml": True,
        }
    )

    data_list = []

    for file_path in file_paths:
        print(f"\nProcessing: {file_path}")
        result = engine(file_path)
        data = extract_data(result.txts, file_path)

        # print(f"\nProcessing: {file_path}")
        # results = reader.readtext(file_path, detail=0)
        # data = extract_data(results, file_path)

        data_list.append(data)
        save_extracted_record(data, file_path)

    return data_list


def main():
    templates = [f for f in os.listdir(IMAGE_PATH) if f.lower().endswith(".jpg")]

    fake_file_paths = []
    fake_data = []
    for template_file in templates[:1]:
        fake_file_path, data = add_fake_data(template_file, IMAGE_PATH, OUTPUT_PATH)
        fake_file_paths.append(fake_file_path)
        fake_data.append(data)

    extracted_data = ocr(fake_file_paths)

    validation(fake_data, extracted_data)


if __name__ == "__main__":
    main()