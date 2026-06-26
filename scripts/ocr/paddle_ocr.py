from paddleocr import PaddleOCR
from PIL import Image
from ..utils import crop_top, extract_data, save_extracted_record  


ocr = None


def init_ocr():
    global ocr

    if ocr is None:
        ocr = PaddleOCR(
            use_doc_orientation_classify=False, # Disables document orientation classification model via this parameter
            use_doc_unwarping=False, # Disables text image rectification model via this parameter
            use_textline_orientation=False, # Disables text line orientation classification model via this parameter
            device="gpu",
        )


def paddle_ocr(file_paths):
    init_ocr()
    data_list = []
    tmp_path = "outputs/debug/crop_top.png"

    for file_path in file_paths[:30]:
        print(f"\nProcessing: {file_path}")
        img = Image.open(file_path)
        img = crop_top(img)
        result = ocr.predict(tmp_path)
        texts = [str(res['rec_texts']) for res in result]
        text = "\n".join(texts)
        data = extract_data(text.splitlines(), file_path)
        data_list.append(data)
        save_extracted_record(data, file_path)

    # for file_path in file_paths:
    #     print(f"\nProcessing: {file_path}")
    #     result = ocr.predict(file_path)
    #     texts = []
    #     for res in result:
    #         texts.extend(res['rec_texts'])

    #     text = "".join(texts)
    #     data_list.append(text)

    return data_list


# ocr = PaddleOCR(lang="en") # Uses English model by specifying language parameter
if __name__ == "__main__":
    init_ocr()
    IMAGE_PATH = "outputs/ocr/f17.jpg"
    OUTPUT_PATH = "outputs/ocr/paddle"
    result = ocr.predict(IMAGE_PATH)
    texts = [str(res['rec_texts']) for res in result]
    print("\n".join(texts))