from PIL import Image
from ..utils import (
    crop_top,
    extract_data,
    pil_to_ocr_image,
    prepare_input_image,
    save_crop_image,
)


ocr = None


def _extract_texts(result):
    texts = []

    if not result:
        return texts

    if isinstance(result, dict):
        rec_texts = result.get("rec_texts", [])
        if isinstance(rec_texts, str):
            texts.append(rec_texts)
        else:
            texts.extend(map(str, rec_texts))
        return texts

    if isinstance(result, (list, tuple)):
        if (
            len(result) >= 2
            and isinstance(result[1], (list, tuple))
            and result[1]
            and isinstance(result[1][0], str)
        ):
            texts.append(result[1][0])
            return texts

        for item in result:
            texts.extend(_extract_texts(item))

    return texts


def _run_ocr(image_input):
    if hasattr(ocr, "predict"):
        return ocr.predict(image_input)

    return ocr.ocr(image_input, cls=False)


def init_ocr():
    global ocr

    if ocr is None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR could not import its PaddlePaddle backend. "
                "Install the correct package (`paddlepaddle`, or `paddlepaddle-gpu` "
                "for your CUDA setup) and remove the unrelated `paddle` package."
            ) from exc

        ocr = PaddleOCR(
            # lang="chinese_cht",
            use_doc_orientation_classify=False, # Disables document orientation classification model via this parameter
            use_doc_unwarping=False, # Disables text image rectification model via this parameter
            use_textline_orientation=False, # Disables text line orientation classification model via this parameter
            device="gpu",
        )


def paddle_ocr(file_paths):
    init_ocr()
    data_list = []

    for file_path in file_paths:
        file_path = prepare_input_image(file_path)
        print(f"\nProcessing: {file_path}")
        img = Image.open(file_path)
        img = crop_top(img)
        try:
            result = _run_ocr(pil_to_ocr_image(img))
        except (TypeError, ValueError, AssertionError):
            result = _run_ocr(save_crop_image(img))
        texts = _extract_texts(result)
        text = "\n".join(texts)
        data = extract_data(text.splitlines(), file_path)
        data_list.append(data)

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
    result = _run_ocr(IMAGE_PATH)
    texts = _extract_texts(result)
    print("\n".join(texts))
