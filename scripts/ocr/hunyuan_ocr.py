from PIL import Image
import torch

from ..utils import (
    crop_top,
    extract_hunyuan_data,
    prepare_input_image,
)


def clean_repeated_substrings(text):
    """Clean repeated substrings in text"""
    n = len(text)

    if n < 8000:
        return text

    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length

        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[:n - length * (count - 1)]

    return text


# =========================
# Global model
# =========================

model = None
processor = None

MODEL_PATH = "tencent/HunyuanOCR"


def build_messages(img):
    return [
        {
            "role": "system",
            "content": "",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                },
                {
                    "type": "text",
                    "text": """
                        請辨識圖片中的病人資料。

                        只允許輸出以下 6 行，格式必須完全一致：
                        Name: <姓名>
                        聽檢編號: <編號>
                        性別: <Male 或 Female>
                        Birthdate: <YYYY/M/D>
                        Test Date: <YYYY/M/D>
                        醫師: <姓名>
                        """,
                },
            ],
        },
    ]


def init_model():
    global model
    global processor

    if model is not None:
        return

    try:
        from transformers import (
            AutoProcessor,
            HunYuanVLForConditionalGeneration,
        )
    except ImportError as exc:
        raise ImportError(
            "HunyuanOCR requires transformers. "
            "Install with: uv pip install 'transformers>=5.13.0' accelerate"
        ) from exc

    print("Loading HunyuanOCR...")

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        backend="pil",
    )

    model = HunYuanVLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    print("HunyuanOCR loaded.")


def run_ocr(img, max_new_tokens=1024):
    """
    Run HunyuanOCR on one PIL image.
    Return OCR text.
    """
    init_model()

    if img.mode != "RGB":
        img = img.convert("RGB")

    messages = build_messages(img)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move inputs to GPU
    inputs = inputs.to(model.device)

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids = output_ids[:, input_length:]

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    text = text.strip()
    text = clean_repeated_substrings(text)

    return text


def hunyuanocr(file_paths):
    init_model()

    data_list = []

    for file_path in file_paths[:]:
        file_path = prepare_input_image(file_path)

        print(f"\nProcessing: {file_path}")

        with Image.open(file_path) as img:
            img = img.convert("RGB")
            img = crop_top(img)

            text = run_ocr(img)

        print("OCR result:")
        print(text)

        data = extract_hunyuan_data(text, file_path)

        data_list.append(data)

    return data_list


if __name__ == "__main__":
    IMAGE_PATH = "outputs/ocr/f17.jpg"
    # IMAGE_PATH = "scripts/ocr/TC-STR/images/sign_02614_997_鶯歌.jpg"

    img_path = prepare_input_image(IMAGE_PATH)

    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = crop_top(img)

        text = run_ocr(img)

    print("\n================ OCR RESULT ================")
    print(text)

    if ":" in text:
        text = text.split(":")[-1].strip()

    if "：" in text:
        text = text.split("：")[-1].strip()

    print("\n================ CLEAN RESULT ================")
    print(text)
