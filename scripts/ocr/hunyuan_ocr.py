from vllm import LLM, SamplingParams
from PIL import Image
from transformers import AutoProcessor
from ..utils import extract_data, save_extracted_record, crop_top


def clean_repeated_substrings(text):
    """Clean repeated substrings in text"""
    n = len(text)
    if n<8000:
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


llm = None
processor = None
sampling_params = None


def init_model():
    global llm
    global processor
    global sampling_params

    if llm is None:
        model_path = "tencent/HunyuanOCR"
        llm = LLM(model=model_path, trust_remote_code=True, max_model_len=4096, gpu_memory_utilization=0.8)
        processor = AutoProcessor.from_pretrained(model_path)
        sampling_params = SamplingParams(temperature=0, max_tokens=16384)


def hunyuanocr(file_paths):
    init_model()
    data_list = []
    tmp_path = "outputs/debug/crop_top.png"

    for file_path in file_paths[:30]:
        print(f"\nProcessing: {file_path}")

        img = Image.open(file_path)
        img = crop_top(img)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [
                {"type": "image", "image": tmp_path},
                {"type": "text", "text": "識別圖片中的文字。"}
            ]}
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = {"prompt": prompt, "multi_modal_data": {"image": [img]}}
        output = llm.generate([inputs], sampling_params)[0]

        text = output.outputs[0].text.strip()
        data = extract_data(text, file_path)
        save_extracted_record(data, file_path)
        data_list.append(data)

    return data_list


if __name__ == "__main__":
    init_model()
    IMAGE_PATH = "outputs/ocr/f17.jpg"
    # IMAGE_PATH = "scripts/ocr/TC-STR/images/sign_02614_997_鶯歌.jpg"
    img_path = IMAGE_PATH
    img = Image.open(img_path)
    img = crop_top(img)
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [
            {"type": "image", "image": img_path},
            # {"type": "text", "text": "識別圖片中的繁體中文，只輸出答案。"}
            {"type": "text", "text": "識別圖片中的文字。"}
        ]}
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = {"prompt": prompt, "multi_modal_data": {"image": [img]}}
    output = llm.generate([inputs], sampling_params)[0]
    text = output.outputs[0].text.strip()
    print(text)
    if ":" in text:
        text = text.split(":")[-1].strip()
    if "：" in text:
        text = text.split("：")[-1].strip()
    print(text)
    # print(clean_repeated_substrings(output.outputs[0].text))