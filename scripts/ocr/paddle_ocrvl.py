from pathlib import Path
from paddleocr import PaddleOCRVL
from PIL import Image
from ..utils import crop_top, extract_data, save_extracted_record


pipeline = None


def init_pipeline():
    global pipeline

    if pipeline is None:
        # NVIDIA GPU
        pipeline = PaddleOCRVL(
            # pipeline_version="v1.5",

            # use_doc_orientation_classify=False,
            # use_doc_unwarping=False,
            # use_layout_detection=False,
            # use_chart_recognition=False,
            # use_seal_recognition=False,
            # use_ocr_for_image_block=False,

            # format_block_content=False,
            # merge_layout_blocks=False,
            device="gpu",
            engine="transformers",
            )


def paddle_ocrvl(file_paths):
    init_pipeline()
    data_list = []
    tmp_path = "outputs/debug/crop_top.png"

    for file_path in file_paths[:30]:
        print(f"\nProcessing: {file_path}")
        img = Image.open(file_path)
        img = crop_top(img)
        output = pipeline.predict(
            input=tmp_path,
            prompt_label="ocr",
        )

        texts = []
        for block in output[0]["parsing_res_list"]:
            if block.label == "text":
                texts.append(block.content)

        data = extract_data(texts, file_path)
        data_list.append(data)
        save_extracted_record(data, file_path)
        
    # for file_path in file_paths:
    #     print(f"\nProcessing: {file_path}")
    #     output = pipeline.predict(
    #         input=file_path,
    #         prompt_label="ocr",
    #     )

    #     texts = []
    #     for block in output[0]["parsing_res_list"]:
    #         texts.append(block.content)

    #     text = "".join(texts)
    #     data_list.append(text)

    return data_list


# pipeline = PaddleOCRVL(use_doc_orientation_classify=True) # Use use_doc_orientation_classify to enable/disable document orientation classification model
# pipeline = PaddleOCRVL(use_doc_unwarping=True) # Use use_doc_unwarping to enable/disable document unwarping module
# pipeline = PaddleOCRVL(use_layout_detection=False) # Use use_layout_detection to enable/disable layout analysis module


if __name__ == "__main__":
    init_pipeline()
    # IMAGE_PATH = "outputs/ocr/f31.jpg"
    IMAGE_PATH = "scripts/ocr/TC-STR/images/billboard_00000_010_雜貨舖.jpg"
    OUTPUT_PATH = "outputs/ocr/paddlevl"
    # output_dir = Path(OUTPUT_PATH) 8983.28 42.28%
    # output_dir.mkdir(parents=True, exist_ok=True)
    output = pipeline.predict(
        input=IMAGE_PATH,
        # prompt_label="Extract all text exactly as written. Do not analyze layout, tables, charts, or document structure.",
    )
    for res in output:
        res.print()
    texts = []

    for block in output[0]["parsing_res_list"]:
        texts.append(block.content)

    text = "".join(texts)
    print(f"  → OCR Result: {text}")
    # for res in output:
    #     print(res['parsing_res_list']) ## Print the structured prediction output
        # res.save_to_json(save_path=output_dir) ## Save the current image's structured result in JSON format
        # res.save_to_markdown(save_path=output_dir) ## Save the current image's result in Markdown format
        # res.save_to_word(save_path=OUTPUT_PATH) ## Save the current image's result in Word format