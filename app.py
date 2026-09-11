import gradio as gr
import pandas as pd
from src.detect_symbol import process_audiogram
from scripts.validate_ocr import OCR_MODEL_LABELS, ocr
from scripts.utils import prepare_input_image
from scripts.db import load_audiogram_df, search_patient_cases


def preview_upload(file_path):
    if not file_path:
        return None

    return str(prepare_input_image(file_path))


def process_all(file_path, ocr_model):
    image_path = prepare_input_image(file_path)

    output_image, df, csv = process_audiogram(image_path)

    ocr_result = ocr(
        image_path,
        model_name=ocr_model,
        audiogram_df=df,
        save_record=True,
    )

    return (
        output_image,
        df,
        csv,
        ocr_result["doctor"],
        ocr_result["patient"],
        ocr_result["serial"],
        ocr_result["gender"],
        ocr_result["birthdate"],
        ocr_result["test_date"],
    )


def select_case(cases_df, evt: gr.SelectData):
    if cases_df is None:
        return pd.DataFrame()

    df = pd.DataFrame(cases_df)

    row_index = evt.index[0]

    case_id = df.iloc[row_index]["case_id"]

    return load_audiogram_df(case_id)


# --- Gradio UI Definition ---

css = """
#main-row {
    height: 85vh !important;
    overflow: hidden !important;
}

/* 左右欄 */
#left-column,
#right-column {
    height: 85vh !important;
    min-height: 0 !important;
    overflow: hidden !important;
}

/* 真正負責 scroll 的區域 */
#left-scroll,
#right-scroll {
    height: 75vh !important;
    max-height: 75vh !important;

    overflow-y: scroll !important;
    overflow-x: hidden !important;

    min-height: 0 !important;
}

/* 關鍵：不要讓 Gradio 把裡面的元件縮小 */
#left-scroll > *,
#right-scroll > * {
    flex-shrink: 0 !important;
}
"""

with gr.Blocks(title="Audiogram Digitizer") as demo:

    # =====================================
    # Tab 1: Audiogram
    # =====================================

    with gr.Tab("Audiogram"):

        with gr.Row(elem_id="main-row"):

            with gr.Column(elem_id="left-column"):

                with gr.Column(elem_id="left-scroll"):
                    # Input image from user
                    input_image = gr.File(
                        type="filepath",
                        file_types=[".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".pdf"],
                        label="Upload Audiogram",
                    )
                    upload_preview = gr.Image(
                        type="filepath",
                        label="Preview",
                        interactive=False,
                    )
                    ocr_model = gr.Dropdown(
                        choices=list(OCR_MODEL_LABELS.keys()),
                        value="HunyuanOCR",
                        label="OCR Model",
                    )
                submit_btn = gr.Button("Process Audiogram", variant="primary")

            with gr.Column(elem_id="right-column"):
                with gr.Column(elem_id="right-scroll"):
                    # Gradio Image component to display the physically annotated result
                    output_image = gr.Image(
                        type="filepath", label="Detected Symbols (Drawn on Image)"
                    )
                    # Extracted Data Table for visualization
                    output_dataframe = gr.Dataframe(label="Extracted dB Values")
                    # CSV File Download link
                    output_csv = gr.File(label="Download CSV")

                    doctor = gr.Textbox(label="Doctor")
                    patient = gr.Textbox(label="Patient")
                    serial = gr.Textbox(label="Serial Number")
                    gender = gr.Textbox(label="Gender")
                    birthdate = gr.Textbox(label="Birthdate")
                    test_date = gr.Textbox(label="Test Date")

        input_image.change(
            fn=preview_upload,
            inputs=input_image,
            outputs=upload_preview,
        )

        # Wire the button to the function
        submit_btn.click(
            fn=process_all,
            inputs=[input_image, ocr_model],
            outputs=[
                output_image,
                output_dataframe,
                output_csv,
                doctor,
                patient,
                serial,
                gender,
                birthdate,
                test_date,
            ],
        )

    # =====================================
    # Tab 2: Patient Search
    # =====================================

    with gr.Tab("Patient Search"):

        gr.Markdown("## Patient Database")

        with gr.Row():
            search_name = gr.Textbox(
                label="Patient Name"
            )

            search_gender = gr.Dropdown(
                choices=["", "Male", "Female"],
                value="",
                label="Gender",
            )

            search_birthdate = gr.Textbox(
                label="Birthdate"
            )

        search_btn = gr.Button(
            "Search",
            variant="primary",
        )

        case_results = gr.Dataframe(
            label="Examination History",
            interactive=False,
        )

        audiogram_df_output = gr.Dataframe(
            label="Audiogram Data",
            interactive=False,
            wrap=False,
        )

        search_btn.click(
            fn=search_patient_cases,
            inputs=[
                search_name,
                search_gender,
                search_birthdate,
            ],
            outputs=case_results,
        )

        case_results.select(
            fn=select_case,
            inputs=[case_results],
            outputs=audiogram_df_output,
        )

    

if __name__ == "__main__":
    demo.launch(css=css, share=True)
