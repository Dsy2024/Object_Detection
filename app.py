import gradio as gr
import pandas as pd
from pathlib import Path
from src.detect_symbol import process_audiogram
from scripts.validate_ocr import OCR_MODEL_LABELS, ocr
from scripts.utils import prepare_input_image
from scripts.db import search_patient_cases, load_audiogram_df, load_audiogram_path


def preview_upload(file_path):
    if not file_path:
        return None

    return str(prepare_input_image(file_path))


def process_all(file_path, ocr_model):
    image_path = prepare_input_image(file_path)

    output_image, df, csv = process_audiogram(image_path)

    if "Error" in df.columns:
        message = str(df["Error"].iloc[0]) if not df.empty else "Audiogram detection failed."
        raise gr.Error(message)

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


def process_batch(file_paths, ocr_model, progress=gr.Progress()):
    if not file_paths:
        raise gr.Error("Please upload at least one audiogram.")

    results = []
    total = len(file_paths)

    for i, file_path in enumerate(file_paths):
        progress(
            (i, total),
            desc=f"Processing {i + 1}/{total}: {Path(file_path).name}",
        )

        try:
            result = process_all(file_path, ocr_model)

            results.append({
                "File": Path(file_path).name,
                "Status": "Success",
                "Patient": result[4],
                "Serial Number": result[5],
                "Doctor": result[3],
                "Gender": result[6],
                "Birthdate": result[7],
                "Test Date": result[8],
                "Error": "",
            })

        except Exception as e:
            results.append({
                "File": Path(file_path).name,
                "Status": "Failed",
                "Patient": "",
                "Serial Number": "",
                "Doctor": "",
                "Gender": "",
                "Birthdate": "",
                "Test Date": "",
                "Error": str(e),
            })

    progress(1, desc="Batch processing completed")

    return pd.DataFrame(results)


def select_case(cases_df, evt: gr.SelectData):
    if cases_df is None:
        return None, pd.DataFrame()

    df = pd.DataFrame(cases_df)

    row_index = evt.index[0]
    if df.empty or row_index < 0 or row_index >= len(df):
        return None, pd.DataFrame()

    case_id = df.iloc[row_index]["case_id"]
    if pd.isna(case_id):
        return None, pd.DataFrame()

    return (
        load_audiogram_path(case_id),
        load_audiogram_df(case_id),
    )


def search_patient_ui(name, gender, birthdate):
    cases = search_patient_cases(name, gender, birthdate)

    return cases, None, pd.DataFrame()


# --- Gradio UI Definition ---

css = """
.fixed-row {
    height: 85vh !important;
    overflow: hidden !important;
}

.fixed-column {
    height: 85vh !important;
    min-height: 0 !important;
    overflow: hidden !important;
}

.scroll-column {
    height: 75vh !important;
    max-height: 75vh !important;
    min-height: 0 !important;

    overflow-y: auto !important;
    overflow-x: hidden !important;
}

.scroll-column > * {
    flex-shrink: 0 !important;
}

.audiogram-image img {
    width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
}
"""

with gr.Blocks(title="Audiogram Digitizer") as demo:

    # =====================================
    # Tab 1: Audiogram
    # =====================================

    with gr.Tab("Audiogram"):

        with gr.Row(elem_classes=["fixed-row"]):

            with gr.Column(elem_classes=["fixed-column"]):

                with gr.Column(elem_classes=["scroll-column"]):
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

            with gr.Column(elem_classes=["fixed-column"]):
                with gr.Column(elem_classes=["scroll-column"]):
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
    # Tab 2: Batch Processing
    # =====================================

    with gr.Tab("Batch Processing"):
        batch_files = gr.File(
            label="Upload Multiple Audiograms",
            file_count="multiple",
            type="filepath",
            file_types=[".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".pdf"],
        )

        batch_ocr_model = gr.Dropdown(
            choices=list(OCR_MODEL_LABELS.keys()),
            value="HunyuanOCR",
            label="OCR Model",
        )

        batch_btn = gr.Button(
            "Process All Audiograms",
            variant="primary",
        )

        batch_results = gr.Dataframe(
            label="Batch Processing Results",
            interactive=False,
        )

        batch_btn.click(
            fn=process_batch,
            inputs=[batch_files, batch_ocr_model],
            outputs=batch_results,
        )

    # =====================================
    # Tab 3: Patient Search
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

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=350):
                with gr.Column(elem_classes=["scroll-column"]):
                    audiogram_image = gr.Image(
                        label="Audiogram",
                        interactive=False,
                        elem_classes=["audiogram-image"],
                    )

            with gr.Column(scale=1, min_width=350):
                audiogram_df_output = gr.Dataframe(
                    label="Audiogram Data",
                    interactive=False,
                    wrap=False,
                )

        search_btn.click(
            fn=search_patient_ui,
            inputs=[search_name, search_gender, search_birthdate],
            outputs=[
                case_results,
                audiogram_image,
                audiogram_df_output,
            ],
        )

        case_results.select(
            fn=select_case,
            inputs=[case_results],
            outputs=[
                audiogram_image,
                audiogram_df_output,
            ],
        )

    

if __name__ == "__main__":
    demo.launch(css=css, share=True)
