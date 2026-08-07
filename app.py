import gradio as gr

from scripts.add_fake_data import web_results
from scripts.db import DB_PATH, get_database_snapshot, init_db, save_hearing_results
from scripts.validate_ocr import ocr
from src.detect_symbol import process_audiogram


# SQLite does not need a separate server. Ensure its file/tables are ready whenever
# the web application starts.
init_db()


def process_all(image_path):
    if not image_path:
        raise gr.Error("請先上傳聽力圖。")

    output_image, df, csv = process_audiogram(image_path)
    ocr_result = ocr(image_path)
    saved_count = save_hearing_results(ocr_result["serial"], df)
    if saved_count:
        save_status = f"已將 {saved_count} 筆聽力結果存入病例 {ocr_result['serial']}。"
    elif ocr_result["serial"] == "N/A":
        save_status = "未辨識到有效病例序號，因此沒有寫入聽力結果。"
    else:
        save_status = "沒有可儲存的聽力結果；資料庫中的既有結果未被刪除。"
    return (
        output_image,
        df,
        csv,
        ocr_result["doctor"],
        ocr_result["patient"],
        ocr_result["serial"],
        save_status,
    )


with gr.Blocks(title="Audiogram Digitizer") as demo:
    gr.Markdown("# 聽力圖數位化系統")

    with gr.Tab("聽力圖辨識"):
        gr.Markdown("上傳聽力圖，系統會辨識符號、擷取資料，並將有效的病患資料保存至資料庫。")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="filepath", label="上傳聽力圖")
                submit_btn = gr.Button("開始辨識", variant="primary")
            with gr.Column():
                output_image = gr.Image(type="filepath", label="符號辨識結果")
                output_dataframe = gr.Dataframe(label="擷取的 dB 數值")
                output_csv = gr.File(label="下載 CSV")
                doctor = gr.Textbox(label="醫師")
                patient = gr.Textbox(label="病患")
                serial = gr.Textbox(label="序號")
                db_save_status = gr.Textbox(label="資料庫儲存狀態", interactive=False)

        submit_btn.click(
            fn=process_all,
            inputs=input_image,
            outputs=[
                output_image,
                output_dataframe,
                output_csv,
                doctor,
                patient,
                serial,
                db_save_status,
            ],
        )

    with gr.Tab("資料庫") as database_tab:
        gr.Markdown("## 本機資料庫內容")
        db_status = gr.Markdown()
        refresh_db = gr.Button("重新整理資料庫", variant="primary")
        cases_table = gr.Dataframe(label="病例（含病患資料）", interactive=False)
        hearing_results_table = gr.Dataframe(
            label="PTA／YOLO 聽力辨識結果", interactive=False
        )
        patients_table = gr.Dataframe(label="病患", interactive=False)

        db_outputs = [patients_table, cases_table, hearing_results_table, db_status]
        refresh_db.click(fn=get_database_snapshot, outputs=db_outputs)
        database_tab.select(fn=get_database_snapshot, outputs=db_outputs)

    with gr.Tab("假資料生成") as fake_data_tab:
        gr.Markdown("## 假資料生成結果")
        gr.Markdown("來源圖片放在 `data/raw`；按下按鈕後會產生含假姓名與序號的圖片。")
        generate_btn = gr.Button("生成假資料", variant="primary")
        fake_status = gr.Markdown()
        fake_gallery = gr.Gallery(label="生成圖片", columns=2, height="auto")
        fake_table = gr.Dataframe(label="生成明細與儲存位置", interactive=False)

        fake_outputs = [fake_gallery, fake_table, fake_status]
        generate_btn.click(fn=lambda: web_results(generate=True), outputs=fake_outputs)
        fake_data_tab.select(fn=web_results, outputs=fake_outputs)

    demo.load(fn=get_database_snapshot, outputs=db_outputs)
    demo.load(fn=web_results, outputs=fake_outputs)


if __name__ == "__main__":
    print(f"Database: {DB_PATH}")
    demo.launch(share=False)
