import gradio as gr
from detect import process_audiogram


# --- Gradio UI Definition ---

with gr.Blocks(title="Audiogram Digitizer") as demo:
    gr.Markdown("# 📊 Audiogram Digitizer")
    gr.Markdown(
        "Upload an audiogram image. The app will detect the grid, physically draw bounding boxes for identified symbols on the image, and output the extracted data."
    )

    with gr.Row():
        with gr.Column():
            # Input image from user
            input_image = gr.Image(type="filepath", label="Upload Audiogram")
            submit_btn = gr.Button("Process Audiogram", variant="primary")

        with gr.Column():
            # Gradio Image component to display the physically annotated result
            output_image = gr.Image(
                type="filepath", label="Detected Symbols (Drawn on Image)"
            )

    with gr.Row():
        # Extracted Data Table for visualization
        output_dataframe = gr.Dataframe(label="Extracted dB Values")

    with gr.Row():
        # CSV File Download link
        output_csv = gr.File(label="Download CSV")

    # Wire the button to the function
    submit_btn.click(
        fn=process_audiogram,
        inputs=[input_image],
        outputs=[output_image, output_dataframe, output_csv],
    )

if __name__ == "__main__":
    demo.launch(share=True)
