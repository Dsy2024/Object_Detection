import cv2
import gradio as gr
import numpy as np
import pandas as pd

from yolo import yolo_infer  # Assuming this is your custom module

# --- 1. Define placeholders for your specific YOLO classes ---
# Since your model is "detect_OX", a common mapping is 0 -> O, 1 -> X.
# IMPORTANT: You need to replace these placeholder names with your actual class names.
CLASS_NAMES = ["Symbol O", "Symbol X"]  # e.g., ['circle', 'cross'] or ['O', 'X']

# Define colors for different classes (BGR format) for visual drawing
CLASS_COLORS = [(255, 165, 0), (0, 255, 0)]


# --- Keep your existing OpenCV functions intact ---


def detect_and_draw_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (300, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 300))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
    grid_mesh = cv2.add(h_lines, v_lines)
    contours, _ = cv2.findContours(
        grid_mesh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return x, y, w, h


def refine_crop(img, initial_bbox):
    x, y, w, h = initial_bbox
    crop = img[y : y + h, x : x + w]
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_crop, 200, 255, cv2.THRESH_BINARY_INV)

    row_sums = np.sum(binary, axis=1)
    col_sums = np.sum(binary, axis=0)

    def find_bounds(arr, min_val):
        indices = np.where(arr > min_val)[0]
        if len(indices) == 0:
            return 0, len(arr)
        return indices[0], indices[-1]

    y_start, y_end = find_bounds(row_sums, w * 0.2 * 255)
    x_start, x_end = find_bounds(col_sums, h * 0.2 * 255)

    new_x = x + x_start
    new_y = y + y_start
    new_w = x_end - x_start
    new_h = y_end - y_start

    # We can skip saving intermediate files unless you really need them.
    # We will just pass the refined crop image object to the next function.
    refined_crop_img = img[new_y : new_y + new_h, new_x : new_x + new_w]
    return new_x, new_y, new_w, new_h, refined_crop_img


def mark_lines(crop_img):
    """Modified to take an image object instead of a path."""
    if crop_img is None:
        return [], []
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    h_img, w_img = crop_img.shape[:2]

    h_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    h_connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, h_close_kernel)
    h_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(h_connected, cv2.MORPH_OPEN, h_open_kernel)

    v_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    v_connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, v_close_kernel)
    v_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(v_connected, cv2.MORPH_OPEN, v_open_kernel)

    row_sums = np.sum(horizontal_lines, axis=1)
    y_indices = np.where(row_sums > (w_img * 0.9 * 255))[0]

    col_sums = np.sum(vertical_lines, axis=0)
    x_indices = np.where(col_sums > (h_img * 0.9 * 255))[0]

    def group_indices(indices, gap=10):
        if len(indices) == 0:
            return []
        groups = []
        curr = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] <= indices[i - 1] + gap:
                curr.append(indices[i])
            else:
                groups.append(int(np.mean(curr)))
                curr = [indices[i]]
        groups.append(int(np.mean(curr)))
        return groups

    final_x = group_indices(x_indices)
    final_y = group_indices(y_indices)

    return final_x, final_y


DEFAULT_FREQUENCIES = [
    125,
    250,
    500,
    750,
    1000,
    1500,
    2000,
    3000,
    4000,
    6000,
    8000,
    12000,
]

# --- Gradio specific wrapper ---


def process_audiogram(image_filepath):
    """Main function called by Gradio."""
    try:
        # Paths
        model_path = "outputs/yolo11n_detect_OX/weights/best.pt"
        # A name for the temporarily cropped file that YOLO will read
        temp_crop_path = "temp_cropped_audiogram.png"
        # Output paths for results
        final_annotated_img_path = "annotated_symbols_audiogram.jpg"
        csv_output_path = "audiogram_results.csv"

        # 1. Run OpenCV crop and grid detection
        img = cv2.imread(image_filepath)
        x, y, w, h = detect_and_draw_grid(img)
        # Modified refine_crop to return the image object
        new_x, new_y, new_w, new_h, cropped_img = refine_crop(img, (x, y, w, h))

        # Modified mark_lines to take an image object
        final_x, final_y = mark_lines(cropped_img)

        if len(final_x) < 2 or len(final_y) < 2:
            raise ValueError("Could not detect enough grid lines.")

        final_x = final_x[1:-1]
        final_y = final_y[1:-1]

        # Map detected x-coordinates to standard frequencies
        px_to_hz = dict(zip(map(int, final_x), map(int, DEFAULT_FREQUENCIES)))
        y_ratio = final_y[-1]

        # Save the cropped image to a temporary file so YOLO can read it
        cv2.imwrite(temp_crop_path, cropped_img)

        # 2. Run YOLO inference on the newly created temporary file
        results = yolo_infer(
            model_path=model_path,
            img=temp_crop_path,
            # save_name=yolo_output_img # Commented out; we will do the drawing explicitly for a visual result.
        )

        # Prepare the cropped image for drawing annotations directly in Python
        # We start with the original cropped image object.
        annotated_img = cropped_img.copy()

        rows = []
        # 3. Process YOLO results, draw bounding boxes, and extract data
        for r in results:
            for b in r.boxes:
                # Get xyxy coordinates for rectangle drawing
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cls = int(b.cls.item())
                conf = float(b.conf.item()) if hasattr(b, "conf") else np.nan

                # Select a color and name for the class, with a safeguard for unknown classes
                color = CLASS_COLORS[cls % len(CLASS_COLORS)]
                # class_name = CLASS_NAMES[cls % len(CLASS_NAMES)]

                # Draw the bounding box on the image
                cv2.rectangle(
                    annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2
                )

                # Draw class label and confidence score above the box
                # label = f"{class_name}: {conf:.2f}"
                # cv2.putText(
                #     annotated_img,
                #     label,
                #     (int(x1), int(y1) - 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.5,
                #     color,
                #     2,
                # )

                # Extract data for CSV
                x_center = b.xywh[0, 0].item()
                y_center = b.xywh[0, 1].item()
                # find nearest frequency column
                k = min(px_to_hz.keys(), key=lambda p: abs(p - x_center))
                nearest_hz = px_to_hz[k]
                # Calculate dB
                db_level = -20 + round((y_center / y_ratio) * 140)
                rows.append(
                    {"cls": cls, "hz": nearest_hz, "db": db_level, "conf": conf}
                )

        # Save the finally annotated image to a file
        cv2.imwrite(final_annotated_img_path, annotated_img)

        # 4. Format the Data for output
        df = pd.DataFrame(rows)

        if df.empty:
            return (
                final_annotated_img_path,
                pd.DataFrame({"Message": ["No symbols detected."]}),
                None,
            )

        wide = df.pivot(index="cls", columns="hz", values="db")
        wide = wide.reindex(columns=DEFAULT_FREQUENCIES).sort_index()

        # Save CSV file
        wide.to_csv(csv_output_path, index_label="cls", float_format="%.2f")

        # Gradio expects: Image file path, Dataframe object, File file path (for download)
        # We are now returning the explicitly annotated image path.
        return final_annotated_img_path, wide.reset_index(), csv_output_path

    except Exception as e:
        return None, pd.DataFrame({"Error": [str(e)]}), None


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
