import os
import cv2
import numpy as np
import pandas as pd
from yolo import yolo_infer
from pathlib import Path


# Paths
# model_path = "models/yolo11n_detect_OX/weights/best.pt"
model_path = "models/yolo26l_detect/weights/best.pt"
# Input image path
input_image_path = "images/144.jpg"

# CLASS_NAMES = ["O", "X"]  
CLASS_NAMES = ["<", ">", "O", "X", "[", "]", "□", "△"]
CLASS_COLORS = [
    (0, 165, 255),    # orange
    (0, 255, 0),      # green
    (255, 255, 0),    # cyan
    (255, 0, 255),    # magenta
    (255, 128, 0),    # blue
    (0, 255, 255),    # yellow
    (128, 255, 0),    # lime
    (128, 128, 128),  # gray
]


def detect_and_draw_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Isolate H/V lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (300, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 300))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

    # Combine
    grid_mesh = cv2.add(h_lines, v_lines)

    # Find contours - FIX: Using findContours
    contours, _ = cv2.findContours(
        grid_mesh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        raise ValueError("No grid detected")
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    return x, y, w, h


def refine_crop(img):
    x, y, w, h = detect_and_draw_grid(img)
    crop = img[y : y + h, x : x + w]

    # 1. Get horizontal and vertical projections
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Threshold to make ink purely black (0) and paper purely white (255)
    _, binary = cv2.threshold(gray_crop, 200, 255, cv2.THRESH_BINARY_INV)

    # 2. Find exact boundaries
    # Sum along rows (axis 1) -> locates horizontal lines
    row_sums = np.sum(binary, axis=1)
    # Sum along cols (axis 0) -> locates vertical lines
    col_sums = np.sum(binary, axis=0)

    # Find the first and last index where the sum is significant (contains a line)

    def find_bounds(arr, min_val):
        indices = np.where(arr > min_val)[0]
        if len(indices) == 0:
            return 0, len(arr)
        return indices[0], indices[-1]

    # Threshold: Line must be at least 20% of the dimension length to count as a grid line
    y_start, y_end = find_bounds(row_sums, w * 0.2 * 255)
    x_start, x_end = find_bounds(col_sums, h * 0.2 * 255)

    # 3. Adjust the original coordinates
    new_x = x + x_start
    new_y = y + y_start
    new_w = x_end - x_start
    new_h = y_end - y_start

    # Draw Rectangle (Blue)
    img_with_rect = img.copy()
    cv2.rectangle(
        img_with_rect, (x, y), (x + w, y + h), (0, 255, 0), 2
    )  # Original in Green
    cv2.rectangle(
        img_with_rect, (new_x, new_y), (new_x + new_w, new_y + new_h), (255, 0, 0), 2
    )
    # Crop
    cropped_audiogram = img[new_y : new_y + new_h, new_x : new_x + new_w]
    cv2.imwrite("outputs/image_with_rect.png", img_with_rect)
    
    return cropped_audiogram


def mark_lines(crop_img):
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
    
    # Optional: Save intermediate steps to debug
    cv2.imwrite("outputs/debug_h_connected.png", h_connected)  # See dashes connected
    cv2.imwrite("outputs/debug_h_clean.png", horizontal_lines)  # See symbols removed

    final_x = group_indices(x_indices)
    final_y = group_indices(y_indices)

    vis_img = crop_img.copy()
    # Draw straight lines based on detected coordinates
    for x in final_x:
        cv2.line(vis_img, (x, 0), (x, h_img), (0, 0, 255), 2)  # Red Vertical
    for y in final_y:
        cv2.line(vis_img, (0, y), (w_img, y), (0, 255, 0), 2)  # Green Horizontal
    # Save results
    cv2.imwrite("outputs/annotated_grid.png", vis_img)

    return final_x, final_y


DEFAULT_FREQUENCIES = [125, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000]

def process_audiogram(image_filepath):
    try:
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("outputs/csv", exist_ok=True)
        os.makedirs("outputs/images", exist_ok=True)

        # A name for the temporarily cropped file that YOLO will read
        temp_crop_path = "outputs/temp_cropped_audiogram.png"
        # Output paths for results
        final_annotated_img_path = "outputs/annotated_symbols_audiogram.jpg"
        csv_output_path = "outputs/csv/" + image_filepath.split("/")[-1].rsplit(".", 1)[0] + ".csv"

        # 1. Run OpenCV crop and grid detection
        img = cv2.imread(image_filepath)
        # Modified refine_crop to return the image object
        cropped_img = refine_crop(img)

        # Modified mark_lines to take an image object
        final_x, final_y = mark_lines(cropped_img)

        if len(final_x) < 2 or len(final_y) < 2:
            raise ValueError("Could not detect enough grid lines.")

        final_x = final_x[1:-1]
        final_y = final_y[1:-1]
        px_to_hz = dict(zip(map(int, final_x), map(int, DEFAULT_FREQUENCIES)))
        y_max = final_y[-1]

        # print("\nX-axis mapping (pixels → frequency):")
        # for px, freq in zip(final_x, DEFAULT_FREQUENCIES):
        #     print(f"  {px:4d} px → {freq:5d} Hz")

        # Save the cropped image to a temporary file so YOLO can read it
        cv2.imwrite(temp_crop_path, cropped_img)

        # 2. Run YOLO inference on the newly created temporary file
        results = yolo_infer(
            model_path=model_path,
            img=temp_crop_path,
            save_name=image_filepath.split("/")[-1]
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
                ratio = y_center / y_max
                db_level = -20 + ratio * 140
                db_level = round(db_level)
                rows.append(
                    {"cls": CLASS_NAMES[cls], "hz": nearest_hz, "db": db_level, "conf": conf}
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

        wide = df.pivot_table(
            index="cls",
            columns="hz",
            values="db",
            aggfunc="mean"
        )
        wide = wide.reindex(columns=DEFAULT_FREQUENCIES).sort_index()

        # Save CSV file
        wide.to_csv(csv_output_path, index_label="Hz")
        display_df = wide.reset_index().fillna("")

        # Gradio expects: Image file path, Dataframe object, File file path (for download)
        # We are now returning the explicitly annotated image path.
        return final_annotated_img_path, display_df, csv_output_path

    except Exception as e:
        return None, pd.DataFrame({"Error": [str(e)]}), None


if __name__ == "__main__":
    # process_audiogram(input_image_path)
    
    folder = Path("images")
    for f in sorted(folder.iterdir()):
        if f.is_file():
            process_audiogram("images/" + f.name)