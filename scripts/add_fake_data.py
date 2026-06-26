import os
import pandas as pd
from .utils import add_fake_data

IMAGE_PATH = "data/raw/"
OUTPUT_PATH = "outputs/ocr/"
GT_CSV = "outputs/ocr/fake_ground_truth.csv"


def main():
    templates = [f for f in os.listdir(IMAGE_PATH) if f.lower().endswith(".jpg")]

    rows = []

    for template_file in templates[:]:
        fake_file_path, data = add_fake_data(template_file, IMAGE_PATH, OUTPUT_PATH)

        data["file_path"] = fake_file_path
        rows.append(data)

    df = pd.DataFrame(rows)
    df.to_csv(GT_CSV, index=False, encoding="utf-8-sig")

    print(f"Saved fake data ground truth to {GT_CSV}")


if __name__ == "__main__":
    main()