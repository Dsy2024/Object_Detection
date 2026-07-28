from pathlib import Path

import pandas as pd

from .utils import add_fake_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "ocr"
GT_CSV = OUTPUT_PATH / "fake_ground_truth.csv"
RESULT_COLUMNS = ["path", "doctor", "patient", "serial"]


def load_results():
    """Load results from the previous run so they remain visible after restart."""
    if not GT_CSV.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    try:
        df = pd.read_csv(GT_CSV).fillna("")
        if "path" in df.columns:
            df["path"] = df["path"].map(
                lambda value: str(
                    (PROJECT_ROOT / value).resolve()
                    if value and not Path(str(value)).is_absolute()
                    else value
                )
            )
        return df
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=RESULT_COLUMNS)


def generate_fake_data():
    """Generate images and return their metadata for both CLI and web usage."""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    templates = sorted(
        path for path in IMAGE_PATH.iterdir() if path.suffix.lower() in {".jpg", ".jpeg"}
    ) if IMAGE_PATH.exists() else []

    rows = []
    for template in templates:
        _, data = add_fake_data(template.name, str(IMAGE_PATH), str(OUTPUT_PATH))
        data["path"] = str(Path(data["path"]).resolve())
        rows.append(data)

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    df.to_csv(GT_CSV, index=False, encoding="utf-8-sig")
    return df


def web_results(generate=False):
    df = generate_fake_data() if generate else load_results()
    gallery = [
        (path, f"{row.patient}｜{row.serial}")
        for row in df.itertuples(index=False)
        if (path := str(row.path)) and Path(path).exists()
    ]
    action = "本次已生成" if generate else "已載入上次"
    status = (
        f"{action} {len(df)} 筆結果。\n\n"
        f"圖片資料夾：{OUTPUT_PATH}\n\n"
        f"結果 CSV：{GT_CSV}"
    )
    return gallery, df, status


def main():
    df = generate_fake_data()
    print(f"Generated {len(df)} image(s)")
    print(f"Images saved to: {OUTPUT_PATH}")
    print(f"Ground truth saved to: {GT_CSV}")


if __name__ == "__main__":
    main()
