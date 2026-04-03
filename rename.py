from pathlib import Path
import re
import uuid


def rename_files(folder, dry_run=False, exts=(".jpg", ".jpeg", ".txt")):
    """
    folder   : target folder
    dry_run  : True -> 只預覽，不真的改名
    exts     : 要處理的副檔名
    """
    folder = Path(folder)
    exts = {e.lower() for e in exts}

    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]

    def natkey(p: Path):
        return [
            int(t) if t.isdigit() else t.casefold()
            for t in re.split(r"(\d+)", p.name)
        ]

    files.sort(key=natkey)

    if not files:
        print("No matching files found in the specified folder.")
        return

    temps = []
    for p in files:
        tmp = p.with_name(f"{p.name}.tmp-{uuid.uuid4().hex}")
        if dry_run:
            print(f"[DRY] {p.name} -> {tmp.name}")
        else:
            p.rename(tmp)
        temps.append((tmp, p.suffix.lower()))

    for i, (tmp, suffix) in enumerate(temps, start=1):
        target = tmp.with_name(f"{i}{suffix}")
        if dry_run:
            print(f"[DRY] {tmp.name} -> {target.name}")
        else:
            tmp.rename(target)

    print(f"Renamed {len(files)} files in '{folder}'.")


if __name__ == "__main__":
    # rename_files("data/images/")
    # rename_files("data/labels/")
    rename_files("images/")