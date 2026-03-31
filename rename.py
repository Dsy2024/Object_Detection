from pathlib import Path
import re, uuid

def rename_jpgs(folder, dry_run=False):
    """
    folder   : target folder
    dry_run  : True to only print the intended renaming without actually performing it
    """
    folder = Path(folder)

    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}]
    def natkey(p: Path):
        return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", p.name)]
    files.sort(key=natkey)

    if not files:
        print("No JPG files found in the specified folder.")
        return

    temps = []
    for p in files:
        tmp = p.with_name(f"{p.name}.tmp-{uuid.uuid4().hex}")
        if dry_run:
            print(f"[DRY] {p.name} -> {tmp.name}")
        else:
            p.rename(tmp)
        temps.append(tmp)

    for i, tmp in enumerate(temps, start=1):
        target = tmp.with_name(f"{i}.jpg")
        if dry_run:
            print(f"[DRY] {tmp.name} -> {target.name}")
        else:
            tmp.rename(target)

    print(f"Renamed {len(files)} files in '{folder}'.")

rename_jpgs("temp/")