from pathlib import Path


def strip_prefix(folder):
    p = Path(folder)
    used_names = set()

    for f in p.iterdir():
        if not f.is_file():
            continue

        stem = f.stem
        suffix = f.suffix

        last_dash = stem.rfind("-")
        last_underscore = stem.rfind("_")
        last_sep = max(last_dash, last_underscore)

        if last_sep == -1:
            continue

        new_name = stem[last_sep + 1:] + suffix
        target = f.with_name(new_name)

        if new_name in used_names or target.exists():
            print(f"Skipping {f.name} -> {new_name}")
            continue

        f.rename(target)
        used_names.add(new_name)
        print(f"{f.name} -> {new_name}")


if __name__ == "__main__":
    strip_prefix("data/images/")
    strip_prefix("data/labels/")
