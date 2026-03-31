from pathlib import Path

def strip_prefix_before_dash(folder):
    p = Path(folder)
    for f in p.iterdir():
        if not f.is_file(): 
            continue
        name = f.name
        if "-" not in name:
            continue
        new_name = name.rsplit("-", 1)[1]
        f.rename(f.with_name(new_name))

strip_prefix_before_dash("data/images/")
strip_prefix_before_dash("data/labels/")
