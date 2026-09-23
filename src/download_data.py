"""
Helper for organizing raw datasets into the folder structure this project
expects:

    data/raw/<class_name>/*.wav

This script does NOT auto-download UrbanSound8K or ESC-50 (both require
accepting a license / manual download). Instead it helps you *map* the
files you've already downloaded into the right structure.

--------------------------------------------------------------------------
Suggested sources
--------------------------------------------------------------------------
1. UrbanSound8K   (siren, gun_shot, car_horn, engine_idling, street_music,
                    dog_bark, drilling, air_conditioner, children_playing,
                    jackhammer)
   -> https://urbansounddataset.weebly.com/urbansound8k.html
   Comes with a metadata CSV (UrbanSound8K.csv) mapping file -> class.

2. ESC-50         (glass_breaking, siren, screaming/human sounds,
                    explosion via "fireworks"/"gun_shot" adjacent classes)
   -> https://github.com/karolpiczak/ESC-50
   Comes with meta/esc50.csv mapping filename -> category.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
1. Download & unzip UrbanSound8K and ESC-50 somewhere on disk.
2. Edit the CLASS_MAPPING dict below so source category names map to your
   target CLASSES in config.py.
3. Run:
       python src/download_data.py \
           --urbansound8k /path/to/UrbanSound8K \
           --esc50 /path/to/ESC-50-master

This copies (or symlinks) matching files into data/raw/<class>/.
"""
import argparse
import csv
import os
import shutil

from config import RAW_DATA_DIR, CLASSES

# Map SOURCE dataset category -> TARGET class in this project.
# Anything not listed here is ignored (or dumped into "background" if you add it).
URBANSOUND8K_MAPPING = {
    "siren": "siren",
    "gun_shot": "gunshot",
    "car_horn": "background",
    "engine_idling": "background",
    "air_conditioner": "background",
    "street_music": "background",
    "children_playing": "background",
    "dog_bark": "background",
    "drilling": "background",
    "jackhammer": "background",
}

ESC50_MAPPING = {
    "siren": "siren",
    "glass_breaking": "glass_breaking",
    "screaming": "scream",
    "gun_shot": "gunshot",
    "fireworks": "explosion",
    "car_horn": "background",
    "engine": "background",
    "clapping": "background",
    "footsteps": "background",
    "rain": "background",
    "wind": "background",
    "crying_baby": "background",
}


def ensure_dirs():
    for c in CLASSES:
        os.makedirs(os.path.join(RAW_DATA_DIR, c), exist_ok=True)


def copy_file(src, dst_class, filename, symlink=False):
    dst_dir = os.path.join(RAW_DATA_DIR, dst_class)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, filename)
    if os.path.exists(dst):
        return
    if symlink:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def process_urbansound8k(root, symlink=False):
    csv_path = os.path.join(root, "metadata", "UrbanSound8K.csv")
    if not os.path.exists(csv_path):
        print(f"[UrbanSound8K] metadata CSV not found at {csv_path}, skipping.")
        return
    count = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["class"]
            target = URBANSOUND8K_MAPPING.get(category)
            if target is None or target not in CLASSES:
                continue
            fold = f"fold{row['fold']}"
            src_path = os.path.join(root, "audio", fold, row["slice_file_name"])
            if not os.path.exists(src_path):
                continue
            copy_file(src_path, target, f"us8k_{row['slice_file_name']}", symlink)
            count += 1
    print(f"[UrbanSound8K] organized {count} files.")


def process_esc50(root, symlink=False):
    csv_path = os.path.join(root, "meta", "esc50.csv")
    if not os.path.exists(csv_path):
        print(f"[ESC-50] metadata CSV not found at {csv_path}, skipping.")
        return
    count = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["category"]
            target = ESC50_MAPPING.get(category)
            if target is None or target not in CLASSES:
                continue
            src_path = os.path.join(root, "audio", row["filename"])
            if not os.path.exists(src_path):
                continue
            copy_file(src_path, target, f"esc50_{row['filename']}", symlink)
            count += 1
    print(f"[ESC-50] organized {count} files.")


def summarize():
    print("\nClass distribution in data/raw/:")
    for c in CLASSES:
        d = os.path.join(RAW_DATA_DIR, c)
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        print(f"  {c:<18} {n} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urbansound8k", type=str, default=None, help="Path to unzipped UrbanSound8K root")
    parser.add_argument("--esc50", type=str, default=None, help="Path to unzipped ESC-50-master root")
    parser.add_argument("--symlink", action="store_true", help="Symlink instead of copy (saves disk space)")
    args = parser.parse_args()

    ensure_dirs()
    if args.urbansound8k:
        process_urbansound8k(args.urbansound8k, args.symlink)
    if args.esc50:
        process_esc50(args.esc50, args.symlink)
    if not args.urbansound8k and not args.esc50:
        print("No dataset paths given. Pass --urbansound8k and/or --esc50.")
    summarize()
