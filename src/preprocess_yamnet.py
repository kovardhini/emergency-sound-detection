"""
Builds YAMNet-embedding features from data/raw/<class>/*.wav, for transfer
learning instead of training a CNN from scratch on spectrograms.

Usage:
    python src/preprocess_yamnet.py --augment --augment-factor 4 --max-per-class 40
"""
import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CLASSES
from preprocess import load_and_fix_length, augment_waveform
from yamnet_features import embed_waveform, YAMNET_SR

FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, "yamnet_features.npy")
LABELS_PATH = os.path.join(PROCESSED_DATA_DIR, "yamnet_labels.npy")
GROUPS_PATH = os.path.join(PROCESSED_DATA_DIR, "yamnet_groups.npy")
LABEL_MAP_PATH = os.path.join(PROCESSED_DATA_DIR, "yamnet_label_map.json")

CLIP_DURATION = 4.0
TARGET_LEN = int(YAMNET_SR * CLIP_DURATION)


def build_dataset(augment=False, augment_factor=4, max_per_class=None):
    features, labels, groups = [], [], []
    label_map = {name: idx for idx, name in enumerate(CLASSES)}
    group_id = 0
    rng = np.random.default_rng(42)

    for class_name in CLASSES:
        class_dir = os.path.join(RAW_DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            print(f"[warn] missing folder for class '{class_name}', skipping.")
            continue
        files = [f for f in os.listdir(class_dir) if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))]

        if max_per_class is not None and len(files) > max_per_class:
            files = list(rng.choice(files, size=max_per_class, replace=False))

        print(f"Processing '{class_name}': {len(files)} files")

        for fname in tqdm(files, desc=class_name):
            fpath = os.path.join(class_dir, fname)
            try:
                y = load_and_fix_length(fpath, sr=YAMNET_SR, target_len=TARGET_LEN)
            except Exception as e:
                print(f"  [skip] {fname}: {e}")
                continue

            features.append(embed_waveform(y))
            labels.append(label_map[class_name])
            groups.append(group_id)

            if augment:
                for _ in range(augment_factor):
                    y_aug = augment_waveform(y, sr=YAMNET_SR)
                    features.append(embed_waveform(y_aug))
                    labels.append(label_map[class_name])
                    groups.append(group_id)

            group_id += 1

    if not features:
        raise RuntimeError(
            "No audio files found under data/raw/<class>/. "
            "Populate it first (see src/download_data.py or README)."
        )

    X = np.stack(features)
    y = np.array(labels, dtype=np.int64)
    g = np.array(groups, dtype=np.int64)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    np.save(FEATURES_PATH, X)
    np.save(LABELS_PATH, y)
    np.save(GROUPS_PATH, g)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nSaved YAMNet features: {X.shape} -> {FEATURES_PATH}")
    print(f"Saved labels:          {y.shape} -> {LABELS_PATH}")
    print(f"Saved groups:          {g.shape} -> {GROUPS_PATH} ({group_id} unique source clips)")
    print(f"Label map: {label_map}")
    return X, y, g


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment-factor", type=int, default=4)
    parser.add_argument("--max-per-class", type=int, default=None)
    args = parser.parse_args()
    build_dataset(augment=args.augment, augment_factor=args.augment_factor, max_per_class=args.max_per_class)
