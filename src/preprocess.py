"""
Turns raw .wav clips in data/raw/<class>/ into a fixed-size mel-spectrogram
feature array, cached to disk as .npy for fast training iteration.

Usage:
    python src/preprocess.py
    python src/preprocess.py --augment      # adds pitch/noise/time-shift augmentation
"""
import argparse
import json
import os

import librosa
import numpy as np
from tqdm import tqdm

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_PATH, LABELS_PATH,
    LABEL_MAP_PATH, CLASSES, SAMPLE_RATE, CLIP_DURATION, N_MELS, N_FFT,
    HOP_LENGTH,
)

GROUPS_PATH = os.path.join(PROCESSED_DATA_DIR, "groups.npy")

TARGET_LEN = int(SAMPLE_RATE * CLIP_DURATION)


def load_and_fix_length(path, sr=SAMPLE_RATE, target_len=TARGET_LEN):
    """Load audio, resample, and pad/trim to a fixed number of samples."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def augment_waveform(y, sr=SAMPLE_RATE):
    """Light augmentation: random pitch shift, time shift, and noise."""
    out = y.copy()
    shift = np.random.randint(-sr // 4, sr // 4)
    out = np.roll(out, shift)
    if np.random.rand() < 0.5:
        n_steps = np.random.uniform(-2, 2)
        out = librosa.effects.pitch_shift(out, sr=sr, n_steps=n_steps)
    if np.random.rand() < 0.5:
        noise_amp = 0.005 * np.random.uniform() * np.max(np.abs(out))
        out = out + noise_amp * np.random.normal(size=out.shape[0])
    return out.astype(np.float32)


def extract_mel_spectrogram(y, sr=SAMPLE_RATE):
    """Log-mel spectrogram, normalized to zero mean / unit variance."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel.astype(np.float32)


def build_dataset(augment=False, augment_factor=2, max_per_class=None):
    """
    max_per_class: if set, randomly subsamples each class's source files
    down to this many BEFORE augmentation. Use this to balance a dataset
    where one class (e.g. "background") has far more raw clips than the
    others — otherwise the model just learns to predict the majority class.
    """
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
                y = load_and_fix_length(fpath)
            except Exception as e:
                print(f"  [skip] {fname}: {e}")
                continue

            features.append(extract_mel_spectrogram(y))
            labels.append(label_map[class_name])
            groups.append(group_id)

            if augment:
                for _ in range(augment_factor):
                    y_aug = augment_waveform(y)
                    features.append(extract_mel_spectrogram(y_aug))
                    labels.append(label_map[class_name])
                    groups.append(group_id)

            group_id += 1

    if not features:
        raise RuntimeError(
            "No audio files found under data/raw/<class>/. "
            "Populate it first (see src/download_data.py or README)."
        )

    X = np.stack(features)
    X = X[..., np.newaxis]
    y = np.array(labels, dtype=np.int64)
    g = np.array(groups, dtype=np.int64)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    np.save(FEATURES_PATH, X)
    np.save(LABELS_PATH, y)
    np.save(GROUPS_PATH, g)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nSaved features: {X.shape} -> {FEATURES_PATH}")
    print(f"Saved labels:   {y.shape} -> {LABELS_PATH}")
    print(f"Saved groups:   {g.shape} -> {GROUPS_PATH} ({group_id} unique source clips)")
    print(f"Label map: {label_map}")
    return X, y, g


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", action="store_true", help="Apply waveform augmentation")
    parser.add_argument("--augment-factor", type=int, default=2, help="Augmented copies per original clip")
    parser.add_argument("--max-per-class", type=int, default=None,
                         help="Cap the number of raw source files used per class, to fix imbalance")
    args = parser.parse_args()
    build_dataset(augment=args.augment, augment_factor=args.augment_factor, max_per_class=args.max_per_class)
