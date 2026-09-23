"""
Loads cached YAMNet embeddings, splits into train/val/test BY GROUP so
augmented copies never leak across the split boundary.
"""
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

from config import PROCESSED_DATA_DIR, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED, BATCH_SIZE

FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, "yamnet_features.npy")
LABELS_PATH = os.path.join(PROCESSED_DATA_DIR, "yamnet_labels.npy")
GROUPS_PATH = os.path.join(PROCESSED_DATA_DIR, "yamnet_groups.npy")
LABEL_MAP_PATH = os.path.join(PROCESSED_DATA_DIR, "yamnet_label_map.json")


def load_arrays():
    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH)
    g = np.load(GROUPS_PATH)
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    return X, y, g, label_map


def _group_split(X, y, groups, test_size):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return train_idx, test_idx


def split_data(X, y, groups):
    train_idx, test_idx = _group_split(X, y, groups, test_size=TEST_SPLIT)
    X_train, y_train, g_train = X[train_idx], y[train_idx], groups[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    val_fraction_of_remainder = VAL_SPLIT / (1 - TEST_SPLIT)
    train_idx2, val_idx2 = _group_split(X_train, y_train, g_train, test_size=val_fraction_of_remainder)
    X_val, y_val = X_train[val_idx2], y_train[val_idx2]
    X_train, y_train = X_train[train_idx2], y_train[train_idx2]

    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes)
    total = counts.sum()
    return {i: total / (num_classes * max(c, 1)) for i, c in enumerate(counts)}


def make_tf_dataset(X, y, batch_size=BATCH_SIZE, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=RANDOM_SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def get_datasets(batch_size=BATCH_SIZE):
    X, y, groups, label_map = load_arrays()
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, groups)

    train_ds = make_tf_dataset(X_train, y_train, batch_size, shuffle=True)
    val_ds = make_tf_dataset(X_val, y_val, batch_size, shuffle=False)
    test_ds = make_tf_dataset(X_test, y_test, batch_size, shuffle=False)

    class_weights = compute_class_weights(y_train, len(label_map))

    return {
        "train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
        "X_test": X_test, "y_test": y_test,
        "label_map": label_map, "class_weights": class_weights,
        "embedding_dim": X.shape[1],
    }
