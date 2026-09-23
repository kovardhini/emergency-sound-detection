"""
Evaluates the YAMNet-based classifier: classification report + confusion matrix.

Usage:
    python src/evaluate_yamnet.py --model models/yamnet_best_model.h5
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import MODELS_DIR
from dataset_yamnet import get_datasets


def evaluate(model_path):
    data = get_datasets()
    label_map = data["label_map"]
    idx_to_label = {v: k for k, v in label_map.items()}
    class_names = [idx_to_label[i] for i in range(len(label_map))]

    model = tf.keras.models.load_model(model_path)
    y_true = data["y_test"]
    y_pred = np.argmax(model.predict(data["X_test"]), axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)

    report_path = os.path.join(MODELS_DIR, "yamnet_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report -> {report_path}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — YAMNet Transfer Learning")
    plt.tight_layout()
    cm_path = os.path.join(MODELS_DIR, "yamnet_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Saved confusion matrix -> {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.join(MODELS_DIR, "yamnet_best_model.h5"))
    args = parser.parse_args()
    evaluate(args.model)
