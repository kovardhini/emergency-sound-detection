"""
Trains the CNN on cached mel-spectrogram features and saves the best
checkpoint. Optionally exports a TFLite version for edge deployment.

Usage:
    python src/train.py --epochs 40 --batch-size 32
    python src/train.py --epochs 40 --export-tflite
"""
import argparse
import json
import os

import tensorflow as tf

from config import MODELS_DIR, LOGS_DIR, LEARNING_RATE, EPOCHS, BATCH_SIZE
from dataset import get_datasets
from model import build_cnn, compile_model


def train(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, export_tflite=False):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    data = get_datasets(batch_size=batch_size)
    label_map = data["label_map"]
    num_classes = len(label_map)

    model = build_cnn(input_shape=data["input_shape"], num_classes=num_classes)
    compile_model(model, learning_rate=lr)
    model.summary()

    ckpt_path = os.path.join(MODELS_DIR, "best_model.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        tf.keras.callbacks.TensorBoard(log_dir=LOGS_DIR),
    ]

    history = model.fit(
        data["train_ds"],
        validation_data=data["val_ds"],
        epochs=epochs,
        class_weight=data["class_weights"],
        callbacks=callbacks,
    )

    # Save final model + label map together so inference scripts are self-contained.
    final_path = os.path.join(MODELS_DIR, "final_model.h5")
    model.save(final_path)
    with open(os.path.join(MODELS_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nBest checkpoint: {ckpt_path}")
    print(f"Final model:     {final_path}")

    # Quick test-set report
    test_loss, test_acc = model.evaluate(data["test_ds"])
    print(f"\nTest accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    if export_tflite:
        export_to_tflite(model)

    return model, history


def export_to_tflite(model, out_path=None):
    out_path = out_path or os.path.join(MODELS_DIR, "model.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Exported TFLite model ({size_kb:.1f} KB) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--export-tflite", action="store_true")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, export_tflite=args.export_tflite)
