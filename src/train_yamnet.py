"""
Trains the small classifier head on top of frozen YAMNet embeddings.

Usage:
    python src/train_yamnet.py --epochs 40
"""
import argparse
import json
import os

import tensorflow as tf

from config import MODELS_DIR, LEARNING_RATE, EPOCHS, BATCH_SIZE
from dataset_yamnet import get_datasets
from model_yamnet import build_classifier_head, compile_model


def train(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    os.makedirs(MODELS_DIR, exist_ok=True)

    data = get_datasets(batch_size=batch_size)
    label_map = data["label_map"]
    num_classes = len(label_map)

    model = build_classifier_head(embedding_dim=data["embedding_dim"], num_classes=num_classes)
    compile_model(model, learning_rate=lr)
    model.summary()

    ckpt_path = os.path.join(MODELS_DIR, "yamnet_best_model.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    model.fit(
        data["train_ds"],
        validation_data=data["val_ds"],
        epochs=epochs,
        class_weight=data["class_weights"],
        callbacks=callbacks,
    )

    with open(os.path.join(MODELS_DIR, "yamnet_label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    test_loss, test_acc = model.evaluate(data["test_ds"])
    print(f"\nTest accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")
    print(f"Best checkpoint: {ckpt_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
