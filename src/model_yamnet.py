"""
Small classifier head trained on top of frozen YAMNet embeddings.
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def build_classifier_head(embedding_dim=1024, num_classes=4, dropout=0.3):
    inputs = layers.Input(shape=(embedding_dim,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="yamnet_classifier_head")


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_classifier_head(embedding_dim=1024, num_classes=4)
    compile_model(m)
    m.summary()
