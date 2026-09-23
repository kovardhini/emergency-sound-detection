"""
Live microphone streaming inference. Maintains a rolling audio buffer,
runs the model on a sliding window every STREAM_HOP_SECONDS, and prints
an ALERT when an emergency class is detected with high confidence.

Usage:
    python src/infer_realtime.py --model models/best_model.h5
"""
import argparse
import collections
import json
import os
import time

import numpy as np
import sounddevice as sd
import tensorflow as tf

from config import (
    MODELS_DIR, SAMPLE_RATE, STREAM_WINDOW_SECONDS, STREAM_HOP_SECONDS,
    CONFIDENCE_THRESHOLD, SMOOTHING_WINDOW, EMERGENCY_CLASSES, N_MELS,
    N_FFT, HOP_LENGTH,
)
from preprocess import extract_mel_spectrogram

WINDOW_SAMPLES = int(SAMPLE_RATE * STREAM_WINDOW_SECONDS)
HOP_SAMPLES = int(SAMPLE_RATE * STREAM_HOP_SECONDS)


class RollingBuffer:
    """Fixed-size ring buffer of the most recent audio samples."""
    def __init__(self, size):
        self.size = size
        self.buf = np.zeros(size, dtype=np.float32)

    def push(self, chunk):
        n = len(chunk)
        if n >= self.size:
            self.buf = chunk[-self.size:]
        else:
            self.buf = np.concatenate([self.buf[n:], chunk])

    def get(self):
        return self.buf.copy()


def load_label_map(model_dir):
    path = os.path.join(model_dir, "label_map.json")
    with open(path) as f:
        label_map = json.load(f)
    return {v: k for k, v in label_map.items()}


def run(model_path, device=None):
    model_dir = os.path.dirname(model_path) or MODELS_DIR
    idx_to_label = load_label_map(model_dir)
    model = tf.keras.models.load_model(model_path)

    buffer = RollingBuffer(WINDOW_SAMPLES)
    recent_preds = collections.deque(maxlen=SMOOTHING_WINDOW)

    print("🎙️  Listening... (Ctrl+C to stop)\n")

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status)
        buffer.push(indata[:, 0])

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, callback=audio_callback,
        blocksize=HOP_SAMPLES, device=device,
    ):
        try:
            while True:
                time.sleep(STREAM_HOP_SECONDS)
                window = buffer.get()

                # Skip near-silent windows — saves compute and avoids noise on "background"
                if np.abs(window).mean() < 1e-4:
                    continue

                spec = extract_mel_spectrogram(window)
                x = spec[np.newaxis, ..., np.newaxis]  # (1, N_MELS, T, 1)

                probs = model.predict(x, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                label = idx_to_label[pred_idx]

                recent_preds.append(label)
                # Majority vote over the last few predictions to reduce flicker
                smoothed_label = max(set(recent_preds), key=list(recent_preds).count)

                timestamp = time.strftime("%H:%M:%S")
                is_emergency = smoothed_label in EMERGENCY_CLASSES and confidence >= CONFIDENCE_THRESHOLD
                tag = "🚨 ALERT" if is_emergency else ""

                print(f"[{timestamp}] {smoothed_label:<16} {confidence*100:5.1f}%  {tag}")

        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.join(MODELS_DIR, "best_model.h5"))
    parser.add_argument("--device", type=int, default=None, help="Input device index (see `sd.query_devices()`)")
    args = parser.parse_args()
    run(args.model, device=args.device)
