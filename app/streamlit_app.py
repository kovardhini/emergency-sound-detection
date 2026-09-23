"""
Streamlit demo for Emergency Sound Detection.

Two modes:
  1. Upload an audio clip -> get a prediction + confidence chart
  2. Live microphone (works when run locally; disabled on most cloud hosts
     because they don't expose an audio input device)

Run:
    streamlit run app/streamlit_app.py
"""
import json
import os
import sys

import numpy as np
import streamlit as st
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from config import MODELS_DIR, EMERGENCY_CLASSES  # noqa: E402
from preprocess import load_and_fix_length, extract_mel_spectrogram  # noqa: E402

st.set_page_config(page_title="Emergency Sound Detection", page_icon="🚨", layout="centered")


@st.cache_resource
def load_model_and_labels(model_path):
    model = tf.keras.models.load_model(model_path)
    label_map_path = os.path.join(os.path.dirname(model_path), "label_map.json")
    with open(label_map_path) as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}
    return model, idx_to_label


def predict(model, idx_to_label, wav_path):
    y = load_and_fix_length(wav_path)
    spec = extract_mel_spectrogram(y)
    x = spec[np.newaxis, ..., np.newaxis]
    probs = model.predict(x, verbose=0)[0]
    return probs, idx_to_label


st.title("🚨 Emergency Sound Detection")
st.caption("Upload an audio clip to classify it as an emergency sound or background noise.")

model_path = os.path.join(MODELS_DIR, "best_model.h5")

if not os.path.exists(model_path):
    st.warning(
        "No trained model found at `models/best_model.h5`. "
        "Train one first with `python src/train.py`, or point this app "
        "at a different path below."
    )
    model_path = st.text_input("Model path", value=model_path)

if os.path.exists(model_path):
    model, idx_to_label = load_model_and_labels(model_path)

    uploaded = st.file_uploader("Upload a .wav / .mp3 clip", type=["wav", "mp3", "flac", "ogg"])

    if uploaded is not None:
        tmp_path = os.path.join("/tmp", uploaded.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.audio(uploaded)

        with st.spinner("Analyzing..."):
            probs, idx_to_label = predict(model, idx_to_label, tmp_path)

        pred_idx = int(np.argmax(probs))
        label = idx_to_label[pred_idx]
        confidence = float(probs[pred_idx])

        if label in EMERGENCY_CLASSES:
            st.error(f"🚨 **{label.upper()}** detected — {confidence*100:.1f}% confidence")
        else:
            st.success(f"✅ **{label}** — {confidence*100:.1f}% confidence")

        st.subheader("Confidence by class")
        chart_data = {idx_to_label[i]: float(p) for i, p in enumerate(probs)}
        st.bar_chart(chart_data)
else:
    st.stop()

st.divider()
st.markdown(
    "**Real-time microphone mode** is available when running locally: \n"
    "```bash\npython src/infer_realtime.py --model models/best_model.h5\n```"
)
