# 🚨 Emergency Sound Detection

Real-time audio classification system that detects emergency sounds — **sirens, gunshots, glass breaking, screaming, explosions, car crashes** — from a live microphone feed or audio files. Built to demonstrate the full ML lifecycle: signal processing → deep learning → real-time inference → deployment.

## Why this project

- **Audio DSP**: MFCCs, mel-spectrograms, waveform augmentation
- **Deep learning**: CNN trained on spectrogram images (Keras/TensorFlow)
- **Transfer learning**: fine-tuning a classifier head on top of frozen YAMNet embeddings
- **Real-time systems**: streaming inference on live mic input with a rolling buffer
- **Deployment**: Streamlit web demo + exportable TFLite model for edge devices
- **MLOps basics**: reproducible pipeline, config-driven training, saved artifacts, tests

---

## Demo
cat > README.md << 'MDEOF'
# 🚨 Emergency Sound Detection

Real-time audio classification system that detects emergency sounds — **sirens, gunshots, glass breaking, screaming, explosions, car crashes** — from a live microphone feed or audio files. Built to demonstrate the full ML lifecycle: signal processing → deep learning → real-time inference → deployment.

## Why this project

- **Audio DSP**: MFCCs, mel-spectrograms, waveform augmentation
- **Deep learning**: CNN trained on spectrogram images (Keras/TensorFlow)
- **Transfer learning**: fine-tuning a classifier head on top of frozen YAMNet embeddings
- **Real-time systems**: streaming inference on live mic input with a rolling buffer
- **Deployment**: Streamlit web demo + exportable TFLite model for edge devices
- **MLOps basics**: reproducible pipeline, config-driven training, saved artifacts, tests

---

## Demo
cat > README.md << 'MDEOF'
# Emergency Sound Detection

Real-time audio classification system that detects emergency sounds — sirens, gunshots, glass breaking, screaming, explosions, car crashes — from a live microphone feed or audio files. Built to demonstrate the full ML lifecycle: signal processing, deep learning, real-time inference, deployment.

## Why this project

- Audio DSP: MFCCs, mel-spectrograms, waveform augmentation
- Deep learning: CNN trained on spectrogram images (Keras/TensorFlow)
- Transfer learning: fine-tuning a classifier head on top of frozen YAMNet embeddings
- Real-time systems: streaming inference on live mic input with a rolling buffer
- Deployment: Streamlit web demo
- MLOps basics: reproducible pipeline, config-driven training, saved artifacts, tests

## Project Structure

## Results

Two approaches were built and compared on the same leakage-free, group-based train/val/test split (4 classes: background, siren, glass_breaking, explosion).

### Approach 1: CNN trained from scratch on mel-spectrograms
Struggled with the small dataset (about 40 clips per class from ESC-50). The siren class collapsed entirely (0% recall) because the model didn't have enough examples to learn its acoustic pattern.

| Class          | Precision | Recall | F1   |
|----------------|-----------|--------|------|
| Background     | 0.615     | 0.178  | 0.276|
| Siren          | 0.000     | 0.000  | 0.000|
| Glass Breaking | 0.433     | 0.743  | 0.547|
| Explosion      | 0.319     | 0.750  | 0.448|
| Overall Acc    | 0.408     |        |      |

### Approach 2: Transfer learning with YAMNet (recommended)
Switched to using YAMNet, pretrained on over 2 million YouTube clips across 521 sound classes, as a frozen feature extractor, with only a small classifier head trained on our data.

| Class          | Precision | Recall | F1   |
|----------------|-----------|--------|------|
| Background     | 1.000     | 0.867  | 0.929|
| Siren          | 1.000     | 1.000  | 1.000|
| Glass Breaking | 0.972     | 1.000  | 0.986|
| Explosion      | 0.800     | 1.000  | 0.889|
| Overall Acc    | 0.950     |        |      |

Takeaway: with a small labeled dataset, transfer learning from a model pretrained on a large general-purpose audio corpus substantially outperforms training a CNN from scratch. Accuracy went from 41% to 95%, and the siren class went from completely unrecognized to perfect recall.

## Tech Stack
Python, TensorFlow/Keras, TensorFlow Hub (YAMNet), librosa, sounddevice, scikit-learn, Streamlit, NumPy/Pandas

## License
MIT — see LICENSE
