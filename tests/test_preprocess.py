"""
Basic unit tests for the preprocessing pipeline. Run with:
    pytest tests/
"""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import load_and_fix_length, extract_mel_spectrogram, TARGET_LEN  # noqa: E402
from config import SAMPLE_RATE, N_MELS  # noqa: E402


def _make_dummy_wav(path, duration_sec=1.0, sr=SAMPLE_RATE):
    import soundfile as sf
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    y = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(path, y, sr)


def test_load_and_fix_length_pads_short_clips(tmp_path):
    wav_path = tmp_path / "short.wav"
    _make_dummy_wav(str(wav_path), duration_sec=1.0)

    y = load_and_fix_length(str(wav_path))
    assert len(y) == TARGET_LEN


def test_load_and_fix_length_trims_long_clips(tmp_path):
    wav_path = tmp_path / "long.wav"
    _make_dummy_wav(str(wav_path), duration_sec=10.0)

    y = load_and_fix_length(str(wav_path))
    assert len(y) == TARGET_LEN


def test_mel_spectrogram_shape():
    y = np.random.randn(TARGET_LEN).astype(np.float32)
    spec = extract_mel_spectrogram(y)
    assert spec.shape[0] == N_MELS
    assert spec.ndim == 2


def test_mel_spectrogram_is_normalized():
    y = np.random.randn(TARGET_LEN).astype(np.float32)
    spec = extract_mel_spectrogram(y)
    assert abs(spec.mean()) < 1.0  # roughly zero-centered after normalization
