"""
Wrapper around Google's YAMNet — a model pretrained on AudioSet (~2 million
YouTube clips across 521 sound categories, including several siren,
explosion, and glass-breaking related classes). We use it purely as a
frozen feature extractor: run audio through it, take the 1024-dim
embedding, and train a small classifier head on top for our 4 classes.

This is the standard fix for "I don't have enough labeled audio" — YAMNet
already learned what matters in a sound from millions of examples, so our
tiny dataset only needs to learn how to map that existing understanding to
our specific categories, instead of learning acoustic features from zero.
"""
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

YAMNET_SR = 16000  # YAMNet requires exactly 16kHz mono input
_yamnet_model = None


def get_yamnet():
    """Loads YAMNet once and caches it in memory for the process lifetime."""
    global _yamnet_model
    if _yamnet_model is None:
        print("Loading YAMNet from TensorFlow Hub (downloads once, then caches locally)...")
        _yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    return _yamnet_model


def embed_waveform(y, sr=YAMNET_SR):
    """
    y: 1D float32 waveform at 16kHz, values roughly in [-1, 1].
    Returns a single 1024-dim embedding: YAMNet internally chunks the clip
    into ~0.96s frames (0.48s hop) and returns one embedding per frame; we
    average across frames to get one fixed-size vector per clip.
    """
    if sr != YAMNET_SR:
        raise ValueError(f"YAMNet requires {YAMNET_SR}Hz audio, got {sr}Hz")
    model = get_yamnet()
    _scores, embeddings, _spectrogram = model(y)
    return tf.reduce_mean(embeddings, axis=0).numpy()  # (1024,)
