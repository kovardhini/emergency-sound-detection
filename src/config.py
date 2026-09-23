"""
Central configuration for the Emergency Sound Detection project.
"""
import os

# ---------- Paths ----------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, "features.npy")
LABELS_PATH = os.path.join(PROCESSED_DATA_DIR, "labels.npy")
LABEL_MAP_PATH = os.path.join(PROCESSED_DATA_DIR, "label_map.json")

# ---------- Classes ----------
CLASSES = [
    "background",
    "siren",
    "glass_breaking",
    "explosion",
]

EMERGENCY_CLASSES = {"siren", "glass_breaking", "explosion"}

# ---------- Audio ----------
SAMPLE_RATE = 22050
CLIP_DURATION = 4.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 40

# ---------- Training ----------
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ---------- Real-time inference ----------
STREAM_WINDOW_SECONDS = CLIP_DURATION
STREAM_HOP_SECONDS = 1.0
CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW = 3
