from pathlib import Path
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

DAGHAR_DIR = (
    DATA_DIR
    / "processed"
    / "daghar"
    / "standardized_view"
    / "UCI"
)

TRAIN_FILE = DAGHAR_DIR / "train.csv"
VAL_FILE = DAGHAR_DIR / "validation.csv"
TEST_FILE = DAGHAR_DIR / "test.csv"


# ============================================================
# RESULT PATHS
# ============================================================

RESULTS_DIR = PROJECT_ROOT / "results"

METRICS_DIR = RESULTS_DIR / "metrics"
CONFUSION_MATRIX_DIR = RESULTS_DIR / "confusion_matrices"
MODEL_DIR = RESULTS_DIR / "models"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
CONFUSION_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

SENSOR_CHANNELS = [
    "accel-x",
    "accel-y",
    "accel-z",
    "gyro-x",
    "gyro-y",
    "gyro-z",
]

NUM_CHANNELS = 6

WINDOW_SIZE = 60

LABEL_COLUMN = "standard activity code"


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50

RANDOM_SEED = 42

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)