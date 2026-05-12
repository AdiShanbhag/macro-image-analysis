from pathlib import Path

# ── Root paths

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = BASE_DIR / "outputs"
EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"
MODEL_OUTPUT_DIR = OUTPUTS_DIR / "models"
REPORTS_OUTPUT_DIR = OUTPUTS_DIR / "reports"

# ── Image settings

IMAGE_SIZE = (128, 128)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Model settings

MODEL_FILENAME = "macro_classifier.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 200
CLASS_WEIGHT = "balanced"

# EDA settings

# Number of sample images to include in the visual sample grid
SAMPLE_GRID_COUNT: int = 9

# Random seed used throughout the project for reproducibility
RANDOM_SEED: int = 42
