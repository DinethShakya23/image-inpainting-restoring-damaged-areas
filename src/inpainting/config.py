"""Project-wide configuration and default hyperparameters."""
from __future__ import annotations

from pathlib import Path

# Repository layout ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_MODEL_PATH = MODELS_DIR / "inpainting_model.h5"

# Image / model defaults -----------------------------------------------------
IMG_SIZE = 128          # images are resized to IMG_SIZE x IMG_SIZE
CHANNELS = 3            # RGB
MASK_SIZE = 32          # side length of the square "damage" mask in pixels

# Training defaults ----------------------------------------------------------
EPOCHS = 10
BATCH_SIZE = 16
TEST_SPLIT = 0.2
RANDOM_SEED = 42
