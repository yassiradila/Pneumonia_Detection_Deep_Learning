"""Global configuration and hyperparameters for Pneumonia Detection."""

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """Project-wide hyperparameters and configuration settings."""

    # Image specifications
    IMG_HEIGHT: int = 299
    IMG_WIDTH: int = 299
    CHANNELS: int = 3

    @property
    def target_size(self) -> Tuple[int, int]:
        return (self.IMG_HEIGHT, self.IMG_WIDTH)

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNELS)

    # Class mappings
    CLASS_NAMES = ("NORMAL", "PNEUMONIA")
    NUM_CLASSES: int = 1  # Binary classification (Sigmoid)
    POSITIVE_CLASS: str = "PNEUMONIA"
    DECISION_THRESHOLD: float = 0.5

    # Training parameters
    BATCH_SIZE: int = 32
    DENSE_UNITS: int = 1024
    DROPOUT_RATE: float = 0.2

    # Phase 1: Feature Extraction (Frozen Base)
    EPOCHS_PHASE1: int = 20
    LR_PHASE1: float = 1e-3

    # Phase 2: Fine-Tuning (Unfreezing top layers)
    EPOCHS_PHASE2: int = 10
    LR_PHASE2: float = 1e-5
    UNFREEZE_TOP_LAYERS: int = 100

    # Callbacks
    PATIENCE_EARLY_STOP: int = 5

    # Data Augmentation parameters
    ROTATION_RANGE: int = 20
    WIDTH_SHIFT_RANGE: float = 0.2
    HEIGHT_SHIFT_RANGE: float = 0.2
    SHEAR_RANGE: float = 0.2
    ZOOM_RANGE: float = 0.2
    HORIZONTAL_FLIP: bool = True
    FILL_MODE: str = "nearest"

    # Default file paths
    DEFAULT_DATA_DIR: str = os.environ.get("CHEST_XRAY_DIR", os.path.join("data", "chest_xray"))
    CHECKPOINT_PATH: str = "best_model.h5"
    FINAL_MODEL_PATH: str = "pneumonia_inceptionv3.keras"
    RESULTS_DIR: str = "assets"
