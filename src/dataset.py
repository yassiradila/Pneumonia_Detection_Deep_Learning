"""Data preprocessing, augmentation, and dataset loading utilities."""

import os
from typing import Optional, Tuple
import numpy as np
from PIL import Image

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
except (ImportError, ModuleNotFoundError):
    try:
        from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
    except (ImportError, ModuleNotFoundError):
        from keras.src.legacy.preprocessing.image import ImageDataGenerator, DirectoryIterator

from .config import Config


def get_data_generators(
    data_dir: Optional[str] = None,
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
    test_dir: Optional[str] = None,
    target_size: Tuple[int, int] = (299, 299),
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator]:
    """Create training, validation, and testing data generators.

    Applies data augmentation on training set (rotation, width/height shift,
    zoom, shear, horizontal flip) and standard normalization (1/255) on all sets.

    Args:
        data_dir: Root dataset folder containing 'train', 'val', and 'test' subdirectories.
        train_dir: Explicit path to train directory.
        val_dir: Explicit path to val directory.
        test_dir: Explicit path to test directory.
        target_size: Image dimensions expected by the model (default: (299, 299)).
        batch_size: Batch size for training and evaluation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_generator, val_generator, test_generator)
    """
    if data_dir is not None:
        train_dir = train_dir or os.path.join(data_dir, "train")
        val_dir = val_dir or os.path.join(data_dir, "val")
        test_dir = test_dir or os.path.join(data_dir, "test")

    for path, name in [(train_dir, "train"), (val_dir, "val"), (test_dir, "test")]:
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"Directory for {name} data does not exist at '{path}'. "
                f"Please verify your dataset directory structure."
            )

    # Training generator with robust data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=Config.ROTATION_RANGE,
        width_shift_range=Config.WIDTH_SHIFT_RANGE,
        height_shift_range=Config.HEIGHT_SHIFT_RANGE,
        shear_range=Config.SHEAR_RANGE,
        zoom_range=Config.ZOOM_RANGE,
        horizontal_flip=Config.HORIZONTAL_FLIP,
        fill_mode=Config.FILL_MODE,
    )

    # Validation and testing generators (scaling only)
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
        seed=seed,
    )

    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    test_gen = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def preprocess_single_image(
    image_path: str,
    target_size: Tuple[int, int] = (299, 299),
) -> np.ndarray:
    """Load, convert to RGB, resize, and normalize a single chest X-ray image.

    Args:
        image_path: Absolute or relative path to the image file.
        target_size: Expected (height, width) tuple.

    Returns:
        np.ndarray: Preprocessed batch tensor of shape (1, H, W, 3) scaled to [0, 1].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at '{image_path}'")

    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        img_resized = img_rgb.resize(target_size, Image.Resampling.BILINEAR)
        img_arr = np.array(img_resized, dtype=np.float32) / 255.0
        # Expand dims to simulate batch: (1, 299, 299, 3)
        return np.expand_dims(img_arr, axis=0)
