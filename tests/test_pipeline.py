"""Unit tests for configuration, data preprocessing, and model architecture."""

import os
import tempfile
import unittest
import numpy as np
from PIL import Image

from src.config import Config
from src.dataset import preprocess_single_image
from src.model import build_inception_v3, unfreeze_for_fine_tuning


class TestPneumoniaPipeline(unittest.TestCase):
    """Test suite for the Pneumonia InceptionV3 deep learning pipeline."""

    def test_config_validity(self):
        """Verify that configuration settings and dimensions are consistent."""
        cfg = Config()
        self.assertEqual(cfg.IMG_HEIGHT, 299)
        self.assertEqual(cfg.IMG_WIDTH, 299)
        self.assertEqual(cfg.CHANNELS, 3)
        self.assertEqual(cfg.target_size, (299, 299))
        self.assertEqual(cfg.input_shape, (299, 299, 3))
        self.assertIn("NORMAL", cfg.CLASS_NAMES)
        self.assertIn("PNEUMONIA", cfg.CLASS_NAMES)
        self.assertTrue(0.0 < cfg.DECISION_THRESHOLD < 1.0)

    def test_preprocess_single_image(self):
        """Verify that preprocess_single_image resizes and scales correctly."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            dummy_img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
            dummy_img.save(tmp.name)
            tmp_path = tmp.name

        try:
            tensor = preprocess_single_image(tmp_path, target_size=(299, 299))
            self.assertEqual(tensor.shape, (1, 299, 299, 3))
            self.assertEqual(tensor.dtype, np.float32)
            self.assertGreaterEqual(float(np.min(tensor)), 0.0)
            self.assertLessEqual(float(np.max(tensor)), 1.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_model_architecture_and_unfreezing(self):
        """Verify model building and layer freezing/unfreezing logic without downloading weights."""
        model, base_model = build_inception_v3(
            input_shape=(299, 299, 3),
            dense_units=128,
            dropout_rate=0.1,
            weights=None,  # Do not download ImageNet weights during test
        )

        self.assertEqual(model.input_shape, (None, 299, 299, 3))
        self.assertEqual(model.output_shape, (None, 1))

        # Check that base model layers are frozen
        trainable_base_count = sum(1 for layer in base_model.layers if layer.trainable)
        self.assertEqual(trainable_base_count, 0, "Base model layers should be frozen initially")

        # Test unfreezing top layers for fine-tuning
        unfrozen_model = unfreeze_for_fine_tuning(
            model=model,
            base_model=base_model,
            unfreeze_layers=50,
            learning_rate=1e-5,
        )

        trainable_after = sum(1 for layer in base_model.layers if layer.trainable)
        self.assertEqual(trainable_after, 50, f"Expected 50 unfrozen base layers, got {trainable_after}")


if __name__ == "__main__":
    unittest.main()
