"""
NeuroScan AI — Unit Tests for Vision Inference and Grad-CAM Modules
Validates checkpoint loading, preprocessing, inference accuracy,
and Grad-CAM hook mechanics.
"""
import unittest
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import matplotlib.figure

from src.neuroscan.vision.inference import (
    load_trained_model,
    preprocess_image,
    predict_mri,
    get_device,
    get_transform,
)
from src.neuroscan.explainability.gradcam import (
    generate_gradcam,
    create_gradcam_figure,
)
from src.neuroscan.config import ARTIFACTS_DIR, CLASS_NAMES


class TestVisionAndGradCAM(unittest.TestCase):
    """Unit tests for inference and explainability pipelines."""

    @classmethod
    def setUpClass(cls):
        """Loads model once for tests."""
        cls.ckpt_path = ARTIFACTS_DIR / "best_efficientnet_b0.pth"
        if not cls.ckpt_path.exists():
            raise unittest.SkipTest("Trained checkpoint not found in artifacts/.")

        cls.device = torch.device("cpu")  # Run unit tests on CPU for speed & determinism
        cls.model, cls.idx_to_class, cls.class_to_idx, cls.metadata = load_trained_model(
            checkpoint_path=cls.ckpt_path,
            device=cls.device,
        )

        # Create a synthetic 128x128 grayscale test image matching dataset format
        arr = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        cls.synthetic_img = Image.fromarray(arr)

    def test_checkpoint_loaded_properties(self):
        """Tests that checkpoint loads valid architecture and metadata."""
        self.assertFalse(self.model.training)
        self.assertEqual(len(self.idx_to_class), 4)
        self.assertEqual(len(self.class_to_idx), 4)
        for name in CLASS_NAMES:
            self.assertIn(name, self.class_to_idx)
        self.assertEqual(self.metadata["epoch"], 15)

    def test_preprocessing_pipeline(self):
        """Tests that preprocessing produces tensor of shape (1, 3, 224, 224)."""
        tensor = preprocess_image(self.synthetic_img, device=self.device)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.device, self.device)
        self.assertEqual(tensor.dtype, torch.float32)

    def test_predict_mri(self):
        """Tests inference generates valid probabilities summing to ~1.0."""
        tensor = preprocess_image(self.synthetic_img, device=self.device)
        res = predict_mri(self.model, tensor, self.idx_to_class)

        self.assertIn("predicted_class", res)
        self.assertIn(res["predicted_class"], CLASS_NAMES)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)

        # Check all 4 probabilities
        probs = res["probabilities"]
        self.assertEqual(len(probs), 4)
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_generate_gradcam(self):
        """Tests that Grad-CAM produces a normalized 224x224 heatmap."""
        tensor = preprocess_image(self.synthetic_img, device=self.device)
        heatmap = generate_gradcam(self.model, tensor)

        self.assertEqual(heatmap.shape, (224, 224))
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)

    def test_create_gradcam_figure(self):
        """Tests figure generation returns valid Matplotlib Figure."""
        tensor = preprocess_image(self.synthetic_img, device=self.device)
        heatmap = generate_gradcam(self.model, tensor)
        fig = create_gradcam_figure(
            self.synthetic_img, heatmap, "Moderate_Demented", 0.88
        )
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
