"""
NeuroScan AI — Unit Tests for UI Components & Pipeline Integration
Validates end-to-end inference, Grad-CAM generation, and explainer formatting
for the Streamlit application.
"""
import unittest
from pathlib import Path
from PIL import Image
import torch
import matplotlib.figure

from src.neuroscan.vision.inference import (
    load_trained_model,
    preprocess_image,
    predict_mri,
)
from src.neuroscan.explainability.gradcam import (
    generate_gradcam,
    create_gradcam_figure,
)
from src.neuroscan.llm.explainer import explain_result
from src.neuroscan.config import ARTIFACTS_DIR, CLASS_NAMES


class TestUIPipelineIntegration(unittest.TestCase):
    """Integration test suite ensuring Streamlit pipeline functions reliably."""

    @classmethod
    def setUpClass(cls):
        """Sets up CPU model and sample image."""
        cls.ckpt_path = ARTIFACTS_DIR / "best_efficientnet_b0.pth"
        if not cls.ckpt_path.exists():
            raise unittest.SkipTest("Artifacts checkpoint missing.")

        cls.device = torch.device("cpu")
        cls.model, cls.idx_to_class, cls.class_to_idx, cls.metadata = load_trained_model(
            checkpoint_path=cls.ckpt_path,
            device=cls.device,
        )

        # Locate sample image from dataset or create synthetic image
        sample_path = Path("data/Alzheimer (Preprocessed Data)/Non_Demented/non.jpg")
        if sample_path.exists():
            cls.test_image = Image.open(sample_path)
        else:
            cls.test_image = Image.new("L", (128, 128), color=128)

    def test_complete_ui_inference_flow(self):
        """Tests the exact flow executed by the 'Analyze MRI' Streamlit button."""
        # 1. Preprocess
        tensor = preprocess_image(self.test_image, device=self.device)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

        # 2. Predict
        res = predict_mri(self.model, tensor, self.idx_to_class)
        self.assertIn(res["predicted_class"], CLASS_NAMES)
        self.assertGreater(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertEqual(len(res["probabilities"]), 4)

        # 3. Grad-CAM
        heatmap = generate_gradcam(self.model, tensor, target_class=res["predicted_index"])
        self.assertEqual(heatmap.shape, (224, 224))

        # 4. Figure generation
        fig = create_gradcam_figure(
            self.test_image, heatmap, res["predicted_class"], res["confidence"]
        )
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

        # 5. Explainer input formatting (Strictly NO image or path data)
        explainer_input = {
            "predicted_class": res["predicted_class"],
            "confidence": res["confidence"],
            "probabilities": res["probabilities"],
        }
        # Run with fallback enabled (works completely offline)
        exp_res = explain_result(explainer_input, language="english", use_fallback_on_error=True)
        self.assertIn(exp_res["status"], ["success", "fallback"])
        self.assertEqual(exp_res["predicted_class"], res["predicted_class"])
        self.assertEqual(exp_res["confidence"], res["confidence"])
        self.assertTrue(len(exp_res["explanation"]) > 50)


if __name__ == "__main__":
    unittest.main()

