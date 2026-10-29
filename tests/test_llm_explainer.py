"""
NeuroScan AI — Unit Tests for LLM Result Explainer Module
Validates validation rules, provider handling, prompt building, language options,
error fallbacks, privacy guards, and result immutability.
"""
import os
import unittest
from unittest.mock import patch, MagicMock

from src.neuroscan.llm.explainer import (
    validate_classifier_result,
    build_explanation_prompt,
    build_fallback_explanation,
    explain_result,
)
from src.neuroscan.llm.provider import call_llm


class TestLLMExplainer(unittest.TestCase):
    """Test suite for Phase 5 LLM Result Explainer."""

    def setUp(self):
        """Set up standard valid classifier outputs for testing."""
        self.valid_result = {
            "predicted_class": "Mild_Demented",
            "confidence": 0.8950,
            "probabilities": {
                "Mild_Demented": 0.8950,
                "Moderate_Demented": 0.0050,
                "Non_Demented": 0.0200,
                "Very_Mild_Demented": 0.0800,
            },
        }

    # -------------------------------------------------------------------------
    # 1. Valid classifier result & English output
    # -------------------------------------------------------------------------
    @patch("src.neuroscan.llm.explainer.call_llm")
    def test_valid_classifier_result_english(self, mock_call_llm):
        """Tests that a valid result in English returns success status and correct fields."""
        mock_call_llm.return_value = (
            "The model predicted Mild Demented with an 89.50% confidence score. "
            "This reflects noticeable structural differences in this 2D slice dataset. "
            "This is an educational student project and not a clinical diagnosis."
        )

        res = explain_result(self.valid_result, language="english")

        self.assertEqual(res["status"], "success")
        self.assertIsNone(res["error"])
        self.assertEqual(res["language"], "english")
        self.assertEqual(res["predicted_class"], "Mild_Demented")
        self.assertEqual(res["confidence"], 0.8950)
        self.assertIn("Mild Demented", res["explanation"])
        self.assertTrue(mock_call_llm.called)

    # -------------------------------------------------------------------------
    # 2. Hindi / Hinglish output request
    # -------------------------------------------------------------------------
    @patch("src.neuroscan.llm.explainer.call_llm")
    def test_valid_classifier_result_hinglish(self, mock_call_llm):
        """Tests Hinglish request passes language instructions and returns Hinglish response."""
        mock_call_llm.return_value = (
            "Model ne 89.50% confidence ke saath Mild Demented predict kiya hai. "
            "Yeh ek student college project hai, koi medical diagnosis nahi hai."
        )

        res = explain_result(self.valid_result, language="hinglish")

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["language"], "hinglish")
        self.assertIn("Mild Demented", res["explanation"])

        # Check that the prompt passed to call_llm requested Hinglish
        called_prompt = mock_call_llm.call_args[1].get("prompt") or mock_call_llm.call_args[0][0]
        self.assertIn("Hinglish", called_prompt)

    # -------------------------------------------------------------------------
    # 3. Missing or malformed result fields
    # -------------------------------------------------------------------------
    def test_missing_required_fields(self):
        """Tests that missing 'predicted_class' or 'confidence' raises ValueError."""
        # Non-dict
        with self.assertRaises(ValueError):
            validate_classifier_result(["not", "a", "dict"])

        # Missing predicted_class
        with self.assertRaises(ValueError):
            validate_classifier_result({"confidence": 0.95})

        # Missing confidence
        with self.assertRaises(ValueError):
            validate_classifier_result({"predicted_class": "Non_Demented"})

        # Empty dict
        with self.assertRaises(ValueError):
            validate_classifier_result({})

    def test_malformed_fields(self):
        """Tests that invalid classes or out-of-range confidence values raise ValueError."""
        # Unknown class name
        with self.assertRaises(ValueError):
            validate_classifier_result({
                "predicted_class": "Severe_Alzheimers",
                "confidence": 0.85,
            })

        # Non-numeric confidence
        with self.assertRaises(ValueError):
            validate_classifier_result({
                "predicted_class": "Non_Demented",
                "confidence": "high",
            })

        # Out-of-bounds confidence
        with self.assertRaises(ValueError):
            validate_classifier_result({
                "predicted_class": "Non_Demented",
                "confidence": -0.5,
            })

        with self.assertRaises(ValueError):
            validate_classifier_result({
                "predicted_class": "Non_Demented",
                "confidence": 150.0,
            })

    def test_confidence_normalization(self):
        """Tests that percentages like 95.5 are correctly normalized to 0.955."""
        validated = validate_classifier_result({
            "predicted_class": "Non_Demented",
            "confidence": 95.5,
        })
        self.assertAlmostEqual(validated["confidence"], 0.955, places=3)

    # -------------------------------------------------------------------------
    # 4. Missing API credentials
    # -------------------------------------------------------------------------
    def test_missing_api_credentials_returns_safe_fallback(self):
        """Tests that missing API credentials return a safe fallback without crashing."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=True):
            res = explain_result(self.valid_result, language="english", use_fallback_on_error=True)

            self.assertEqual(res["status"], "fallback")
            self.assertIsNotNone(res["error"])
            self.assertIn("API key is not configured", res["error"])
            # Fallback explanation contains educational disclaimer and exact prediction
            self.assertIn("Automated Fallback Notice", res["explanation"])
            self.assertIn("Mild_Demented", res["explanation"])
            self.assertIn("89.5%", res["explanation"])
            self.assertEqual(res["predicted_class"], "Mild_Demented")
            self.assertEqual(res["confidence"], 0.8950)

    def test_missing_api_credentials_raises_when_fallback_disabled(self):
        """Tests that missing API credentials raise ValueError when fallback is disabled."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=True):
            with self.assertRaises(ValueError):
                call_llm("test prompt", provider="gemini")

    # -------------------------------------------------------------------------
    # 5. Provider / API failure
    # -------------------------------------------------------------------------
    @patch("requests.post")
    def test_provider_network_failure(self, mock_post):
        """Tests network timeout or connection error triggers fallback gracefully."""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out after 20s")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_key"}):
            res = explain_result(self.valid_result, language="english", use_fallback_on_error=True)

            self.assertEqual(res["status"], "fallback")
            self.assertIn("timed out", res["error"])
            self.assertIn("Automated Fallback Notice", res["explanation"])

    @patch("requests.post")
    def test_provider_http_500_failure(self, mock_post):
        """Tests provider returning HTTP 500 error triggers fallback gracefully."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_key"}):
            res = explain_result(self.valid_result, language="english", use_fallback_on_error=True)

            self.assertEqual(res["status"], "fallback")
            self.assertIn("HTTP 500", res["error"])

    # -------------------------------------------------------------------------
    # 6. Invalid or empty LLM response
    # -------------------------------------------------------------------------
    @patch("requests.post")
    def test_empty_candidates_response(self, mock_post):
        """Tests handling when provider returns JSON with no candidates."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"candidates": []}
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_key"}):
            res = explain_result(self.valid_result, language="english", use_fallback_on_error=True)

            self.assertEqual(res["status"], "fallback")
            self.assertIn("no candidates", res["error"])

    @patch("requests.post")
    def test_empty_text_part_response(self, mock_post):
        """Tests handling when provider candidate has empty text part."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": ""}]}}]
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_key"}):
            res = explain_result(self.valid_result, language="english", use_fallback_on_error=True)

            self.assertEqual(res["status"], "fallback")
            self.assertIn("no text content", res["error"])

    # -------------------------------------------------------------------------
    # 7. Privacy: Ensuring explainer does not receive or transmit MRI image data
    # -------------------------------------------------------------------------
    def test_rejects_image_data_and_file_paths(self):
        """Tests that passing image data or file paths is immediately rejected with ValueError."""
        # Image bytes
        bad_result_bytes = {
            "predicted_class": "Non_Demented",
            "confidence": 0.95,
            "image": b"raw_image_bytes_here",
        }
        with self.assertRaises(ValueError) as ctx:
            validate_classifier_result(bad_result_bytes)
        self.assertIn("Security violation", str(ctx.exception))

        # Image file path
        bad_result_path = {
            "predicted_class": "Non_Demented",
            "confidence": 0.95,
            "image_path": "data/Alzheimer/Non_Demented/scan_001.jpg",
        }
        with self.assertRaises(ValueError) as ctx:
            validate_classifier_result(bad_result_path)
        self.assertIn("Security violation", str(ctx.exception))

        # Raw tensor
        bad_result_tensor = {
            "predicted_class": "Non_Demented",
            "confidence": 0.95,
            "tensor": [0.1, 0.2, 0.3],
        }
        with self.assertRaises(ValueError) as ctx:
            validate_classifier_result(bad_result_tensor)
        self.assertIn("Security violation", str(ctx.exception))

    def test_prompt_contains_only_text_and_no_image_data(self):
        """Tests that build_explanation_prompt contains only structured text/numbers."""
        sys_prompt, user_prompt = build_explanation_prompt(self.valid_result)

        # Check for absence of common image indicators
        for marker in [".jpg", ".png", ".jpeg", ".dcm", "image_data", "pixels"]:
            self.assertNotIn(marker, sys_prompt.lower())
            self.assertNotIn(marker, user_prompt.lower())

        # Check presence of text labels and numbers
        self.assertIn("Mild_Demented", user_prompt)
        self.assertIn("89.50%", user_prompt)

    # -------------------------------------------------------------------------
    # 8. Integrity: Ensuring supplied label and scores are not silently replaced
    # -------------------------------------------------------------------------
    @patch("src.neuroscan.llm.explainer.call_llm")
    def test_predicted_label_and_scores_not_silently_replaced(self, mock_call_llm):
        """Tests that even if the LLM hallucinates an altered class, the returned

        dictionary strictly preserves the original classifier prediction and scores.
        """
        # Simulated LLM hallucination claiming the opposite class
        mock_call_llm.return_value = (
            "I have re-evaluated the MRI and concluded the patient is Non_Demented with 99.9% certainty."
        )

        input_result = {
            "predicted_class": "Moderate_Demented",
            "confidence": 0.8800,
            "probabilities": {
                "Moderate_Demented": 0.8800,
                "Mild_Demented": 0.0800,
                "Very_Mild_Demented": 0.0300,
                "Non_Demented": 0.0100,
            },
        }

        res = explain_result(input_result, language="english")

        # The output dictionary's quantitative fields MUST match input_result, NOT the hallucination
        self.assertEqual(res["predicted_class"], "Moderate_Demented")
        self.assertEqual(res["confidence"], 0.8800)
        self.assertEqual(res["probabilities"]["Moderate_Demented"], 0.8800)
        self.assertNotEqual(res["predicted_class"], "Non_Demented")
        self.assertNotEqual(res["confidence"], 0.999)


if __name__ == "__main__":
    unittest.main()

