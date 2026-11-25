"""
NeuroScan AI — Unit Tests for Phase 8 LangGraph Workflow
Validates state schema, intent classification, node transitions,
safety validation, classifier immutability, and dual-engine execution.
"""
import unittest
from unittest.mock import patch, MagicMock
import os

from src.neuroscan.workflow.state import NeuroScanState, create_initial_state
from src.neuroscan.workflow.intent import classify_intent
from src.neuroscan.workflow.nodes import (
    input_validation_node,
    intent_identification_node,
    context_preparation_node,
    retrieval_node,
    generation_node,
    safety_validation_node,
)
from src.neuroscan.workflow.graph import (
    is_langgraph_available,
    build_langgraph_workflow,
    run_neuroscan_workflow,
    _execute_sequential_fallback,
)


class TestWorkflowStateAndIntent(unittest.TestCase):
    """Tests state initialization and semantic intent identification."""

    def test_state_initialization(self):
        state = create_initial_state("What is Alzheimer's?", language="english")
        self.assertEqual(state["user_question"], "What is Alzheimer's?")
        self.assertEqual(state["intent"], "unknown")
        self.assertEqual(state["language"], "english")
        self.assertEqual(state["retrieved_documents"], [])
        self.assertEqual(state["workflow_steps"], [])
        self.assertIsNone(state["classifier_context"])
        self.assertIsNone(state["error"])

    def test_intent_classification_medical_advice(self):
        self.assertEqual(classify_intent("Do I have Alzheimer's disease?"), "medical_advice")
        self.assertEqual(classify_intent("What medicine should I take for dementia?"), "medical_advice")
        self.assertEqual(classify_intent("Kya mujhe Alzheimer hai?"), "medical_advice")

    def test_intent_classification_mri_explanation(self):
        self.assertEqual(classify_intent("Explain this classification result."), "mri_explanation")
        self.assertEqual(classify_intent("Why did the model predict this class?"), "mri_explanation")
        ctx = {"predicted_class": "Mild_Demented", "confidence": 0.89, "probabilities": {}}
        self.assertEqual(classify_intent("What is the prediction for this scan?", classifier_context=ctx), "mri_explanation")

    def test_intent_classification_gradcam(self):
        self.assertEqual(classify_intent("How does Grad-CAM work in this model?"), "gradcam_explanation")
        self.assertEqual(classify_intent("Explain the attention heatmap saliency."), "gradcam_explanation")

    def test_intent_classification_limitations(self):
        self.assertEqual(classify_intent("Why is patient-level data leakage a concern?"), "limitations")
        self.assertEqual(classify_intent("Explain the minority class imbalance."), "limitations")

    def test_intent_classification_educational(self):
        self.assertEqual(classify_intent("What is Alzheimer's disease?"), "educational_question")
        self.assertEqual(classify_intent("Explain the four dataset categories."), "educational_question")
        self.assertEqual(classify_intent("What is an axial brain MRI slice?"), "educational_question")
        self.assertEqual(classify_intent("Explain EfficientNet-B0 and compound scaling."), "educational_question")

    def test_intent_classification_out_of_scope(self):
        self.assertEqual(classify_intent("How do I make chocolate chip cookies?"), "out_of_scope")
        self.assertEqual(classify_intent("Who is the prime minister of Canada?"), "out_of_scope")


class TestWorkflowNodesAndSafety(unittest.TestCase):
    """Tests execution of individual workflow stages and guardrails."""

    def test_input_validation_node_rejects_forbidden_keys(self):
        bad_state = create_initial_state("Question with 'image_path': 'test.jpg'")
        res = input_validation_node(bad_state)
        self.assertEqual(res["safety_status"], "security_violation")
        self.assertIn("Security violation", res["error"])
        self.assertIn("input_validation", res["workflow_steps"])

    def test_context_preparation_node_validates_classifier_result(self):
        valid_ctx = {
            "predicted_class": "Mild_Demented",
            "confidence": 0.88,
            "probabilities": {
                "Mild_Demented": 0.88,
                "Moderate_Demented": 0.02,
                "Non_Demented": 0.05,
                "Very_Mild_Demented": 0.05,
            }
        }
        state = create_initial_state("Explain result", classifier_context=valid_ctx)
        res = context_preparation_node(state)
        self.assertIsNotNone(res["classifier_context"])
        self.assertIn("context_preparation", res["workflow_steps"])

    def test_context_preparation_node_rejects_image_payloads(self):
        bad_ctx = {
            "predicted_class": "Mild_Demented",
            "confidence": 0.88,
            "raw_image": [1, 2, 3],  # Prohibited key
        }
        state = create_initial_state("Explain result", classifier_context=bad_ctx)
        res = context_preparation_node(state)
        self.assertEqual(res["safety_status"], "security_violation")
        self.assertIn("Security violation", res["error"])

    def test_medical_advice_routing_and_disclaimer(self):
        state = create_initial_state("Do I have Alzheimer's disease? Prescribe medicine.")
        state = input_validation_node(state)
        state = intent_identification_node(state)
        state = retrieval_node(state)
        state = generation_node(state)
        state = safety_validation_node(state)

        self.assertEqual(state["intent"], "medical_advice")
        self.assertEqual(state["safety_status"], "medical_advice_intercepted")
        self.assertEqual(state["response_source"], "clinical_disclaimer")
        self.assertIn("Medical Disclaimer", state["generated_answer"])
        self.assertIn("neurologist", state["generated_answer"].lower())
        # Retrieval was skipped
        self.assertEqual(len(state["retrieved_documents"]), 0)

    def test_out_of_scope_routing(self):
        state = create_initial_state("How do I make pancakes at home?")
        res = _execute_sequential_fallback(state)

        self.assertEqual(res["intent"], "out_of_scope")
        self.assertEqual(res["safety_status"], "out_of_scope")
        self.assertIn("does not contain sufficient information", res["generated_answer"])

    def test_educational_question_routing_through_rag(self):
        state = create_initial_state("What is Alzheimer's disease?", language="english")
        res = _execute_sequential_fallback(state)

        self.assertEqual(res["intent"], "educational_question")
        self.assertGreater(len(res["retrieved_documents"]), 0)
        self.assertIn("alzheimer", res["generated_answer"].lower())
        self.assertEqual(res["safety_status"], "passed")
        self.assertGreater(len(res["sources"]), 0)

    def test_mri_explanation_routing_with_classifier_context(self):
        valid_ctx = {
            "predicted_class": "Moderate_Demented",
            "confidence": 0.88,
            "probabilities": {
                "Mild_Demented": 0.08,
                "Moderate_Demented": 0.88,
                "Non_Demented": 0.01,
                "Very_Mild_Demented": 0.03,
            }
        }
        state = create_initial_state(
            "Explain this classification result.",
            classifier_context=valid_ctx,
            language="english"
        )
        res = _execute_sequential_fallback(state)

        self.assertEqual(res["intent"], "mri_explanation")
        self.assertEqual(res["response_source"], "classifier_explainer")
        self.assertIn("Moderate_Demented", res["generated_answer"])
        # Classifier output was NOT modified
        self.assertEqual(res["classifier_context"]["predicted_class"], "Moderate_Demented")
        self.assertEqual(res["classifier_context"]["confidence"], 0.88)


class TestWorkflowEngineAndIntegration(unittest.TestCase):
    """Tests end-to-end workflow execution and dual-engine compatibility."""

    def test_run_neuroscan_workflow_end_to_end(self):
        res = run_neuroscan_workflow("What is an axial brain MRI slice?", language="english")
        self.assertIn("query", res)
        self.assertIn("answer", res)
        self.assertIn("sources", res)
        self.assertIn("intent", res)
        self.assertIn("safety_status", res)
        self.assertIn("workflow_steps", res)
        self.assertIn("engine", res)
        self.assertEqual(res["safety_status"], "passed")
        self.assertIn("axial", res["answer"].lower())
        # Confirm all 6 workflow steps executed
        expected_steps = [
            "input_validation",
            "intent_identification",
            "context_preparation",
            "knowledge_retrieval",
            "answer_generation",
            "safety_validation",
        ]
        self.assertEqual(res["workflow_steps"], expected_steps)

    def test_run_neuroscan_workflow_hinglish(self):
        res = run_neuroscan_workflow("What is Alzheimer's disease?", language="hinglish")
        self.assertEqual(res["language"], "hinglish")
        self.assertIn("alzheimer", res["answer"].lower())

    def test_run_neuroscan_workflow_offline_fallback(self):
        """Tests that workflow gracefully generates local fallback when LLM credentials are empty."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=True):
            res = run_neuroscan_workflow("What is Alzheimer's disease?", language="english")
            self.assertEqual(res["safety_status"], "passed")
            self.assertIn("alzheimer", res["answer"].lower())
            self.assertIn("Notice: Local Knowledge Base Retrieval", res["answer"])

    def test_langgraph_dependency_check(self):
        """Verifies is_langgraph_available reflects installed package state without crashing."""
        avail = is_langgraph_available()
        self.assertIsInstance(avail, bool)
        if avail:
            compiled = build_langgraph_workflow()
            self.assertIsNotNone(compiled)


if __name__ == "__main__":
    unittest.main()
