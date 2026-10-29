"""
NeuroScan AI — LLM Result Explainer Package
Provides educational result explanation for classifier predictions without sending image data.
"""

from .explainer import explain_result, validate_classifier_result
from .provider import call_llm

__all__ = ["explain_result", "validate_classifier_result", "call_llm"]

