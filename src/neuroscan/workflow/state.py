"""
NeuroScan AI — Workflow State Definition
Defines the state schema for the LangGraph multi-stage educational workflow.
Preserves strict safety: never stores raw images, tensors, or pixel arrays.
"""
from typing import TypedDict, List, Dict, Any, Optional


class NeuroScanState(TypedDict, total=False):
    """
    State dictionary passed through all nodes in the NeuroScan AI workflow.

    Fields:
        user_question: Plain-text natural language query.
        intent: Identified question intent.
        language: "english" or "hinglish".
        classifier_context: Optional dictionary containing structured classifier outputs
            (predicted_class, confidence, probabilities). Never image bytes or tensors.
        retrieved_documents: List of retrieved educational chunks from the knowledge base.
        sources: Deduplicated source citations with title, publisher, URL, and category.
        generated_answer: Synthesized, grounded Markdown response.
        response_source: Identifier for how the answer was created
            ("langgraph_gemini", "langgraph_fallback", "rag_pipeline", "clinical_disclaimer").
        safety_status: Status indicator ("passed", "medical_advice_intercepted", "security_violation", "out_of_scope").
        workflow_steps: List of executed node names for transparency and telemetry.
        error: Optional telemetry string if a node encountered an exception.
    """
    user_question: str
    intent: str
    language: str
    classifier_context: Optional[Dict[str, Any]]
    retrieved_documents: List[Dict[str, Any]]
    sources: List[Dict[str, str]]
    generated_answer: str
    response_source: str
    safety_status: str
    workflow_steps: List[str]
    error: Optional[str]


def create_initial_state(
    user_question: str,
    classifier_context: Optional[Dict[str, Any]] = None,
    language: str = "english",
) -> NeuroScanState:
    """Initializes a clean NeuroScanState with validated parameters."""
    return {
        "user_question": user_question.strip(),
        "intent": "unknown",
        "language": language.lower().strip() if language else "english",
        "classifier_context": classifier_context,
        "retrieved_documents": [],
        "sources": [],
        "generated_answer": "",
        "response_source": "uninitialized",
        "safety_status": "pending",
        "workflow_steps": [],
        "error": None,
    }
