"""
NeuroScan AI — LangGraph Workflow Orchestrator
Builds and compiles the StateGraph workflow connecting input validation,
intent routing, context preparation, knowledge retrieval, grounded generation,
and safety validation.
Provides a resilient pure-Python sequential fallback when the langgraph package is absent.
"""
from typing import Dict, Any, Optional
import os

from .state import NeuroScanState, create_initial_state
from .nodes import (
    input_validation_node,
    intent_identification_node,
    context_preparation_node,
    retrieval_node,
    generation_node,
    safety_validation_node,
)


def is_langgraph_available() -> bool:
    """Checks whether the langgraph package is installed in the active environment."""
    try:
        import langgraph
        from langgraph.graph import StateGraph, END
        return True
    except ImportError:
        return False


def build_langgraph_workflow():
    """
    Constructs and compiles the LangGraph StateGraph when langgraph is installed.
    Raises ImportError if langgraph is not available.
    """
    from langgraph.graph import StateGraph, END

    workflow = StateGraph(NeuroScanState)

    # Add pipeline nodes
    workflow.add_node("input_validation", input_validation_node)
    workflow.add_node("intent_identification", intent_identification_node)
    workflow.add_node("context_preparation", context_preparation_node)
    workflow.add_node("knowledge_retrieval", retrieval_node)
    workflow.add_node("answer_generation", generation_node)
    workflow.add_node("safety_validation", safety_validation_node)

    # Establish linear state graph edges
    workflow.set_entry_point("input_validation")
    workflow.add_edge("input_validation", "intent_identification")
    workflow.add_edge("intent_identification", "context_preparation")
    workflow.add_edge("context_preparation", "knowledge_retrieval")
    workflow.add_edge("knowledge_retrieval", "answer_generation")
    workflow.add_edge("answer_generation", "safety_validation")
    workflow.add_edge("safety_validation", END)

    return workflow.compile()


def _execute_sequential_fallback(state: NeuroScanState) -> NeuroScanState:
    """
    Executes the exact same multi-stage state transitions using pure Python
    when the langgraph package is missing from the environment.
    Guarantees LangGraph is never a single point of failure.
    """
    current_state = input_validation_node(state)
    current_state = intent_identification_node(current_state)
    current_state = context_preparation_node(current_state)
    current_state = retrieval_node(current_state)
    current_state = generation_node(current_state)
    current_state = safety_validation_node(current_state)
    return current_state


# Cached compiled graph instance if available
_compiled_graph = None


def get_compiled_graph():
    """Returns the cached compiled LangGraph instance or None if langgraph is absent."""
    global _compiled_graph
    if _compiled_graph is None and is_langgraph_available():
        try:
            _compiled_graph = build_langgraph_workflow()
        except Exception:
            _compiled_graph = None
    return _compiled_graph


def run_neuroscan_workflow(
    user_question: str,
    classifier_context: Optional[Dict[str, Any]] = None,
    language: str = "english",
) -> Dict[str, Any]:
    """
    Main entry point for executing the multi-stage NeuroScan AI workflow.

    Args:
        user_question: Natural language question text.
        classifier_context: Optional dictionary with quantitative classifier predictions
            (predicted_class, confidence, probabilities).
        language: "english" or "hinglish".

    Returns:
        Structured response dictionary with answer, sources, intent, safety_status,
        response_source, and workflow execution steps.
    """
    initial_state = create_initial_state(
        user_question=user_question,
        classifier_context=classifier_context,
        language=language,
    )

    compiled_app = get_compiled_graph()

    if compiled_app is not None:
        try:
            final_state = compiled_app.invoke(initial_state)
            engine = "langgraph_compiled"
        except Exception as e:
            # Fall back to sequential execution on runtime exception
            final_state = _execute_sequential_fallback(initial_state)
            final_state["error"] = f"LangGraph execution exception: {e}"
            engine = "langgraph_sequential_fallback"
    else:
        final_state = _execute_sequential_fallback(initial_state)
        engine = "python_sequential_engine"

    return {
        "query": final_state.get("user_question", user_question),
        "answer": final_state.get("generated_answer", ""),
        "sources": final_state.get("sources", []),
        "intent": final_state.get("intent", "unknown"),
        "safety_status": final_state.get("safety_status", "passed"),
        "response_source": final_state.get("response_source", "unknown"),
        "workflow_steps": final_state.get("workflow_steps", []),
        "engine": engine,
        "language": final_state.get("language", language),
        "error": final_state.get("error"),
    }
