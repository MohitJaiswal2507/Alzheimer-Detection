"""
NeuroScan AI — Workflow Package
Multi-stage educational reasoning and coordination workflow powered by LangGraph
with resilient sequential fallback for environments lacking langgraph.
"""
from .state import NeuroScanState, create_initial_state
from .intent import classify_intent
from .graph import (
    is_langgraph_available,
    build_langgraph_workflow,
    run_neuroscan_workflow,
    get_compiled_graph,
)

__all__ = [
    "NeuroScanState",
    "create_initial_state",
    "classify_intent",
    "is_langgraph_available",
    "build_langgraph_workflow",
    "run_neuroscan_workflow",
    "get_compiled_graph",
]
