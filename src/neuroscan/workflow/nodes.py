"""
NeuroScan AI — Workflow Processing Nodes
Implements modular functional nodes for each stage of the LangGraph workflow:
1. Input validation
2. Intent identification
3. Context preparation
4. Knowledge retrieval
5. Answer generation
6. Safety validation
"""
from typing import Dict, Any, List, Optional
import os
import re

from .state import NeuroScanState
from .intent import classify_intent
from ..rag.generator import sanitize_input, generate_rag_answer, FORBIDDEN_IMAGE_KEYS
from ..rag.retriever import EducationalRetriever, format_source_citations
from ..llm.explainer import explain_result, validate_classifier_result


def input_validation_node(state: NeuroScanState) -> NeuroScanState:
    """
    Validates and sanitizes the user query.
    Rejects forbidden image keys, tensors, or pixel payloads.
    """
    steps = list(state.get("workflow_steps", []))
    steps.append("input_validation")

    query = state.get("user_question", "")
    try:
        clean_query = sanitize_input(query)
        return {
            **state,
            "user_question": clean_query,
            "workflow_steps": steps,
        }
    except Exception as e:
        return {
            **state,
            "error": str(e),
            "safety_status": "security_violation",
            "generated_answer": f"⚠️ Validation Error: {e}",
            "response_source": "error_handler",
            "workflow_steps": steps,
        }


def intent_identification_node(state: NeuroScanState) -> NeuroScanState:
    """
    Identifies the semantic intent of the query and routes downstream nodes.
    """
    steps = list(state.get("workflow_steps", []))
    steps.append("intent_identification")

    # If previous stage encountered a security error, skip intent identification
    if state.get("safety_status") == "security_violation":
        return {**state, "workflow_steps": steps}

    query = state.get("user_question", "")
    classifier_ctx = state.get("classifier_context")
    intent = classify_intent(query, classifier_ctx)

    return {
        **state,
        "intent": intent,
        "workflow_steps": steps,
    }


def context_preparation_node(state: NeuroScanState) -> NeuroScanState:
    """
    Validates and prepares classifier context if provided.
    Guarantees no raw image bytes, tensors, or local file paths are passed.
    """
    steps = list(state.get("workflow_steps", []))
    steps.append("context_preparation")

    if state.get("safety_status") == "security_violation":
        return {**state, "workflow_steps": steps}

    ctx = state.get("classifier_context")
    if ctx:
        try:
            # Validate using existing strict classifier schema
            validated_ctx = validate_classifier_result(ctx)
            return {
                **state,
                "classifier_context": validated_ctx,
                "workflow_steps": steps,
            }
        except Exception as e:
            return {
                **state,
                "error": str(e),
                "safety_status": "security_violation",
                "generated_answer": f"⚠️ Classifier Context Validation Error: {e}",
                "response_source": "error_handler",
                "workflow_steps": steps,
            }

    return {**state, "workflow_steps": steps}


def retrieval_node(state: NeuroScanState) -> NeuroScanState:
    """
    Retrieves grounded educational knowledge chunks from the knowledge base
    when the query requires medical or deep learning background.
    """
    steps = list(state.get("workflow_steps", []))
    steps.append("knowledge_retrieval")

    if state.get("safety_status") == "security_violation":
        return {**state, "workflow_steps": steps}

    intent = state.get("intent", "educational_question")

    # Medical advice requests and pure classifier explanations do not require knowledge retrieval
    if intent in ["medical_advice"]:
        return {
            **state,
            "retrieved_documents": [],
            "sources": [],
            "workflow_steps": steps,
        }

    retriever = EducationalRetriever()
    query = state.get("user_question", "")

    retrieval_res = retriever.retrieve(query, top_k=3)
    chunks = retrieval_res.get("chunks", [])
    sources = retrieval_res.get("sources", [])

    return {
        **state,
        "retrieved_documents": chunks,
        "sources": sources,
        "workflow_steps": steps,
    }


def generation_node(state: NeuroScanState) -> NeuroScanState:
    """
    Synthesizes the educational answer based on intent, retrieved context,
    and structured classifier predictions.
    """
    steps = list(state.get("workflow_steps", []))
    steps.append("answer_generation")

    if state.get("safety_status") == "security_violation":
        return {**state, "workflow_steps": steps}

    intent = state.get("intent", "educational_question")
    language = state.get("language", "english")
    query = state.get("user_question", "")
    is_hinglish = language in ["hindi", "hinglish"]

    # 1. Medical Advice / Diagnosis request -> Immediate authoritative disclaimer
    if intent == "medical_advice":
        disclaimer = (
            "⚠️ **Important Medical Disclaimer**: NeuroScan AI is strictly an undergraduate engineering academic project, "
            "not a clinical diagnostic system. It cannot diagnose individual health conditions or recommend medical treatments or drugs. "
            "If you or a loved one are experiencing memory loss or neurological symptoms, please schedule an evaluation with a "
            "licensed physician or certified neurologist."
            if not is_hinglish else
            "⚠️ **Zaroori Medical Disclaimer**: NeuroScan AI sirf ek college student engineering project hai, "
            "koi clinical hospital diagnostic tool nahi hai. Yeh kisi vyakti ki bimari diagnose nahi kar sakta aur na hi dawai (medicines) prescribe kar sakta hai. "
            "Kisi bhi lakshan ya memory concern ke liye kripya turant certified doctor ya neurologist se salah lein."
        )
        return {
            **state,
            "generated_answer": disclaimer,
            "response_source": "clinical_disclaimer",
            "safety_status": "medical_advice_intercepted",
            "workflow_steps": steps,
        }

    # 2. Classifier result explanation -> Use structured classifier explainer
    if intent == "mri_explanation" and state.get("classifier_context"):
        try:
            exp_res = explain_result(state["classifier_context"], language=language)
            return {
                **state,
                "generated_answer": exp_res["explanation"],
                "response_source": "classifier_explainer",
                "safety_status": "passed",
                "workflow_steps": steps,
            }
        except Exception as e:
            return {
                **state,
                "error": str(e),
                "generated_answer": f"Unable to generate classifier explanation: {e}",
                "response_source": "error_handler",
                "workflow_steps": steps,
            }

    # 3. Out-of-scope queries
    if intent == "out_of_scope" or (not state.get("retrieved_documents") and not state.get("classifier_context")):
        out_msg = (
            "The available educational knowledge base does not contain sufficient information to answer this question. "
            "You can ask about Alzheimer's pathology, the four project dataset categories, 2D axial MRI slices, "
            "the EfficientNet-B0 transfer learning architecture, or Grad-CAM explainability."
            if not is_hinglish else
            "Di gayi educational knowledge base mein is sawal ki jankari uplabdh nahi hai. "
            "Aap Alzheimer's pathology, project ke 4 dataset categories, 2D axial MRI slices, "
            "EfficientNet-B0 architecture, ya Grad-CAM explainability ke bare mein puch sakte hain."
        )
        return {
            **state,
            "generated_answer": out_msg,
            "response_source": "langgraph_fallback",
            "safety_status": "out_of_scope",
            "workflow_steps": steps,
        }

    # 4. Standard Educational / Grad-CAM / Limitations / Patient Care query -> Use RAG Generator
    rag_res = generate_rag_answer(
        query=query,
        retrieved_chunks=state.get("retrieved_documents", []),
        language=language,
        use_fallback_on_error=True,
        classifier_context=state.get("classifier_context"),
    )

    return {
        **state,
        "generated_answer": rag_res["answer"],
        "sources": rag_res.get("sources", state.get("sources", [])),
        "response_source": f"langgraph_{rag_res.get('status', 'success')}",
        "safety_status": "passed",
        "workflow_steps": steps,
    }


def safety_validation_node(state: NeuroScanState) -> NeuroScanState:
    """
    Validates final output against safety rules:
    - Never modifies classifier outputs or labels.
    - Prevents prompt-injection leakages.
    - Confirms educational boundary.
    """
    steps = list(state.get("workflow_steps", []))
    steps.append("safety_validation")

    answer = state.get("generated_answer", "")
    safety_status = state.get("safety_status", "passed")

    # Ensure internal tags are not leaked into user text
    if "<retrieved_context>" in answer or "</retrieved_context>" in answer:
        answer = answer.replace("<retrieved_context>", "").replace("</retrieved_context>", "").strip()

    # If classifier context was provided, verify model prediction was not tampered with
    ctx = state.get("classifier_context")
    if ctx:
        pred_class = ctx.get("predicted_class")
        if pred_class and pred_class not in answer and state.get("intent") == "mri_explanation":
            # Append clarification if class was omitted
            answer += f"\n\n*(Verified Dataset Prediction: `{pred_class}` with {ctx.get('confidence', 0.0)*100:.2f}% score)*"

    return {
        **state,
        "generated_answer": answer,
        "safety_status": safety_status if safety_status != "pending" else "passed",
        "workflow_steps": steps,
    }
