"""
NeuroScan AI — LLM Result Explainer Module
Converts structured classifier results into clear, educational explanations
in simple English or Hindi/Hinglish.

Strict Privacy & Safety Guardrails:
- Never receives, reads, or transmits MRI images, image paths, or pixel arrays.
- Explains the classifier's output without altering predicted labels or scores.
- Clearly states that model predictions are educational and not clinical diagnoses.
- Provides a safe rule-based fallback if API credentials are missing or the provider fails.
"""
from typing import Dict, Any, Tuple, Optional
from ..config import CLASS_NAMES, CLASS_DESCRIPTIONS, DEFAULT_LLM_PROVIDER
from .provider import call_llm

# Prohibited keys to guarantee no image data or file paths are passed
FORBIDDEN_IMAGE_KEYS = {
    "image", "img", "image_data", "raw_image", "image_path",
    "file_path", "filepath", "path", "tensor", "pixels", "mri"
}


def validate_classifier_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates the structured classifier output dictionary.

    Args:
        result: Dictionary containing 'predicted_class', 'confidence',
                and optional 'probabilities'.

    Returns:
        Clean, validated dictionary with text/numeric fields only.

    Raises:
        ValueError: If image data is present, required fields are missing,
                    or values are out of bounds.
    """
    if not isinstance(result, dict):
        raise ValueError(
            f"Classifier result must be a dictionary, got {type(result).__name__}."
        )

    # 1. Strict check: Never accept image data, paths, or tensors
    for key in result.keys():
        if key.lower() in FORBIDDEN_IMAGE_KEYS:
            raise ValueError(
                f"Security violation: Image data or file path field '{key}' "
                "must never be passed to the LLM explainer."
            )

    # 2. Required field: predicted_class
    if "predicted_class" not in result or result["predicted_class"] is None:
        raise ValueError("Missing required field: 'predicted_class'.")

    predicted_class = str(result["predicted_class"]).strip()
    if not predicted_class:
        raise ValueError("Field 'predicted_class' cannot be empty.")

    if predicted_class not in CLASS_NAMES:
        raise ValueError(
            f"Invalid predicted_class '{predicted_class}'. "
            f"Expected one of: {CLASS_NAMES}"
        )

    # 3. Required field: confidence
    if "confidence" not in result or result["confidence"] is None:
        raise ValueError("Missing required field: 'confidence'.")

    try:
        confidence = float(result["confidence"])
    except (TypeError, ValueError):
        raise ValueError(
            f"Confidence must be a numeric value, got '{result.get('confidence')}'."
        )

    # Normalize percentage (e.g. 95.5 -> 0.955) or validate range [0.0, 1.0]
    if 1.0 < confidence <= 100.0:
        confidence = confidence / 100.0
    elif confidence < 0.0 or confidence > 100.0:
        raise ValueError(
            f"Confidence score {confidence} is out of valid range [0.0, 1.0]."
        )

    # 4. Optional field: probabilities
    probabilities = {}
    if "probabilities" in result and isinstance(result["probabilities"], dict):
        for cls_name, prob in result["probabilities"].items():
            try:
                prob_float = float(prob)
                if 1.0 < prob_float <= 100.0:
                    prob_float = prob_float / 100.0
                probabilities[str(cls_name)] = prob_float
            except (TypeError, ValueError):
                continue

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
    }


def build_explanation_prompt(
    result: Dict[str, Any],
    language: str = "english"
) -> Tuple[str, str]:
    """
    Builds the system and user prompts for the LLM explainer.
    The prompt contains ONLY structured text and numbers.

    Args:
        result: Validated classifier result dictionary.
        language: "english", "hindi", or "hinglish".

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    probabilities = result.get("probabilities", {})

    prob_str = ""
    if probabilities:
        items = [f"- {cls}: {p * 100:.2f}%" for cls, p in sorted(probabilities.items())]
        prob_str = "Class probability distribution:\n" + "\n".join(items)

    class_desc = CLASS_DESCRIPTIONS.get(
        predicted_class, "Classified dataset category."
    )

    lang_lower = language.lower().strip()
    is_hinglish = lang_lower in ["hindi", "hinglish"]

    system_prompt = (
        "You are the NeuroScan AI Educational Result Explainer, an AI assistant for an undergraduate "
        "engineering project on 2D brain MRI slice classification.\n\n"
        "STRICT SAFETY & ETHICAL BOUNDARIES:\n"
        "1. You must explain the provided model result in plain, clear, educational language.\n"
        "2. This model is an educational student project, NOT a medical diagnostic device.\n"
        "3. You must NEVER claim that the patient has or does not have Alzheimer's disease or dementia.\n"
        "4. A model confidence score is simply mathematical output probability on a single 2D image slice; "
        "it is NOT a clinical probability of human illness.\n"
        "5. High confidence does NOT prove disease, and low confidence does NOT rule out disease.\n"
        "6. Do NOT invent patient history, symptoms, clinical findings, or radiological details.\n"
        "7. Do NOT alter, override, or replace the predicted class or scores.\n"
        "8. State the key limitations: 2D slice dataset, no patient identifiers available (patient-level "
        "leakage cannot be ruled out), and clinical evaluation by a medical doctor is essential.\n"
        "9. Never recommend medication, therapies, or clinical treatment plans."
    )

    if is_hinglish:
        user_prompt = f"""Kripya neeche diye gaye classifier result ka ek clear aur saral Hinglish (conversational Hindi written in English/Latin script) explanation taiyar karein.

CLASSIFIER RESULT (Text metadata only):
- Predicted Dataset Class: {predicted_class}
- Model Confidence: {confidence * 100:.2f}%
{prob_str}
- Dataset Context for this class: {class_desc}

AAPKE EXPLANATION MEIN YEH BATAEIN:
1. Yeh predicted class is dataset ke context mein kya darshati hai (saral shabdon mein).
2. Confidence score ({confidence * 100:.2f}%) ka vastavik matlab kya hai: Yeh model ka 4 classes ke beech ka mathematical calculation hai, kisi vyakti ki bimari hone ki probability nahi hai.
3. Important Limitations:
   - Yeh ek college engineering project hai, medical diagnosis tool nahi hai.
   - Model sirf ek 2D axial MRI slice dekh raha hai.
   - Public dataset mein patient ID na hone ke karan patient-level leakage ko rule out nahi kiya ja sakta.
   - Kisi bhi swasthya sambandhi sawal ke liye qualified neurologist ya doctor se consult karna anivarya hai.

Tone friendly, respectful, and educational honi chahiye. Kisi bhi tarah ki medical advice ya patient ke baare mein man-ghadant batein na likhein."""
    else:
        user_prompt = f"""Please provide a clear, educational explanation of the following classifier result in simple, student-friendly English.

CLASSIFIER RESULT (Text metadata only):
- Predicted Dataset Class: {predicted_class}
- Model Confidence: {confidence * 100:.2f}%
{prob_str}
- Dataset Context for this class: {class_desc}

YOUR EXPLANATION MUST INCLUDE:
1. What the predicted class represents within this educational dataset.
2. What the confidence score ({confidence * 100:.2f}%) actually means: It is the model's mathematical output probability across the four dataset categories for this single 2D slice, NOT the clinical probability of a person having Alzheimer's disease.
3. Essential Project & Dataset Limitations:
   - This is an educational college engineering model, not a medical diagnostic system.
   - The model analyzes an isolated 2D slice without patient clinical records.
   - Patient-level data leakage cannot be ruled out because patient identifiers are unavailable in this public dataset.
   - Remind the reader that high confidence does not prove disease, and real medical assessment requires qualified physicians.

Keep the explanation clear, objective, and supportive. Do not include treatment advice or invented clinical findings."""

    return system_prompt, user_prompt


def build_fallback_explanation(
    result: Dict[str, Any],
    language: str = "english",
    reason: str = ""
) -> str:
    """
    Generates a deterministic, rule-based fallback explanation when the live LLM
    API is unavailable (e.g., missing API key, network error, or invalid response).
    Never fabricates an LLM response.
    """
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    probabilities = result.get("probabilities", {})

    prob_lines = []
    if probabilities:
        for cls, p in sorted(probabilities.items()):
            prob_lines.append(f"  - {cls}: {p * 100:.1f}%")
    prob_text = "\n".join(prob_lines) if prob_lines else "  - No detailed class distribution provided."

    class_desc = CLASS_DESCRIPTIONS.get(
        predicted_class, "Standard dataset category."
    )

    lang_lower = language.lower().strip()
    is_hinglish = lang_lower in ["hindi", "hinglish"]

    reason_note = f" (Reason: {reason})" if reason else ""

    if is_hinglish:
        return f"""[Educational Summary -- Automated Fallback Notice]
Live LLM explanation uplabdh nahi hai{reason_note}. Neeche diye gaye points model ka standard educational summary hain:

1. Predicted Dataset Class: {predicted_class}
   - Dataset mein iska matlab: {class_desc}

2. Model Confidence Score: {confidence * 100:.1f}%
   - Yeh score model ka iss 2D slice par mathematical calculation hai.
   - Yeh kisi vyakti ko Alzheimer hone ya na hone ki clinical probability NAHI hai.
   - High confidence se bimari prove nahi hoti, aur low score se bimari rule-out nahi hoti.

3. Class Distribution:
{prob_text}

4. Important Limitations & Medical Disclaimer:
   - Yeh ek student engineering project hai, certified medical diagnosis nahi hai.
   - Dataset mein patient IDs na hone ki wajah se patient-level data leakage sambhav ho sakta hai.
   - Kisi bhi medical concern ke liye hamesha qualified doctor ya neurologist se salah lein."""
    else:
        return f"""[Educational Summary -- Automated Fallback Notice]
Live LLM explanation is currently unavailable{reason_note}. Below is the verified educational summary of the classifier output:

1. Predicted Dataset Class: {predicted_class}
   - Dataset Context: {class_desc}

2. Model Confidence Score: {confidence * 100:.1f}%
   - This score represents the model's mathematical output score across the 4 dataset categories for this single 2D slice.
   - This is NOT the clinical probability that an individual has Alzheimer's disease.
   - A high score does not confirm medical condition, and a lower score does not rule it out.

3. Class Probability Distribution:
{prob_text}

4. Important Limitations & Medical Disclaimer:
   - This is an educational college project tool, not a clinical diagnostic system.
   - Analysis is performed on standalone 2D slices without patient medical history.
   - Patient-level data leakage cannot be ruled out because patient identifiers are unavailable in the source dataset.
   - Always consult qualified healthcare professionals and certified neurologists for real medical evaluations."""


def explain_result(
    classifier_result: Dict[str, Any],
    language: str = "english",
    provider: Optional[str] = None,
    use_fallback_on_error: bool = True,
    timeout: int = 35,
) -> Dict[str, Any]:
    """
    Main entry point for generating an educational explanation from a structured
    classifier result.

    Guarantees:
    - Never passes MRI image data or file paths.
    - Preserves input 'predicted_class', 'confidence', and 'probabilities' untouched.
    - Returns structured explanation dictionary with clear execution status.

    Args:
        classifier_result: Dictionary with 'predicted_class', 'confidence', and optional 'probabilities'.
        language: Target explanation language ("english", "hindi", or "hinglish").
        provider: Optional provider override ("gemini" or "openai").
        use_fallback_on_error: If True, returns a safe rule-based fallback when the LLM
                               provider is unavailable, rather than crashing.
        timeout: Network timeout in seconds.

    Returns:
        Dictionary containing:
        - predicted_class: Preserved from classifier
        - confidence: Preserved from classifier
        - probabilities: Preserved from classifier
        - explanation: Generated LLM text or safe fallback text
        - language: Requested language
        - status: "success" or "fallback"
        - provider: Provider name attempted
        - error: Error description if fallback was used, otherwise None
    """
    # 1. Validate classifier inputs (rejects image data and invalid formats)
    validated = validate_classifier_result(classifier_result)

    # 2. Determine provider and build prompt
    selected_provider = (provider or DEFAULT_LLM_PROVIDER).lower().strip()
    system_prompt, user_prompt = build_explanation_prompt(validated, language=language)

    explanation_text = ""
    status = "success"
    error_msg = None

    # 3. Call configured LLM provider
    try:
        explanation_text = call_llm(
            prompt=user_prompt,
            system_prompt=system_prompt,
            provider=selected_provider,
            timeout=timeout,
        )
    except Exception as e:
        if not use_fallback_on_error:
            raise e
        status = "fallback"
        error_msg = str(e)
        explanation_text = build_fallback_explanation(
            validated, language=language, reason=error_msg
        )

    # 4. Return structured result — original classifier fields are NEVER overridden
    return {
        "predicted_class": validated["predicted_class"],
        "confidence": validated["confidence"],
        "probabilities": validated["probabilities"],
        "explanation": explanation_text,
        "language": language,
        "status": status,
        "provider": selected_provider,
        "error": error_msg,
    }
