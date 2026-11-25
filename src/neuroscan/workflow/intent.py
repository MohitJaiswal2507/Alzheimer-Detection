"""
NeuroScan AI — Intent Classification & Routing
Classifies natural language queries into distinct operational intents
to direct the workflow along appropriate retrieval and safety branches.
"""
from typing import Dict, Any, Optional
import re

# Prohibited medical advice / diagnosis query patterns
MEDICAL_ADVICE_PATTERNS = [
    r"\b(do i have|diagnose me|what medicine|prescribe|cure my|my symptoms|treatment for me|am i demented)\b",
    r"\b(kya mujhe alzheimer|dawai batayein|ilaj kya hai|meri dawa|kya yeh bimari hai)\b",
]

# Classification result explanation patterns
MRI_EXPLANATION_PATTERNS = [
    r"\b(explain this result|why is it classified|predicted class|confidence score|prediction explanation|model result)\b",
    r"\b(why did the model predict|classifier output|classification result|scan result)\b",
]

# Grad-CAM explainability patterns
GRADCAM_PATTERNS = [
    r"\b(grad-cam|gradcam|heatmap|saliency|attention map|visual explanation|where is the model looking)\b",
]

# Dataset & model limitations patterns
LIMITATION_PATTERNS = [
    r"\b(data leakage|patient-level|patient leakage|imbalance|minority class|limitations?|2d vs 3d|confidence vs probability)\b",
]

# Educational query patterns (clinical/patient education + technical concepts)
EDUCATIONAL_PATTERNS = [
    r"\b(what is alzheimer|alzheimer'?s|dementia|amyloid|tau|plaque|hippocampus|ventricle|brain mri|axial|t1|t2)\b",
    r"\b(efficientnet|compound scaling|mbconv|transfer learning|dataset categories|four categories|4 categories)\b",
    r"\b(mild_demented|moderate_demented|non_demented|very_mild_demented)\b",
    r"\b(symptoms?|caregiver|daily care|home safe|safety|lifestyle|brain health|normal aging|memory loss|memory)\b",
    r"\b(cognitive decline|neurologist|checkup|next steps?|monitor|early signs?|prevent|support|loved one|what should|what to do)\b",
]


def classify_intent(query: str, classifier_context: Optional[Dict[str, Any]] = None) -> str:
    """
    Identifies the user's intent from query text and optional classifier context.

    Returns one of:
        - "medical_advice"
        - "mri_explanation"
        - "gradcam_explanation"
        - "limitations"
        - "educational_question"
        - "out_of_scope"
    """
    q_lower = query.lower().strip()

    # 1. Immediate priority: Medical advice / personal diagnosis detection
    if any(re.search(pat, q_lower) for pat in MEDICAL_ADVICE_PATTERNS):
        return "medical_advice"

    # 2. Classifier result explanation (if explicit query or if classifier_context is present and relevant)
    if any(re.search(pat, q_lower) for pat in MRI_EXPLANATION_PATTERNS):
        return "mri_explanation"

    if classifier_context and any(k in q_lower for k in ["this scan", "my scan", "uploaded image", "prediction", "confidence"]):
        return "mri_explanation"

    # 3. Grad-CAM visual explainability
    if any(re.search(pat, q_lower) for pat in GRADCAM_PATTERNS):
        return "gradcam_explanation"

    # 4. Limitations & methodology
    if any(re.search(pat, q_lower) for pat in LIMITATION_PATTERNS):
        return "limitations"

    # 5. Core educational inquiries
    if any(re.search(pat, q_lower) for pat in EDUCATIONAL_PATTERNS):
        return "educational_question"

    # 6. Fallback: Out of scope
    return "out_of_scope"
