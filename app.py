"""
NeuroScan AI — Educational Brain MRI Classification & Explainability Prototype
Interactive Streamlit demonstration application for college project presentation.
High-end medical AI workstation aesthetic with double-bezel cards, custom probability
meters, instant sample demo scans, Grad-CAM visualization, and LLM explanation.
"""
import io
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Import NeuroScan AI internal modules
from src.neuroscan.config import (
    CLASS_NAMES,
    CLASS_DESCRIPTIONS,
    PROJECT_ROOT,
    ARTIFACTS_DIR,
    DEFAULT_GEMINI_MODEL as GEMINI_MODEL,
)
from src.neuroscan.vision.inference import (
    load_trained_model,
    preprocess_image,
    predict_mri,
    get_device,
)
from src.neuroscan.explainability.gradcam import (
    generate_gradcam,
    create_gradcam_figure,
)
from src.neuroscan.llm.explainer import explain_result
from src.neuroscan.rag import (
    ask_knowledge_assistant,
    get_dependency_status,
    is_vector_store_available,
)
from src.neuroscan.workflow import (
    run_neuroscan_workflow,
    is_langgraph_available,
)

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroScan AI — Brain MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# High-End Presentation & Medical Console Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts: Plus Jakarta Sans & JetBrains Mono */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    code, pre, .mono-text {
        font-family: 'JetBrains Mono', monospace !important;
    }
    h1, h2, h3 {
        text-wrap: balance;
    }

    /* Browser Surfaces: Selection, Scrollbars, Focus Rings, Tabular Numerals */
    ::selection {
        background: rgba(14, 165, 233, 0.28);
        color: #f8fafc;
    }
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0b0f19;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 9999px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }
    button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible {
        outline: 2px solid #38bdf8 !important;
        outline-offset: 2px !important;
    }
    .metric-kpi-value, table td, table th, .prob-header span:last-child {
        font-variant-numeric: tabular-nums;
    }

    /* Global Container Adjustments */
    .block-container {
        padding-top: 2.6rem !important;
        padding-bottom: 3rem !important;
        max-width: 1340px !important;
    }

    /* Sidebar Background, Borders & Radio Navigation */
    section[data-testid="stSidebar"] {
        background-color: #070b14 !important;
        border-right: 1px solid rgba(148, 163, 184, 0.12) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 0.42rem 0.65rem !important;
        border-radius: 8px !important;
        margin-bottom: 0.2rem !important;
        transition: background-color 0.18s ease !important;
        white-space: nowrap !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(148, 163, 184, 0.08) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        white-space: nowrap !important;
    }

    /* Primary Action Button (Restrained Medical Cyan) */
    div.stButton > button[kind="primary"], div.stButton > button:first-child {
        background: #0284c7 !important;
        color: #ffffff !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.4rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 2px 10px rgba(2, 132, 199, 0.28) !important;
        transition: background-color 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease !important;
    }
    div.stButton > button[kind="primary"]:hover, div.stButton > button:first-child:hover {
        background: #0ea5e9 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(14, 165, 233, 0.35) !important;
        border-color: rgba(125, 211, 252, 0.7) !important;
    }
    div.stButton > button[kind="primary"]:active, div.stButton > button:first-child:active {
        transform: translateY(0px) scale(0.99) !important;
    }

    /* Secondary / Preset Buttons */
    div.stButton > button[kind="secondary"] {
        background: #111827 !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 9px !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        min-height: 40px !important;
        white-space: nowrap !important;
        transition: background-color 0.18s ease, border-color 0.18s ease, color 0.18s ease !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: #1e293b !important;
        border-color: rgba(56, 189, 248, 0.55) !important;
        color: #f8fafc !important;
    }
    div.stButton > button[kind="secondary"]:active {
        transform: scale(0.985) !important;
    }

    /* File Uploader Container */
    div[data-testid="stFileUploader"] {
        background: rgba(17, 24, 39, 0.7);
        border: 1px dashed rgba(56, 189, 248, 0.35);
        border-radius: 12px;
        padding: 0.8rem;
        transition: border-color 0.2s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(56, 189, 248, 0.7);
    }

    /* Clean Single-Layer Surface Cards */
    .bezel-card {
        background: #111827;
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 14px;
        padding: 1.35rem;
        margin-bottom: 1.2rem;
    }
    .bezel-card-highlight {
        background: #111827;
        border: 1px solid rgba(56, 189, 248, 0.28);
        border-radius: 14px;
        padding: 1.35rem;
        margin-bottom: 1.2rem;
    }

    /* KPI Metric Cards (Dashboard) */
    .metric-kpi-card {
        background: #111827;
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 12px;
        padding: 1.15rem 1.25rem;
    }
    .metric-kpi-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
    }
    .metric-kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }
    .metric-kpi-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.3rem;
    }

    /* Badges & Status Tags */
    .badge-live {
        background-color: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.28);
        padding: 0.25rem 0.7rem;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
    }
    .badge-fallback {
        background-color: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.28);
        padding: 0.25rem 0.7rem;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
    }

    /* Medical Disclaimer Banners (1px hairline border, no thick side stripes) */
    .disclaimer-box {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(129, 140, 248, 0.28);
        padding: 0.9rem 1.15rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #cbd5e1;
        font-size: 0.85rem;
        line-height: 1.55;
    }
    .warning-box {
        background: rgba(127, 29, 29, 0.22);
        border: 1px solid rgba(248, 113, 113, 0.32);
        padding: 0.9rem 1.15rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #fecaca;
        font-size: 0.85rem;
        line-height: 1.55;
    }

    /* Custom Probability Bar Styles */
    .prob-row {
        margin-bottom: 0.75rem;
    }
    .prob-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.3rem;
        font-size: 0.85rem;
    }
    .prob-track {
        background: rgba(30, 41, 59, 0.85);
        border-radius: 9999px;
        height: 8px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }
    .prob-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.5s cubic-bezier(0.2, 0.8, 0.2, 1);
    }

    /* Markdown Tables in Chat */
    div[data-testid="stChatMessage"] table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.9rem 0;
        font-size: 0.86rem;
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 8px;
        overflow: hidden;
    }
    div[data-testid="stChatMessage"] th {
        background: rgba(30, 41, 59, 0.9);
        color: #38bdf8;
        font-weight: 600;
        padding: 8px 12px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.16);
        text-align: left;
    }
    div[data-testid="stChatMessage"] td {
        padding: 8px 12px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        color: #cbd5e1;
    }
    div[data-testid="stChatMessage"] tr:last-child td {
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Loads and caches the trained EfficientNet-B0 model instance."""
    ckpt_path = ARTIFACTS_DIR / "best_efficientnet_b0.pth"
    device = get_device()
    model, idx_to_class, class_to_idx, metadata = load_trained_model(
        checkpoint_path=ckpt_path,
        device=device,
    )
    return model, idx_to_class, class_to_idx, metadata, device


# -----------------------------------------------------------------------------
# Instant Demo Presets Dictionary
# -----------------------------------------------------------------------------
PRESET_SAMPLES = {
    "Moderate_Demented": {
        "title": "Moderate Demented",
        "badge": "1.0% Dataset Minority (N=64)",
        "path": PROJECT_ROOT / "data" / "Alzheimer (Preprocessed Data)" / "Moderate_Demented" / "moderate.jpg",
        "color": "#f43f5e",
        "icon": "🔴",
    },
    "Mild_Demented": {
        "title": "Mild Demented",
        "badge": "14.0% of Dataset (N=896)",
        "path": PROJECT_ROOT / "data" / "Alzheimer (Preprocessed Data)" / "Mild_Demented" / "mild.jpg",
        "color": "#f59e0b",
        "icon": "🟠",
    },
    "Very_Mild_Demented": {
        "title": "Very Mild Demented",
        "badge": "35.0% of Dataset (N=2,240)",
        "path": PROJECT_ROOT / "data" / "Alzheimer (Preprocessed Data)" / "Very_Mild_Demented" / "verymild.jpg",
        "color": "#38bdf8",
        "icon": "🔵",
    },
    "Non_Demented": {
        "title": "Non-Demented",
        "badge": "50.0% of Dataset (N=3,200)",
        "path": PROJECT_ROOT / "data" / "Alzheimer (Preprocessed Data)" / "Non_Demented" / "non.jpg",
        "color": "#10b981",
        "icon": "🟢",
    },
}


# -----------------------------------------------------------------------------
# Severity-Based Suggested Client Questions (Patient & Caregiver Education)
# -----------------------------------------------------------------------------
SEVERITY_QUESTIONS = {
    "Moderate_Demented": [
        ("Common moderate symptoms", "What symptoms are common in moderate Alzheimer's disease?"),
        ("Daily care & caregiver support", "What daily care and caregiver support steps help most in moderate dementia?"),
        ("Questions for a neurologist", "What questions should we ask a neurologist for moderate cognitive decline?"),
        ("Home safety & calm routines", "How can we keep a loved one safe and calm at home with moderate dementia?"),
    ],
    "Mild_Demented": [
        ("Early-to-mild symptoms", "What are typical early-to-mild Alzheimer's symptoms?"),
        ("Medical checkup & next steps", "What next steps and medical tests should we discuss with a doctor?"),
        ("Brain-healthy lifestyle habits", "What lifestyle habits support brain health in mild cognitive decline?"),
        ("Daily memory routines", "How can family members help with daily memory routines?"),
    ],
    "Very_Mild_Demented": [
        ("Normal aging vs. early decline", "What is the difference between normal aging and very mild cognitive decline?"),
        ("Early signs to monitor", "What early memory signs should we monitor over time?"),
        ("Preventive brain-care habits", "What healthy lifestyle and brain-care steps are recommended?"),
        ("When to see a neurologist", "When should someone schedule a neurological checkup?"),
    ],
    "Non_Demented": [
        ("What Non-Demented means", "What does a Non-Demented scan result mean for brain health?"),
        ("Habits to protect memory", "What lifestyle habits help prevent cognitive decline?"),
        ("Why 2D MRI isn't a checkup", "Why does a normal 2D MRI not replace a full medical checkup?"),
        ("Early warning signs to know", "What early warning signs of memory loss should anyone watch for?"),
    ],
}


# -----------------------------------------------------------------------------
# Sidebar Navigation (Reliable First-Click Programmatic Switching)
# -----------------------------------------------------------------------------
NAV_OPTIONS = ["📊 Dashboard", "🔬 MRI Analysis", "💡 Knowledge Assistant"]

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "🔬 MRI Analysis"


def _on_nav_change():
    st.session_state["active_page"] = st.session_state["nav_radio_widget"]


st.session_state["nav_radio_widget"] = st.session_state["active_page"]

with st.sidebar:
    # High-End Brand Banner
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; padding: 0.2rem 0 1rem 0;">
        <div style="font-size: 1.8rem; background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 16px rgba(56, 189, 248, 0.2);">
            🧠
        </div>
        <div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.01em;">NeuroScan AI</div>
            <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 500;">Clinical AI Workstation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.radio(
        "Navigation",
        options=NAV_OPTIONS,
        key="nav_radio_widget",
        on_change=_on_nav_change,
    )

page = st.session_state["active_page"]


# =============================================================================
# PAGE 1: DASHBOARD
# =============================================================================
if page == "📊 Dashboard":
    st.markdown("""
    <h1 style="font-size: 2.05rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.025em; line-height: 1.15; margin: 0 0 0.85rem 0;">
        Project Overview
    </h1>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Asymmetric Bento Top Stage: [1.3, 1.0]
    # -------------------------------------------------------------------------
    bento_left, bento_right = st.columns([1.3, 1.0], gap="large")

    with bento_left:
        # Primary Benchmark Hero Readout + 3-Metric Sub-Telemetry
        st.markdown(
            '<div style="background: #0d1322; border: 1px solid rgba(148, 163, 184, 0.14); border-radius: 16px; padding: 8px; height: 100%;">'
            '<div style="background: #111827; border: 1px solid rgba(56, 189, 248, 0.22); border-radius: 11px; padding: 1.45rem 1.5rem;">'
            '<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.25rem;">'
            '<div>'
            '<div style="font-size: 0.7rem; font-weight: 700; color: #94a3b8; letter-spacing: 0.09em; text-transform: uppercase;">HELD-OUT TEST ACCURACY</div>'
            '<div style="font-size: 3.1rem; font-weight: 800; font-family: \'JetBrains Mono\', monospace; color: #38bdf8; line-height: 1.05; margin-top: 0.25rem; font-variant-numeric: tabular-nums;">95.52%</div>'
            '<div style="font-size: 0.8rem; color: #64748b; margin-top: 0.35rem;">917 / 960 correctly classified axial slices</div>'
            '</div>'
            '<span style="font-size: 0.72rem; font-weight: 600; color: #34d399; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); padding: 4px 10px; border-radius: 9999px;">Test Split Verified</span>'
            '</div>'
            '<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem;">'
            '<div style="background: #0b0f19; border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 9px; padding: 0.85rem 1rem;">'
            '<div style="font-size: 0.68rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em;">Macro F1</div>'
            '<div style="font-size: 1.35rem; font-weight: 700; font-family: \'JetBrains Mono\', monospace; color: #f8fafc; margin-top: 0.2rem; font-variant-numeric: tabular-nums;">0.9486</div>'
            '<div style="font-size: 0.7rem; color: #64748b; margin-top: 0.15rem;">4-class balanced</div>'
            '</div>'
            '<div style="background: #0b0f19; border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 9px; padding: 0.85rem 1rem;">'
            '<div style="font-size: 0.68rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em;">Specificity</div>'
            '<div style="font-size: 1.35rem; font-weight: 700; font-family: \'JetBrains Mono\', monospace; color: #34d399; margin-top: 0.2rem; font-variant-numeric: tabular-nums;">98.24%</div>'
            '<div style="font-size: 0.7rem; color: #64748b; margin-top: 0.15rem;">One-vs-rest avg</div>'
            '</div>'
            '<div style="background: #0b0f19; border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 9px; padding: 0.85rem 1rem;">'
            '<div style="font-size: 0.68rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em;">Val F1</div>'
            '<div style="font-size: 1.35rem; font-weight: 700; font-family: \'JetBrains Mono\', monospace; color: #f8fafc; margin-top: 0.2rem; font-variant-numeric: tabular-nums;">0.9780</div>'
            '<div style="font-size: 0.7rem; color: #64748b; margin-top: 0.15rem;">Epoch 15 ckpt</div>'
            '</div>'
            '</div></div></div>',
            unsafe_allow_html=True,
        )

    with bento_right:
        # Visual Dataset Class Proportions Card (N = 6,400)
        dist_rows = [
            ("Non Demented", "50.0%", 50.0, "3,200", "#10b981"),
            ("Very Mild Demented", "35.0%", 35.0, "2,240", "#38bdf8"),
            ("Mild Demented", "14.0%", 14.0, "896", "#f59e0b"),
            ("Moderate Demented", "1.0%", 2.5, "64", "#f43f5e"),
        ]
        dist_html = ""
        for d_name, d_pct, d_bar_w, d_cnt, d_col in dist_rows:
            dist_html += (
                f'<div style="margin-bottom: 0.72rem;">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; margin-bottom: 4px;">'
                f'<span style="color: #e2e8f0; font-weight: 500; display: inline-flex; align-items: center; gap: 7px;">'
                f'<span style="width: 7px; height: 7px; border-radius: 50%; background: {d_col};"></span>{d_name}</span>'
                f'<span style="font-family: \'JetBrains Mono\', monospace; color: #94a3b8; font-size: 0.76rem; font-variant-numeric: tabular-nums;">'
                f'<b style="color: #f8fafc;">{d_cnt}</b> &bull; {d_pct}</span>'
                f'</div>'
                f'<div style="background: rgba(30, 41, 59, 0.9); border-radius: 9999px; height: 6px; overflow: hidden;">'
                f'<div style="width: {d_bar_w}%; height: 100%; background: {d_col}; border-radius: 9999px;"></div>'
                f'</div></div>'
            )

        st.markdown(
            f'<div style="background: #0d1322; border: 1px solid rgba(148, 163, 184, 0.14); border-radius: 16px; padding: 8px;">'
            f'<div style="background: #111827; border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 11px; padding: 1.35rem 1.4rem;">'
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">'
            f'<span style="font-size: 0.88rem; font-weight: 700; color: #f8fafc;">Dataset Class Proportions</span>'
            f'<span style="font-size: 0.72rem; font-family: \'JetBrains Mono\', monospace; color: #64748b;">N = 6,400</span>'
            f'</div>'
            f'{dist_html}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Full-Width Held-Out Test Set Performance Matrix (N = 960)
    # -------------------------------------------------------------------------
    matrix_rows = [
        ("Non Demented", "#10b981", "96.85%", "96.04%", "0.9644", "96.88%", "480"),
        ("Very Mild Demented", "#38bdf8", "96.04%", "93.75%", "0.9488", "97.92%", "336"),
        ("Mild Demented", "#f59e0b", "89.86%", "98.52%", "0.9399", "98.18%", "135"),
        ("Moderate Demented", "#f43f5e", "100.00%", "88.89%", "0.9412", "100.00%", "9"),
    ]
    tr_html = ""
    for m_cls, m_col, m_prec, m_rec, m_f1, m_spec, m_n in matrix_rows:
        tr_html += (
            f'<tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.08);">'
            f'<td style="padding: 11px 14px; color: #f8fafc; font-weight: 600; display: flex; align-items: center; gap: 8px;">'
            f'<span style="width: 8px; height: 8px; border-radius: 50%; background: {m_col};"></span>{m_cls}</td>'
            f'<td style="padding: 11px 14px; text-align: right; font-family: \'JetBrains Mono\', monospace; color: #cbd5e1;">{m_prec}</td>'
            f'<td style="padding: 11px 14px; text-align: right; font-family: \'JetBrains Mono\', monospace; color: #cbd5e1;">{m_rec}</td>'
            f'<td style="padding: 11px 14px; text-align: right; font-family: \'JetBrains Mono\', monospace; color: #38bdf8; font-weight: 700;">{m_f1}</td>'
            f'<td style="padding: 11px 14px; text-align: right; font-family: \'JetBrains Mono\', monospace; color: #34d399;">{m_spec}</td>'
            f'<td style="padding: 11px 14px; text-align: right; font-family: \'JetBrains Mono\', monospace; color: #94a3b8;">{m_n}</td>'
            f'</tr>'
        )

    st.markdown(
        f'<div style="background: #0d1322; border: 1px solid rgba(148, 163, 184, 0.14); border-radius: 16px; padding: 8px; margin-bottom: 1.2rem;">'
        f'<div style="background: #111827; border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 11px; overflow: hidden;">'
        f'<div style="padding: 1rem 1.25rem; border-bottom: 1px solid rgba(148, 163, 184, 0.1); display: flex; justify-content: space-between; align-items: center;">'
        f'<span style="font-size: 0.92rem; font-weight: 700; color: #f8fafc;">Held-Out Test Set Performance Breakdown</span>'
        f'<span style="font-size: 0.72rem; font-family: \'JetBrains Mono\', monospace; color: #64748b;">N = 960 Slices (15% Stratified Split)</span>'
        f'</div>'
        f'<table style="width: 100%; border-collapse: collapse; font-size: 0.85rem; font-variant-numeric: tabular-nums;">'
        f'<thead><tr style="background: #0b0f19; border-bottom: 1px solid rgba(148, 163, 184, 0.12);">'
        f'<th style="padding: 10px 14px; text-align: left; font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em;">Dataset Class</th>'
        f'<th style="padding: 10px 14px; text-align: right; font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em;">Precision</th>'
        f'<th style="padding: 10px 14px; text-align: right; font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em;">Recall</th>'
        f'<th style="padding: 10px 14px; text-align: right; font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em;">F1-Score</th>'
        f'<th style="padding: 10px 14px; text-align: right; font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em;">Specificity</th>'
        f'<th style="padding: 10px 14px; text-align: right; font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em;">Test N</th>'
        f'</tr></thead>'
        f'<tbody>{tr_html}</tbody>'
        f'</table></div></div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE 2: MRI ANALYSIS (CORE PRESENTATION DEMO)
# =============================================================================
elif page == "🔬 MRI Analysis":
    # Load model safely
    try:
        model, idx_to_class, class_to_idx, meta, device = get_cached_model()
    except Exception as e:
        st.error(f"Cannot load trained model checkpoint: {e}")
        st.stop()

    # Clean Header Row (Title + Language Selector only)
    hdr_col, lang_col = st.columns([3.0, 1.0], gap="medium")
    with hdr_col:
        st.markdown("""
        <h1 style="font-size: 2.05rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.025em; line-height: 1.15; margin: 0 0 0.25rem 0;">
            2D Brain MRI Analysis
        </h1>
        """, unsafe_allow_html=True)
    with lang_col:
        language_choice = st.selectbox(
            "Language",
            options=["English", "Hindi / Hinglish"],
            index=0,
        )
        lang_code = "hinglish" if "Hinglish" in language_choice else "english"

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Asymmetric 2-Column Upload Workstation: [1.0, 1.35]
    # -------------------------------------------------------------------------
    left_ws, right_ws = st.columns([1.0, 1.35], gap="large")

    with left_ws:
        uploaded_file = st.file_uploader(
            "Upload 2D Axial Brain MRI (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None:
            file_sig = (uploaded_file.name, uploaded_file.size)
            if st.session_state.get("last_uploaded_file_sig") != file_sig:
                try:
                    st.session_state["active_image"] = Image.open(uploaded_file)
                    st.session_state["active_source_name"] = uploaded_file.name
                    st.session_state["last_uploaded_file_sig"] = file_sig
                    # Clear previous scan analysis & Knowledge Assistant context for the new MRI
                    st.session_state.pop("last_analysis", None)
                    st.session_state.pop("active_classifier_context", None)
                    st.session_state.pop("rag_messages", None)
                    st.session_state.pop("pending_rag_question", None)
                except Exception as e:
                    st.error(f"Invalid image format: {e}")
        else:
            # If user cleared the uploaded file with the 'x' button, reset active scan & context
            if st.session_state.get("last_uploaded_file_sig") is not None:
                st.session_state.pop("last_uploaded_file_sig", None)
                st.session_state.pop("active_image", None)
                st.session_state.pop("active_source_name", None)
                st.session_state.pop("last_analysis", None)
                st.session_state.pop("active_classifier_context", None)
                st.session_state.pop("rag_messages", None)
                st.session_state.pop("pending_rag_question", None)

        active_img = st.session_state.get("active_image", None)
        active_name = st.session_state.get("active_source_name", None)

        if active_img is not None:
            w, h = active_img.width, active_img.height
            st.markdown(
                f'<div style="background: #0d1322; border: 1px solid rgba(148, 163, 184, 0.14); border-radius: 16px; padding: 8px 8px 0 8px; margin-top: 0.4rem;">'
                f'<div style="background: #060913; border: 1px solid rgba(148, 163, 184, 0.08); border-bottom: none; border-radius: 10px 10px 0 0; padding: 9px 14px; display: flex; justify-content: space-between; align-items: center;">'
                f'<div style="display: flex; align-items: center; gap: 8px;">'
                f'<span style="width: 7px; height: 7px; border-radius: 50%; background: #10b981;"></span>'
                f'<span style="font-size: 0.72rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.08em;">AXIAL BRAIN MRI</span>'
                f'</div>'
                f'<span style="font-size: 0.7rem; font-family: \'JetBrains Mono\', monospace; color: #94a3b8;">{w}&times;{h}px</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            st.image(active_img, use_container_width=True)

            st.markdown(
                f'<div style="background: #0d1322; border: 1px solid rgba(148, 163, 184, 0.14); border-top: none; border-radius: 0 0 16px 16px; padding: 0 8px 8px 8px; margin-bottom: 0.9rem;">'
                f'<div style="background: #060913; border: 1px solid rgba(148, 163, 184, 0.08); border-top: none; border-radius: 0 0 10px 10px; padding: 7px 14px; font-size: 0.72rem; color: #64748b;">'
                f'File: <b style="color: #cbd5e1;">{active_name}</b>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            analyze_button = st.button("Analyze MRI", type="primary", use_container_width=True)

            if analyze_button:
                with st.spinner("Analyzing brain MRI scan..."):
                    input_tensor = preprocess_image(active_img, device=device)
                    result = predict_mri(model, input_tensor, idx_to_class)
                    pred_class = result["predicted_class"]
                    conf = result["confidence"]
                    probs = result["probabilities"]
                    heatmap = generate_gradcam(model, input_tensor, target_class=result["predicted_index"])
                    explanation_res = explain_result(
                        {
                            "predicted_class": pred_class,
                            "confidence": conf,
                            "probabilities": probs,
                        },
                        language=lang_code,
                    )
                st.session_state["last_analysis"] = {
                    "pil_image": active_img,
                    "result": result,
                    "heatmap": heatmap,
                    "explanation_res": explanation_res,
                    "lang_code": lang_code,
                    "source_name": active_name,
                }
                # Immediately update Knowledge Assistant context for the newly analyzed MRI
                st.session_state["active_classifier_context"] = {
                    "predicted_class": pred_class,
                    "confidence": conf,
                    "probabilities": probs,
                }
        else:
            st.markdown(
                '<div style="background: #0d1322; border: 1px solid rgba(148, 163, 184, 0.14); border-radius: 16px; padding: 8px; margin-top: 0.4rem;">'
                '<div style="background: #060913; border: 1px dashed rgba(148, 163, 184, 0.18); border-radius: 11px; padding: 4.5rem 1.5rem; text-align: center;">'
                '<div style="font-size: 2rem; margin-bottom: 0.6rem;">🧠</div>'
                '<div style="font-size: 0.95rem; font-weight: 600; color: #cbd5e1; margin-bottom: 0.3rem;">No MRI Scan Loaded</div>'
                '<div style="font-size: 0.8rem; color: #64748b;">Upload a JPG, JPEG, or PNG axial brain slice above to begin.</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )

    with right_ws:
        if "last_analysis" in st.session_state:
            data = st.session_state["last_analysis"]
            res = data["result"]
            heatmap = data["heatmap"]
            exp = data["explanation_res"]

            pred_class = res["predicted_class"]
            conf = res["confidence"]
            probs = res["probabilities"]

            color_map = {
                "Non_Demented": {"accent": "#10b981", "bg": "rgba(16, 185, 129, 0.09)", "border": "rgba(16, 185, 129, 0.28)"},
                "Very_Mild_Demented": {"accent": "#38bdf8", "bg": "rgba(56, 189, 248, 0.09)", "border": "rgba(56, 189, 248, 0.28)"},
                "Mild_Demented": {"accent": "#f59e0b", "bg": "rgba(245, 158, 11, 0.09)", "border": "rgba(245, 158, 11, 0.28)"},
                "Moderate_Demented": {"accent": "#f43f5e", "bg": "rgba(244, 63, 94, 0.10)", "border": "rgba(244, 63, 94, 0.32)"},
            }
            theme = color_map.get(pred_class, {"accent": "#38bdf8", "bg": "rgba(56, 189, 248, 0.09)", "border": "rgba(56, 189, 248, 0.28)"})
            clean_name = pred_class.replace("_", " ")

            minority_note = ""
            if pred_class == "Moderate_Demented":
                minority_note = (
                    f'<div style="margin-top: 0.85rem; padding: 0.6rem 0.85rem; border-radius: 8px; '
                    f'background: rgba(244, 63, 94, 0.12); border: 1px solid rgba(244, 63, 94, 0.28); '
                    f'font-size: 0.76rem; color: #fecdd3;">'
                    f'<b>Minority Class Note:</b> Moderate Demented represents 1% (64 of 6,400) of the dataset.</div>'
                )

            # 1. Double-Bezel Prediction Hero Readout
            st.markdown(
                f'<div style="background: #0d1322; border: 1px solid rgba(148, 163, 184, 0.14); border-radius: 16px; padding: 8px; margin-bottom: 1rem;">'
                f'<div style="background: #111827; border: 1px solid {theme["border"]}; border-radius: 11px; padding: 1.35rem;">'
                f'<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; margin-bottom: 0.9rem;">'
                f'<div><div style="font-size: 0.7rem; font-weight: 700; color: #94a3b8; letter-spacing: 0.09em; text-transform: uppercase;">PREDICTED CLASS</div>'
                f'<div style="font-size: 1.95rem; font-weight: 800; color: {theme["accent"]}; margin-top: 0.2rem; letter-spacing: -0.02em; line-height: 1.1;">{clean_name}</div></div>'
                f'<div style="text-align: right;"><div style="font-size: 0.7rem; font-weight: 700; color: #94a3b8; letter-spacing: 0.09em; text-transform: uppercase;">CONFIDENCE</div>'
                f'<div style="font-size: 1.95rem; font-weight: 800; font-family: \'JetBrains Mono\', monospace; color: #f8fafc; margin-top: 0.2rem; line-height: 1.1; font-variant-numeric: tabular-nums;">{conf * 100:.2f}%</div></div>'
                f'</div>'
                f'<div style="font-size: 0.83rem; color: #cbd5e1; background: {theme["bg"]}; padding: 0.7rem 0.95rem; border-radius: 8px; border: 1px solid {theme["border"]}; line-height: 1.5;">'
                f'{CLASS_DESCRIPTIONS.get(pred_class, "Standard dataset category")}</div>'
                f'{minority_note}'
                f'<div style="margin-top: 0.75rem; font-size: 0.72rem; color: #64748b;">'
                f'Softmax output across 4 dataset classes; not a clinical probability of disease.</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            # 2. Unified Double-Bezel 4-Class Softmax Spectrum
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            bars_html = ""
            for c_name, p_val in sorted_probs:
                pct = p_val * 100
                c_clean = c_name.replace("_", " ")
                is_pred = (c_name == pred_class)
                if c_name == "Moderate_Demented":
                    bar_color = "#f43f5e"
                elif c_name == "Mild_Demented":
                    bar_color = "#f59e0b"
                elif c_name == "Very_Mild_Demented":
                    bar_color = "#38bdf8"
                else:
                    bar_color = "#10b981"

                row_bg = "rgba(56, 189, 248, 0.07)" if is_pred else "transparent"
                row_border = "1px solid rgba(56, 189, 248, 0.25)" if is_pred else "1px solid transparent"
                label_color = "#f8fafc" if is_pred else "#94a3b8"
                val_color = "#38bdf8" if is_pred else "#cbd5e1"
                bars_html += (
                    f'<div style="background: {row_bg}; border: {row_border}; border-radius: 8px; padding: 6px 10px; margin-bottom: 0.55rem;">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; font-size: 0.84rem;">'
                    f'<span style="font-weight: {"700" if is_pred else "500"}; color: {label_color};">{c_clean}</span>'
                    f'<span style="font-family: \'JetBrains Mono\', monospace; font-weight: 700; color: {val_color}; font-variant-numeric: tabular-nums;">{pct:.2f}%</span>'
                    f'</div>'
                    f'<div style="background: rgba(30, 41, 59, 0.9); border-radius: 9999px; height: 7px; overflow: hidden;">'
                    f'<div style="width: {max(pct, 1.5):.2f}%; height: 100%; background: {bar_color}; border-radius: 9999px;"></div>'
                    f'</div></div>'
                )

            st.markdown(
                f'<div style="background: #0d1322; border: 1px solid rgba(148, 163, 184, 0.14); border-radius: 16px; padding: 8px; margin-bottom: 1rem;">'
                f'<div style="background: #111827; border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 11px; padding: 1.2rem;">'
                f'<div style="margin-bottom: 0.85rem;">'
                f'<span style="font-size: 0.88rem; font-weight: 700; color: #f8fafc;">Class Confidence Breakdown</span>'
                f'</div>'
                f'{bars_html}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            # 3. Severity-Tailored Suggested Educational Questions (Navigate to Knowledge Assistant)
            st.markdown(
                f'<div style="font-size: 0.74rem; font-weight: 700; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase; margin: 0.25rem 0 0.5rem 0;">'
                f'Suggested Follow-Up Questions ({clean_name})</div>',
                unsafe_allow_html=True,
            )
            sev_qs = SEVERITY_QUESTIONS.get(pred_class, SEVERITY_QUESTIONS["Mild_Demented"])
            sq_col1, sq_col2 = st.columns(2, gap="small")
            for idx_q, (q_label, q_full) in enumerate(sev_qs):
                target_col = sq_col1 if idx_q % 2 == 0 else sq_col2
                with target_col:
                    if st.button(q_label, key=f"sev_q_{idx_q}", use_container_width=True, help=q_full):
                        st.session_state["pending_rag_question"] = q_full
                        st.session_state["active_classifier_context"] = {
                            "predicted_class": pred_class,
                            "confidence": conf,
                            "probabilities": probs,
                        }
                        st.session_state["active_page"] = "💡 Knowledge Assistant"
                        st.rerun()
        else:
            # Composed Ready State (Clean, non-technical)
            standby_msg = (
                "Click <b>Analyze MRI</b> on the left to view the classification, brain attention map, educational summary, and suggested follow-up questions."
                if active_img is not None
                else "Upload an axial brain MRI slice on the left to view the classification, brain attention map, educational summary, and suggested follow-up questions."
            )
            st.markdown(
                f'<div style="background: #0d1322; border: 1px solid rgba(148, 163, 184, 0.14); border-radius: 16px; padding: 8px;">'
                f'<div style="background: #111827; border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 11px; padding: 2rem 1.5rem;">'
                f'<div style="display: inline-flex; align-items: center; gap: 7px; background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.25); padding: 3px 10px; border-radius: 9999px; font-size: 0.72rem; font-weight: 600; color: #38bdf8; margin-bottom: 1rem;">'
                f'{"Ready to Analyze" if active_img is not None else "Awaiting Scan"}</div>'
                f'<div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.5rem;">Scan Summary Standby</div>'
                f'<div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.55; max-width: 52ch;">'
                f'{standby_msg}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # Full-Width Explainability Stage (Grad-CAM + Clinical Context Explainer)
    # -------------------------------------------------------------------------
    if "last_analysis" in st.session_state:
        data = st.session_state["last_analysis"]
        res = data["result"]
        heatmap = data["heatmap"]
        exp = data["explanation_res"]

        st.markdown("<div style='height: 1.6rem;'></div>", unsafe_allow_html=True)

        # Grad-CAM Section Header
        st.markdown(
            '<div style="margin-bottom: 0.75rem;">'
            '<div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">Brain Attention Map (Grad-CAM)</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        fig = create_gradcam_figure(
            pil_image=data["pil_image"],
            heatmap=heatmap,
            predicted_class=res["predicted_class"],
            confidence=res["confidence"],
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("<div style='height: 1.4rem;'></div>", unsafe_allow_html=True)

        # Educational LLM Result Explanation
        raw_err = str(exp.get("error", "") or "")
        if "429" in raw_err or "quota" in raw_err.lower() or "resource_exhausted" in raw_err.lower():
            short_err = "API Quota Limit Reached"
        elif "timeout" in raw_err.lower():
            short_err = "Request Timeout"
        elif "503" in raw_err:
            short_err = "Service Busy"
        elif raw_err:
            short_err = "API Unavailable"
        else:
            short_err = "Offline Mode"

        status_badge_html = (
            f"<span class='badge-live'>Live LLM ({GEMINI_MODEL})</span>"
            if exp["status"] == "success"
            else f"<span class='badge-fallback'>Local Fallback &bull; {short_err}</span>"
        )

        st.markdown(
            f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem; '
            f'background: #111827; border: 1px solid rgba(148, 163, 184, 0.16); border-radius: 12px; '
            f'padding: 0.9rem 1.25rem; margin-bottom: 0.75rem;">'
            f'<span style="font-size: 1.08rem; font-weight: 700; color: #f8fafc;">Clinical Context Explainer</span>'
            f'{status_badge_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if exp["status"] == "success":
            with st.container(border=True):
                st.markdown(exp["explanation"])
        else:
            p_cls = res["predicted_class"]
            p_clean = p_cls.replace("_", " ")
            p_conf = res["confidence"] * 100
            is_hinglish = data.get("lang_code", "english") == "hinglish"

            if is_hinglish:
                summary_html = (
                    f'<p style="margin: 0 0 0.75rem 0; color: #e2e8f0; line-height: 1.6; font-size: 0.92rem;">'
                    f'EfficientNet-B0 classifier ne is 2D axial brain MRI slice ko <b>{p_clean}</b> category mein '
                    f'<b>{p_conf:.1f}%</b> softmax confidence ke saath classify kiya hai. '
                    f'{CLASS_DESCRIPTIONS.get(p_cls, "")}</p>'
                    f'<p style="margin: 0; color: #94a3b8; line-height: 1.55; font-size: 0.84rem;">'
                    f'<b>Note:</b> Yeh score sirf ek single 2D slice par model ka mathematical output hai, clinical diagnosis nahi. '
                    f'Dataset mein patient IDs available nahi hain, isliye real medical evaluation ke liye hamesha qualified neurologist se consult karein.</p>'
                )
            else:
                summary_html = (
                    f'<p style="margin: 0 0 0.75rem 0; color: #e2e8f0; line-height: 1.6; font-size: 0.92rem;">'
                    f'The EfficientNet-B0 classifier assigned this 2D axial MRI slice to the <b>{p_clean}</b> category '
                    f'with a softmax score of <b>{p_conf:.1f}%</b>. '
                    f'{CLASS_DESCRIPTIONS.get(p_cls, "")}</p>'
                    f'<p style="margin: 0; color: #94a3b8; line-height: 1.55; font-size: 0.84rem;">'
                    f'<b>Scope &amp; Limitations:</b> This score reflects model pattern matching on a single 2D slice rather than a clinical probability of disease. '
                    f'Because source patient identifiers are unavailable, patient-level leakage cannot be ruled out; always consult a qualified clinician for medical assessment.</p>'
                )

            st.markdown(
                f'<div style="background: #0d1322; border: 1px solid rgba(148, 163, 184, 0.14); border-radius: 14px; padding: 8px;">'
                f'<div style="background: #111827; border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 9px; padding: 1.25rem 1.35rem;">'
                f'{summary_html}'
                f'</div></div>',
                unsafe_allow_html=True,
            )


# =============================================================================
# PAGE 3: KNOWLEDGE ASSISTANT (PHASE 7 RAG)
# =============================================================================
elif page == "💡 Knowledge Assistant":
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 2.15rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.025em; line-height: 1.15; margin: 0 0 0.45rem 0;">
            Educational Knowledge Assistant
        </h1>
        <p style="font-size: 0.96rem; color: #94a3b8; margin: 0; max-width: 68ch; line-height: 1.55;">
            Ask questions about Alzheimer's symptoms, dementia stages, daily care, brain health, or MRI basics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize RAG session state
    if "rag_messages" not in st.session_state:
        st.session_state["rag_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am the NeuroScan AI Educational Knowledge Assistant. "
                    "You can ask me about Alzheimer's symptoms, memory changes across stages, "
                    "daily care and lifestyle habits, or what to discuss with a neurologist. How can I help you today?"
                ),
                "sources": [
                    {
                        "title": "Alzheimer's Disease Overview and Neuropathology",
                        "publisher": "National Institute on Aging (NIA) / Mayo Clinic",
                        "source_url": "https://www.nia.nih.gov/health/alzheimers-and-dementia/what-is-alzheimers-disease",
                        "access_date": "2026-10-29",
                        "category": "General Medical Education",
                    }
                ],
                "status": "intro",
            }
        ]

    # Controls row: Language and Clear Button
    rag_ctrl1, rag_ctrl2 = st.columns([3, 1], gap="medium")

    with rag_ctrl1:
        rag_lang_choice = st.selectbox(
            "Response Language",
            options=["English", "Hindi / Hinglish"],
            index=0,
            key="rag_language_select",
        )
        rag_lang_code = "hinglish" if "Hinglish" in rag_lang_choice else "english"

    with rag_ctrl2:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat", use_container_width=True, help="Clear chat history"):
            st.session_state["rag_messages"] = []
            st.rerun()

    # Active MRI Scan Context Banner (when navigated from MRI Analysis)
    active_scan_ctx = st.session_state.get("active_classifier_context")
    if active_scan_ctx:
        ctx_col1, ctx_col2 = st.columns([4, 1], gap="small")
        with ctx_col1:
            ctx_cls = active_scan_ctx["predicted_class"].replace("_", " ")
            ctx_conf = active_scan_ctx["confidence"] * 100
            st.markdown(
                f'<div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); '
                f'border-radius: 10px; padding: 0.55rem 0.9rem; font-size: 0.82rem; color: #e2e8f0; margin-bottom: 0.6rem;">'
                f'Active Scan Context: <b style="color: #38bdf8;">{ctx_cls}</b> ({ctx_conf:.1f}% score) &mdash; '
                f'answers are tailored to symptoms, daily care, and next steps for this category.</div>',
                unsafe_allow_html=True,
            )
        with ctx_col2:
            if st.button("Clear Context", key="btn_clear_ctx", use_container_width=True):
                st.session_state.pop("active_classifier_context", None)
                st.rerun()

    # Suggested Prompts Row (dynamically tailored when scan context exists)
    st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #64748b; letter-spacing: 0.08em; text-transform: uppercase; margin: 0.8rem 0 0.4rem 0;'>SUGGESTED EDUCATIONAL QUESTIONS</div>", unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)

    prompt_to_run = st.session_state.pop("pending_rag_question", None)
    if active_scan_ctx:
        dyn_qs = SEVERITY_QUESTIONS.get(active_scan_ctx["predicted_class"], SEVERITY_QUESTIONS["Mild_Demented"])
    else:
        dyn_qs = [
            ("What is Alzheimer's?", "What is Alzheimer's disease and what are its biological hallmarks?"),
            ("Normal aging vs. decline", "What is the difference between normal aging and very mild cognitive decline?"),
            ("Daily care & safety", "What daily care and caregiver support steps help most in moderate dementia?"),
            ("Questions for a doctor", "What next steps and medical tests should we discuss with a doctor?"),
        ]

    for q_col, (q_lbl, q_text) in zip([q1, q2, q3, q4], dyn_qs):
        with q_col:
            if st.button(q_lbl, key=f"rag_sug_{q_lbl}", use_container_width=True, help=q_text):
                prompt_to_run = q_text

    st.markdown("<hr style='border: none; border-top: 1px solid rgba(51, 65, 85, 0.5); margin: 1rem 0;'>", unsafe_allow_html=True)

    # Render Chat History
    for msg in st.session_state["rag_messages"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                msg_status = msg.get("status", "success")
                if msg_status == "success":
                    st.markdown("<div style='margin-bottom: 0.5rem;'><span class='badge-live'>🟢 Live LLM Response (Google Gemini)</span></div>", unsafe_allow_html=True)
                elif msg_status == "fallback":
                    st.markdown("<div style='margin-bottom: 0.5rem;'><span class='badge-fallback'>🛡️ Local Knowledge Base Mode</span></div>", unsafe_allow_html=True)
                elif msg_status == "disclaimer":
                    st.markdown("<div style='margin-bottom: 0.5rem;'><span class='badge-fallback' style='background: rgba(245, 158, 11, 0.15); color: #fbbf24; border-color: rgba(245, 158, 11, 0.3);'>⚠️ Medical Disclaimer</span></div>", unsafe_allow_html=True)

                st.markdown(msg["content"])

                # Render sources if available
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📚 Retrieved Source References ({len(sources)})", expanded=False):
                        for s_idx, src in enumerate(sources, 1):
                            s_title = src.get("title", "Reference")
                            s_pub = src.get("publisher", "NeuroScan AI")
                            s_url = src.get("source_url", "")
                            s_cat = src.get("category", "General")
                            s_date = src.get("access_date", "2026-10-29")

                            url_link = f"[{s_url}]({s_url})" if s_url else "Local Document"
                            st.markdown(
                                f"**{s_idx}. {s_title}**  \n"
                                f"&bull; **Publisher**: {s_pub} | **Category**: `{s_cat}`  \n"
                                f"&bull; **Source Link**: {url_link} | **Accessed**: {s_date}"
                            )

    # Chat Input Box
    user_input = st.chat_input("Ask about Alzheimer's symptoms, daily care, brain health, or MRI findings...")

    active_prompt = prompt_to_run or user_input
    if active_prompt:
        # Add user query to conversation
        st.session_state["rag_messages"].append({"role": "user", "content": active_prompt})

        with st.chat_message("assistant"):
            with st.spinner("Generating grounded educational response..."):
                workflow_result = run_neuroscan_workflow(
                    active_prompt,
                    classifier_context=st.session_state.get("active_classifier_context"),
                    language=rag_lang_code,
                )

            if workflow_result.get("safety_status") == "medical_advice_intercepted":
                res_status = "disclaimer"
            elif "fallback" in workflow_result.get("response_source", "") or workflow_result.get("safety_status") == "out_of_scope":
                res_status = "fallback"
            else:
                res_status = "success"

            if res_status == "success":
                st.markdown("<div style='margin-bottom: 0.5rem;'><span class='badge-live'>🟢 Live LLM Response (Google Gemini)</span></div>", unsafe_allow_html=True)
            elif res_status == "fallback":
                st.markdown("<div style='margin-bottom: 0.5rem;'><span class='badge-fallback'>🛡️ Local Knowledge Base Mode</span></div>", unsafe_allow_html=True)
            elif res_status == "disclaimer":
                st.markdown("<div style='margin-bottom: 0.5rem;'><span class='badge-fallback' style='background: rgba(245, 158, 11, 0.15); color: #fbbf24; border-color: rgba(245, 158, 11, 0.3);'>⚠️ Medical Disclaimer</span></div>", unsafe_allow_html=True)

            st.markdown(workflow_result["answer"])

            # Render sources
            sources = workflow_result.get("sources", [])
            if sources:
                with st.expander(f"📚 Retrieved Source References ({len(sources)})", expanded=True):
                    for s_idx, src in enumerate(sources, 1):
                        s_title = src.get("title", "Reference")
                        s_pub = src.get("publisher", "NeuroScan AI")
                        s_url = src.get("source_url", "")
                        s_cat = src.get("category", "General")
                        s_date = src.get("access_date", "2026-10-29")
                        url_link = f"[{s_url}]({s_url})" if s_url else "Local Document"
                        st.markdown(
                            f"**{s_idx}. {s_title}**  \n"
                            f"&bull; **Publisher**: {s_pub} | **Category**: `{s_cat}`  \n"
                            f"&bull; **Source Link**: {url_link} | **Accessed**: {s_date}"
                        )

        # Store in session state and rerun
        st.session_state["rag_messages"].append({
            "role": "assistant",
            "content": workflow_result["answer"],
            "sources": sources,
            "status": res_status,
        })
        st.rerun()
