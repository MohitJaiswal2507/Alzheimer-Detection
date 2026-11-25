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
# Sidebar Navigation & System Telemetry
# -----------------------------------------------------------------------------
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

    page = st.radio(
        "Navigation",
        options=["📊 Dashboard", "🔬 MRI Analysis", "💡 Knowledge Assistant"],
        index=1,  # Default to MRI Analysis for seamless demo
    )

    st.markdown("<hr style='border: none; border-top: 1px solid rgba(51, 65, 85, 0.6); margin: 1.2rem 0;'>", unsafe_allow_html=True)

    # Sleek System Telemetry Widget
    st.markdown("<div style='font-size: 0.72rem; font-weight: 700; color: #64748b; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.6rem;'>SYSTEM TELEMETRY</div>", unsafe_allow_html=True)

    try:
        _, _, _, meta, dev = get_cached_model()
        dev_label = "CUDA GPU" if dev.type == "cuda" else "PyTorch CPU"
        f1_label = f"{meta['val_f1']:.4f}"
        epoch_label = f"Epoch {meta['epoch']}"

        st.markdown(f"""
        <div style="background: #0f172a; border: 1px solid rgba(51, 65, 85, 0.7); border-radius: 12px; padding: 0.9rem; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.55rem; font-size: 0.8rem;">
                <span style="color: #94a3b8;">Active Model</span>
                <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); padding: 1px 7px; border-radius: 5px; font-weight: 600; font-size: 0.72rem;">EfficientNet-B0</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.55rem; font-size: 0.8rem;">
                <span style="color: #94a3b8;">Checkpoint</span>
                <span style="color: #e2e8f0; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">{epoch_label}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.55rem; font-size: 0.8rem;">
                <span style="color: #94a3b8;">Compute Device</span>
                <span style="background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.25); padding: 1px 7px; border-radius: 5px; font-weight: 600; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;">{dev_label}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.55rem; font-size: 0.8rem;">
                <span style="color: #94a3b8;">Val Macro F1</span>
                <span style="color: #f8fafc; font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;">{f1_label}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem;">
                <span style="color: #94a3b8;">LLM Provider</span>
                <span style="background: rgba(129, 140, 248, 0.12); color: #a5b4fc; border: 1px solid rgba(129, 140, 248, 0.25); padding: 2px 7px; border-radius: 5px; font-weight: 600; font-size: 0.7rem; font-family: 'JetBrains Mono', monospace; white-space: nowrap;">{GEMINI_MODEL}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Checkpoint Error: {e}")

    # Sidebar Educational Disclaimer Notice
    st.markdown("""
    <div style="padding: 0.75rem 0.9rem; border-radius: 10px; background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(51, 65, 85, 0.6); font-size: 0.72rem; color: #94a3b8; line-height: 1.45;">
        <span style="color: #cbd5e1; font-weight: 600;">🎓 CSE Final Year Project</span><br>
        Developed for educational research. Not approved for clinical diagnostics.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 1: DASHBOARD
# =============================================================================
if page == "📊 Dashboard":
    # Header Strip
    dh1, dh2 = st.columns([2.2, 1.1], gap="medium")
    with dh1:
        st.markdown("""
        <h1 style="font-size: 2.05rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.025em; line-height: 1.15; margin: 0 0 0.25rem 0;">
            Project Overview
        </h1>
        """, unsafe_allow_html=True)
    with dh2:
        st.markdown(
            '<div style="display: flex; justify-content: flex-end; align-items: center; padding-top: 0.45rem;">'
            '<span style="font-size: 0.74rem; font-family: \'JetBrains Mono\', monospace; color: #38bdf8; '
            'background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.24); '
            'padding: 5px 12px; border-radius: 9999px; white-space: nowrap;">'
            'EfficientNet-B0 &bull; Epoch 15 &bull; N=960 Eval</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 0.7rem;'></div>", unsafe_allow_html=True)

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

    # Compact Architecture & Training Spec Strip
    st.markdown(
        '<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.85rem;">'
        '<div style="background: #111827; border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 10px; padding: 0.85rem 1rem;">'
        '<div style="font-size: 0.68rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em;">Input Tensor</div>'
        '<div style="font-size: 0.86rem; font-family: \'JetBrains Mono\', monospace; color: #e2e8f0; font-weight: 600; margin-top: 3px;">224&times;224 RGB</div>'
        '</div>'
        '<div style="background: #111827; border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 10px; padding: 0.85rem 1rem;">'
        '<div style="font-size: 0.68rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em;">Loss Function</div>'
        '<div style="font-size: 0.86rem; font-family: \'JetBrains Mono\', monospace; color: #e2e8f0; font-weight: 600; margin-top: 3px;">Weighted Cross-Entropy</div>'
        '</div>'
        '<div style="background: #111827; border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 10px; padding: 0.85rem 1rem;">'
        '<div style="font-size: 0.68rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em;">Data Partition</div>'
        '<div style="font-size: 0.86rem; font-family: \'JetBrains Mono\', monospace; color: #e2e8f0; font-weight: 600; margin-top: 3px;">70 / 15 / 15 Split</div>'
        '</div>'
        '<div style="background: #111827; border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 10px; padding: 0.85rem 1rem;">'
        '<div style="font-size: 0.68rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em;">Explainability</div>'
        '<div style="font-size: 0.86rem; font-family: \'JetBrains Mono\', monospace; color: #e2e8f0; font-weight: 600; margin-top: 3px;">Grad-CAM + Gemini</div>'
        '</div>'
        '</div>',
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

    # Session State Initialization for Loaded Image
    if "active_image" not in st.session_state:
        default_path = PRESET_SAMPLES["Moderate_Demented"]["path"]
        if default_path.exists():
            st.session_state["active_image"] = Image.open(default_path)
            st.session_state["active_source_name"] = "moderate.jpg (Preset: Moderate Demented)"
        else:
            st.session_state["active_image"] = None
            st.session_state["active_source_name"] = None

    # Compact Command Header Bar
    hdr_col, mode_col, lang_col = st.columns([2.2, 1.3, 1.0], gap="medium")
    with hdr_col:
        st.markdown("""
        <h1 style="font-size: 2.05rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.025em; line-height: 1.15; margin: 0 0 0.25rem 0;">
            2D Brain MRI Analysis
        </h1>
        """, unsafe_allow_html=True)
    with mode_col:
        input_mode = st.radio(
            "Source",
            options=["Demo Scans", "Upload Scan"],
            horizontal=True,
        )
    with lang_col:
        language_choice = st.selectbox(
            "Language",
            options=["English", "Hindi / Hinglish"],
            index=0,
        )
        lang_code = "hinglish" if "Hinglish" in language_choice else "english"

    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Asymmetric 2-Column Workstation: [1.05, 1.35]
    # -------------------------------------------------------------------------
    left_ws, right_ws = st.columns([1.05, 1.35], gap="large")

    with left_ws:
        # 1. Specimen Selector or Uploader
        if input_mode == "Demo Scans":
            p1, p2, p3, p4 = st.columns(4, gap="small")
            with p1:
                if st.button("Moderate", key="btn_mod", use_container_width=True, help="Moderate Demented (1%)"):
                    mod_p = PRESET_SAMPLES["Moderate_Demented"]["path"]
                    if mod_p.exists():
                        st.session_state["active_image"] = Image.open(mod_p)
                        st.session_state["active_source_name"] = "moderate.jpg (Preset: Moderate Demented)"
                        st.session_state.pop("last_analysis", None)
            with p2:
                if st.button("Mild", key="btn_mild", use_container_width=True, help="Mild Demented (14%)"):
                    mild_p = PRESET_SAMPLES["Mild_Demented"]["path"]
                    if mild_p.exists():
                        st.session_state["active_image"] = Image.open(mild_p)
                        st.session_state["active_source_name"] = "mild.jpg (Preset: Mild Demented)"
                        st.session_state.pop("last_analysis", None)
            with p3:
                if st.button("Very Mild", key="btn_vmild", use_container_width=True, help="Very Mild Demented (35%)"):
                    vmild_p = PRESET_SAMPLES["Very_Mild_Demented"]["path"]
                    if vmild_p.exists():
                        st.session_state["active_image"] = Image.open(vmild_p)
                        st.session_state["active_source_name"] = "verymild.jpg (Preset: Very Mild Demented)"
                        st.session_state.pop("last_analysis", None)
            with p4:
                if st.button("Non-Dem", key="btn_non", use_container_width=True, help="Non-Demented (50%)"):
                    non_p = PRESET_SAMPLES["Non_Demented"]["path"]
                    if non_p.exists():
                        st.session_state["active_image"] = Image.open(non_p)
                        st.session_state["active_source_name"] = "non.jpg (Preset: Non-Demented)"
                        st.session_state.pop("last_analysis", None)
        else:
            uploaded_file = st.file_uploader(
                "Upload axial brain MRI (JPG, JPEG, PNG)",
                type=["jpg", "jpeg", "png"],
            )
            if uploaded_file is not None:
                try:
                    st.session_state["active_image"] = Image.open(uploaded_file)
                    st.session_state["active_source_name"] = f"{uploaded_file.name} (Uploaded)"
                    st.session_state.pop("last_analysis", None)
                except Exception as e:
                    st.error(f"Invalid image format: {e}")

        active_img = st.session_state.get("active_image", None)
        active_name = st.session_state.get("active_source_name", "None")

        # 2. Machined Double-Bezel PACS Viewport
        if active_img is not None:
            w, h = active_img.width, active_img.height
            mode = active_img.mode

            st.markdown(
                f'<div style="background: #0d1322; border: 1px solid rgba(148, 163, 184, 0.14); border-radius: 16px; padding: 8px 8px 0 8px; margin-top: 0.4rem;">'
                f'<div style="background: #060913; border: 1px solid rgba(148, 163, 184, 0.08); border-bottom: none; border-radius: 10px 10px 0 0; padding: 9px 14px; display: flex; justify-content: space-between; align-items: center;">'
                f'<div style="display: flex; align-items: center; gap: 8px;">'
                f'<span style="width: 7px; height: 7px; border-radius: 50%; background: #10b981;"></span>'
                f'<span style="font-size: 0.72rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.08em;">AXIAL T1/T2 VIEWPORT</span>'
                f'</div>'
                f'<span style="font-size: 0.7rem; font-family: \'JetBrains Mono\', monospace; color: #38bdf8; background: rgba(56, 189, 248, 0.1); padding: 2px 8px; border-radius: 5px;">{w}&times;{h}px &bull; {mode}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            st.image(active_img, use_container_width=True)

            st.markdown(
                f'<div style="background: #0d1322; border: 1px solid rgba(148, 163, 184, 0.14); border-top: none; border-radius: 0 0 16px 16px; padding: 0 8px 8px 8px; margin-bottom: 0.9rem;">'
                f'<div style="background: #060913; border: 1px solid rgba(148, 163, 184, 0.08); border-top: none; border-radius: 0 0 10px 10px; padding: 7px 14px; display: flex; justify-content: space-between; font-size: 0.72rem; color: #64748b;">'
                f'<span>Source: <b style="color: #cbd5e1;">{active_name}</b></span>'
                f'<span style="font-family: \'JetBrains Mono\', monospace;">224&times;224 target</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            # 3. Primary Action CTA anchored directly under the viewport
            analyze_button = st.button("Run Inference & Explainability", type="primary", use_container_width=True)

            if analyze_button:
                with st.spinner("Running EfficientNet-B0 inference & Grad-CAM..."):
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
                f'<div style="background: #0d1322; border: 1px solid rgba(148, 163, 184, 0.14); border-radius: 16px; padding: 8px;">'
                f'<div style="background: #111827; border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 11px; padding: 1.2rem;">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.85rem;">'
                f'<span style="font-size: 0.88rem; font-weight: 700; color: #f8fafc;">Class Softmax Distribution</span>'
                f'<span style="font-size: 0.72rem; font-family: \'JetBrains Mono\', monospace; color: #64748b;">Sum = 100.0%</span>'
                f'</div>'
                f'{bars_html}'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        else:
            # Composed Ready State (No blank void)
            st.markdown(
                f'<div style="background: #0d1322; border: 1px solid rgba(148, 163, 184, 0.14); border-radius: 16px; padding: 8px;">'
                f'<div style="background: #111827; border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 11px; padding: 2rem 1.5rem;">'
                f'<div style="display: inline-flex; align-items: center; gap: 7px; background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.25); padding: 3px 10px; border-radius: 9999px; font-size: 0.72rem; font-weight: 600; color: #38bdf8; margin-bottom: 1rem;">'
                f'Ready for Inference</div>'
                f'<div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.5rem;">Diagnostic Telemetry Standby</div>'
                f'<div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.55; margin-bottom: 1.25rem; max-width: 52ch;">'
                f'Select a preset scan or upload an axial MRI slice on the left, then run inference to inspect class probabilities, Grad-CAM saliency maps, and educational context.</div>'
                f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; font-size: 0.78rem;">'
                f'<div style="background: #0b0f19; border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 8px; padding: 0.7rem 0.9rem;">'
                f'<div style="color: #64748b; margin-bottom: 2px;">Backbone</div>'
                f'<div style="color: #e2e8f0; font-family: \'JetBrains Mono\', monospace; font-weight: 600;">EfficientNet-B0 (Ep {meta["epoch"]})</div></div>'
                f'<div style="background: #0b0f19; border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 8px; padding: 0.7rem 0.9rem;">'
                f'<div style="color: #64748b; margin-bottom: 2px;">Saliency Target</div>'
                f'<div style="color: #e2e8f0; font-family: \'JetBrains Mono\', monospace; font-weight: 600;">model.features[-1]</div></div>'
                f'</div></div></div>',
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
            '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">'
            '<div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">Visual Explainability &bull; Grad-CAM</div>'
            '<span style="font-size: 0.72rem; font-family: \'JetBrains Mono\', monospace; color: #38bdf8; background: rgba(56, 189, 248, 0.1); padding: 3px 10px; border-radius: 9999px; border: 1px solid rgba(56, 189, 248, 0.24);">Target: model.features[-1]</span>'
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
            Ask questions about Alzheimer's pathology, dementia stages, 2D brain MRI slices, EfficientNet-B0, or Grad-CAM explainability.
            Answers are grounded in curated educational sources with transparent citations.
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
                    "You can ask me questions about Alzheimer's disease pathology, clinical dementia stages vs. "
                    "our dataset categories, 2D axial MRI physics, the EfficientNet-B0 transfer learning architecture, "
                    "or Grad-CAM explainability. How can I help you today?"
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

    # Controls row: Language, Retrieval Diagnostics, Clear Button
    rag_ctrl1, rag_ctrl2, rag_ctrl3 = st.columns([2, 2, 1], gap="medium")

    with rag_ctrl1:
        rag_lang_choice = st.selectbox(
            "Response Language",
            options=["English", "Hindi / Hinglish"],
            index=0,
            key="rag_language_select",
        )
        rag_lang_code = "hinglish" if "Hinglish" in rag_lang_choice else "english"

    with rag_ctrl2:
        dep_status = get_dependency_status()
        lg_available = is_langgraph_available()
        workflow_tag = "LangGraph Workflow" if lg_available else "NeuroScan Workflow"
        vector_tag = "Dense ChromaDB" if dep_status["ready"] else "Knowledge Base"

        st.markdown(f"""
        <div style="padding-top: 1.6rem; font-size: 0.78rem; display: flex; gap: 6px; flex-wrap: wrap;">
            <span class="badge-live">🟢 {workflow_tag}</span>
            <span class="badge-live" style="background: rgba(56, 189, 248, 0.12); color: #38bdf8; border-color: rgba(56, 189, 248, 0.25);">📚 {vector_tag}</span>
        </div>
        """, unsafe_allow_html=True)

    with rag_ctrl3:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat", use_container_width=True, help="Clear chat history"):
            st.session_state["rag_messages"] = []
            st.rerun()

    # Suggested Prompts Row
    st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #64748b; letter-spacing: 0.08em; text-transform: uppercase; margin: 0.8rem 0 0.4rem 0;'>SUGGESTED EDUCATIONAL QUESTIONS</div>", unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)

    prompt_to_run = None
    with q1:
        if st.button("🧠 What is Alzheimer's?", use_container_width=True):
            prompt_to_run = "What is Alzheimer's disease and what are its biological hallmarks?"
    with q2:
        if st.button("📊 4 Dataset Categories?", use_container_width=True):
            prompt_to_run = "What are the four dataset categories used in this project and how do they differ from clinical diagnosis?"
    with q3:
        if st.button("🔬 What is an Axial Slice?", use_container_width=True):
            prompt_to_run = "What is an axial MRI slice, and what are the limitations of analyzing 2D slices instead of 3D volume?"
    with q4:
        if st.button("🔍 How does Grad-CAM work?", use_container_width=True):
            prompt_to_run = "How does Grad-CAM work in this model, and why is it not a clinical lesion map?"

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
    user_input = st.chat_input("Ask an educational question about Alzheimer's, MRI, EfficientNet, or Grad-CAM...")

    active_prompt = prompt_to_run or user_input
    if active_prompt:
        # Add user query to conversation
        st.session_state["rag_messages"].append({"role": "user", "content": active_prompt})

        with st.chat_message("assistant"):
            with st.spinner("Executing educational workflow and generating grounded response..."):
                workflow_result = run_neuroscan_workflow(active_prompt, language=rag_lang_code)

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
