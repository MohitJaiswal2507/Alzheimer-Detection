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
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    code, pre, .mono-text {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Global Container Adjustments */
    .block-container {
        padding-top: 1.6rem !important;
        padding-bottom: 3rem !important;
        max-width: 1350px !important;
    }

    /* Sidebar Background & Borders */
    section[data-testid="stSidebar"] {
        background-color: #070b14 !important;
        border-right: 1px solid rgba(30, 41, 59, 0.8) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
    }

    /* Primary Action Button (Electric Cyan Gradient replacing harsh coral) */
    div.stButton > button[kind="primary"], div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.6rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 20px rgba(6, 182, 212, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.2) !important;
        transition: all 0.25s cubic-bezier(0.32, 0.72, 0, 1) !important;
    }
    div.stButton > button[kind="primary"]:hover, div.stButton > button:first-child:hover {
        transform: translateY(-2px) scale(1.008) !important;
        box-shadow: 0 8px 28px rgba(6, 182, 212, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.3) !important;
        border-color: rgba(56, 189, 248, 0.8) !important;
    }
    div.stButton > button[kind="primary"]:active, div.stButton > button:first-child:active {
        transform: translateY(0px) scale(0.99) !important;
    }

    /* Secondary / Preset Buttons */
    div.stButton > button[kind="secondary"] {
        background: #0f172a !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: #1e293b !important;
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.15) !important;
    }

    /* File Uploader Container */
    div[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.6);
        border: 1px dashed rgba(56, 189, 248, 0.35);
        border-radius: 14px;
        padding: 0.8rem;
        transition: border-color 0.2s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(56, 189, 248, 0.7);
    }

    /* Double-Bezel Card Architecture */
    .bezel-card {
        background: #0f172a;
        border: 1px solid rgba(51, 65, 85, 0.7);
        border-radius: 16px;
        padding: 1.3rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.05);
    }
    .bezel-card-highlight {
        background: #0f172a;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 16px;
        padding: 1.3rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px -5px rgba(6, 182, 212, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.08);
    }

    /* Eyebrow Pill Badge */
    .badge-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: #38bdf8;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .pulse-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #38bdf8;
        box-shadow: 0 0 10px #38bdf8;
    }

    /* KPI Metric Cards (Dashboard) */
    .metric-kpi-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
        border-top: 3px solid #0284c7;
    }
    .metric-kpi-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }
    .metric-kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }
    .metric-kpi-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* Badges & Status Tags */
    .badge-live {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-fallback {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Medical Disclaimer Banners */
    .disclaimer-box {
        background: rgba(30, 27, 75, 0.5);
        border-left: 4px solid #818cf8;
        padding: 0.85rem 1.1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #e0e7ff;
        font-size: 0.85rem;
        line-height: 1.5;
    }
    .warning-box {
        background: rgba(69, 10, 10, 0.5);
        border-left: 4px solid #f87171;
        padding: 0.85rem 1.1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #fee2e2;
        font-size: 0.85rem;
        line-height: 1.5;
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
        background: rgba(30, 41, 59, 0.8);
        border-radius: 9999px;
        height: 10px;
        overflow: hidden;
        border: 1px solid rgba(51, 65, 85, 0.6);
    }
    .prob-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.6s cubic-bezier(0.32, 0.72, 0, 1);
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
        options=["📊 Dashboard", "🔬 MRI Analysis", "ℹ️ About Project"],
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
                <span style="background: rgba(129, 140, 248, 0.12); color: #a5b4fc; border: 1px solid rgba(129, 140, 248, 0.25); padding: 1px 7px; border-radius: 5px; font-weight: 600; font-size: 0.72rem;">{GEMINI_MODEL}</span>
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
    st.markdown("""
    <div style="margin-bottom: 1.8rem;">
        <div class="badge-eyebrow"><span class="pulse-dot"></span> EVALUATION &amp; BENCHMARKS</div>
        <h1 style="font-size: 2.3rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em; margin: 0 0 0.4rem 0;">
            Project Overview <span style="background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">&amp; Metrics Dashboard</span>
        </h1>
        <p style="font-size: 1rem; color: #94a3b8; margin: 0; max-width: 850px;">
            Comprehensive performance summary across held-out test splits, dataset distributions, and class-weighted training dynamics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("""
        <div class="metric-kpi-card">
            <div class="metric-kpi-title">Test Accuracy</div>
            <div class="metric-kpi-value">95.52%</div>
            <div class="metric-kpi-sub">917 / 960 held-out scans</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown("""
        <div class="metric-kpi-card" style="border-top-color: #06b6d4;">
            <div class="metric-kpi-title">Macro F1-Score</div>
            <div class="metric-kpi-value">0.9486</div>
            <div class="metric-kpi-sub">Balanced across all 4 classes</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown("""
        <div class="metric-kpi-card" style="border-top-color: #10b981;">
            <div class="metric-kpi-title">Test Specificity</div>
            <div class="metric-kpi-value">98.24%</div>
            <div class="metric-kpi-sub">One-vs-Rest Macro Average</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown("""
        <div class="metric-kpi-card" style="border-top-color: #818cf8;">
            <div class="metric-kpi-title">Validation F1</div>
            <div class="metric-kpi-value">0.9780</div>
            <div class="metric-kpi-sub">Best Checkpoint (Epoch 15)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Two column layout: Pipeline & Dataset Breakdown
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown("""
        <div class="bezel-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.8rem;">
                🔄 Technical Pipeline
            </div>
            <ol style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6; padding-left: 1.2rem; margin: 0;">
                <li><b>Input Acquisition</b>: Ingests 2D axial slice scans in standard formats (JPG, JPEG, PNG).</li>
                <li><b>Preprocessing</b>: Grayscale-to-RGB expansion, bilinear resize to 224&times;224, and ImageNet standardization.</li>
                <li><b>Classification Engine</b>: EfficientNet-B0 feature extractor with inverse-frequency weighted cross-entropy loss.</li>
                <li><b>Visual Explainability</b>: Native PyTorch hook Grad-CAM extracting spatial activation maps from <code>model.features[-1]</code>.</li>
                <li><b>Clinical Context Explainer</b>: Structured text synthesis using Google Gemini (with deterministic offline fallback). <b>Zero image data is ever transmitted to the LLM</b>.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Dataset Class Proportions (N = 6,400)")
        dataset_data = {
            "Class Name": ["Non_Demented", "Very_Mild_Demented", "Mild_Demented", "Moderate_Demented"],
            "Total Images": [3200, 2240, 896, 64],
            "Dataset Share": ["50.0%", "35.0%", "14.0%", "1.0% (Extreme Minority)"],
            "Held-Out Test (N=960)": [480, 336, 135, 9],
        }
        st.dataframe(pd.DataFrame(dataset_data), use_container_width=True, hide_index=True)

    with right_col:
        st.markdown("##### Held-Out Test Set Performance Breakdown (N = 960)")
        test_metrics_data = {
            "Class": ["Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"],
            "Precision": ["89.86%", "100.00%", "96.85%", "96.04%"],
            "Recall": ["98.52%", "88.89%", "96.04%", "93.75%"],
            "F1-Score": ["0.9399", "0.9412", "0.9644", "0.9488"],
            "Specificity": ["98.18%", "100.00%", "96.88%", "97.92%"],
            "Test N": [135, 9, 480, 336],
        }
        st.dataframe(pd.DataFrame(test_metrics_data), use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="disclaimer-box" style="margin-top: 1.2rem;">
            <b>🔬 Minority Class Scrutiny (Moderate_Demented, N=9):</b><br>
            The model achieved <b>8/9 True Positives</b> with <b>0 False Positives</b> (100% precision, 88.89% recall).
            Due to the small sample size (N=9), each test misclassification impacts class recall by 11.1%.
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2: MRI ANALYSIS (CORE PRESENTATION DEMO)
# =============================================================================
elif page == "🔬 MRI Analysis":
    # Hero Title with Eyebrow Pill
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <div class="badge-eyebrow"><span class="pulse-dot"></span> NEUROSCAN AI &bull; DEEP LEARNING WORKSTATION</div>
        <h1 style="font-size: 2.3rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em; margin: 0 0 0.4rem 0;">
            2D Brain MRI Analysis <span style="background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">&amp; Explainability</span>
        </h1>
        <p style="font-size: 0.98rem; color: #94a3b8; margin: 0; max-width: 850px;">
            Select an instant demo scan or upload a custom axial MRI slice to run neural classification, spatial Grad-CAM saliency mapping, and clinical context explanation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load model safely
    try:
        model, idx_to_class, class_to_idx, meta, device = get_cached_model()
    except Exception as e:
        st.error(f"Cannot load trained model checkpoint: {e}")
        st.stop()

    # Session State Initialization for Loaded Image
    if "active_image" not in st.session_state:
        # Default initialize with moderate demo image for immediate presentation readiness
        default_path = PRESET_SAMPLES["Moderate_Demented"]["path"]
        if default_path.exists():
            st.session_state["active_image"] = Image.open(default_path)
            st.session_state["active_source_name"] = "moderate.jpg (Preset: Moderate Demented)"
        else:
            st.session_state["active_image"] = None
            st.session_state["active_source_name"] = None

    # Top Control Area: Input Selection & Analysis Settings
    ctrl_col, config_col = st.columns([2, 1], gap="large")

    with ctrl_col:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>1. Select MRI Scan Source</div>", unsafe_allow_html=True)

        input_mode = st.radio(
            "Input Mode",
            options=["⚡ Instant Demo Scans", "📁 Upload Custom File"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if input_mode == "⚡ Instant Demo Scans":
            st.markdown("<div style='font-size: 0.78rem; color: #94a3b8; margin-bottom: 0.6rem;'>Click any sample to load a verified dataset image instantly:</div>", unsafe_allow_html=True)
            p1, p2, p3, p4 = st.columns(4)

            with p1:
                if st.button("🔴 Moderate (1%)", key="btn_mod", use_container_width=True, help="Load Moderate_Demented sample"):
                    mod_p = PRESET_SAMPLES["Moderate_Demented"]["path"]
                    if mod_p.exists():
                        st.session_state["active_image"] = Image.open(mod_p)
                        st.session_state["active_source_name"] = "moderate.jpg (Preset: Moderate Demented)"
                        st.session_state.pop("last_analysis", None)
            with p2:
                if st.button("🟠 Mild (14%)", key="btn_mild", use_container_width=True, help="Load Mild_Demented sample"):
                    mild_p = PRESET_SAMPLES["Mild_Demented"]["path"]
                    if mild_p.exists():
                        st.session_state["active_image"] = Image.open(mild_p)
                        st.session_state["active_source_name"] = "mild.jpg (Preset: Mild Demented)"
                        st.session_state.pop("last_analysis", None)
            with p3:
                if st.button("🔵 Very Mild (35%)", key="btn_vmild", use_container_width=True, help="Load Very_Mild_Demented sample"):
                    vmild_p = PRESET_SAMPLES["Very_Mild_Demented"]["path"]
                    if vmild_p.exists():
                        st.session_state["active_image"] = Image.open(vmild_p)
                        st.session_state["active_source_name"] = "verymild.jpg (Preset: Very Mild Demented)"
                        st.session_state.pop("last_analysis", None)
            with p4:
                if st.button("🟢 Non-Dem (50%)", key="btn_non", use_container_width=True, help="Load Non_Demented sample"):
                    non_p = PRESET_SAMPLES["Non_Demented"]["path"]
                    if non_p.exists():
                        st.session_state["active_image"] = Image.open(non_p)
                        st.session_state["active_source_name"] = "non.jpg (Preset: Non-Demented)"
                        st.session_state.pop("last_analysis", None)

        else:
            uploaded_file = st.file_uploader(
                "Upload an axial brain MRI scan (JPG, JPEG, PNG)",
                type=["jpg", "jpeg", "png"],
                help="Upload a 2D axial brain slice image.",
            )
            if uploaded_file is not None:
                try:
                    st.session_state["active_image"] = Image.open(uploaded_file)
                    st.session_state["active_source_name"] = f"{uploaded_file.name} (Uploaded File)"
                    st.session_state.pop("last_analysis", None)
                except Exception as e:
                    st.error(f"Invalid image format: {e}")

    with config_col:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>2. Analysis Configuration</div>", unsafe_allow_html=True)
        language_choice = st.selectbox(
            "Explanation Language",
            options=["English", "Hindi / Hinglish"],
            index=0,
            help="Select the language style for the educational result explanation.",
        )
        lang_code = "hinglish" if "Hinglish" in language_choice else "english"

        # Privacy Shield Callout Box
        st.markdown("""
        <div style="background: #0f172a; border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 10px; padding: 0.75rem 0.9rem; margin-top: 0.6rem;">
            <div style="display: flex; align-items: center; gap: 6px; font-size: 0.75rem; font-weight: 700; color: #38bdf8;">
                <span>🔒</span> PRIVACY SHIELD ACTIVE
            </div>
            <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 0.2rem; line-height: 1.4;">
                Only categorical labels &amp; softmax numbers are sent to the LLM. <b>Zero MRI pixels or file paths are ever transmitted.</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border: none; border-top: 1px solid rgba(51, 65, 85, 0.5); margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Image Viewport & Action Trigger Row
    # -------------------------------------------------------------------------
    active_img = st.session_state.get("active_image", None)
    active_name = st.session_state.get("active_source_name", "None")

    if active_img is not None:
        view_col, action_col = st.columns([1, 1], gap="large")

        with view_col:
            st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Active Scan Viewport</div>", unsafe_allow_html=True)

            # High-end Clinical PACS Viewport Enclosure
            w, h = active_img.width, active_img.height
            mode = active_img.mode

            st.markdown(f"""
            <div style="background: #0b1120; border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 14px; overflow: hidden; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);">
                <div style="background: #0f172a; padding: 8px 14px; border-bottom: 1px solid rgba(51, 65, 85, 0.8); display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981;"></span>
                        <span style="font-size: 0.72rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.08em; text-transform: uppercase;">AXIAL MRI T1/T2 CONSOLE</span>
                    </div>
                    <span style="font-size: 0.7rem; font-family: 'JetBrains Mono', monospace; color: #38bdf8; background: rgba(56, 189, 248, 0.1); padding: 2px 8px; border-radius: 4px;">{w} &times; {h} px &bull; {mode}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.image(active_img, use_container_width=True)

            st.markdown(f"""
            <div style="background: #0f172a; padding: 6px 14px; font-size: 0.72rem; color: #64748b; border: 1px solid rgba(51, 65, 85, 0.5); border-top: none; border-radius: 0 0 12px 12px; display: flex; justify-content: space-between;">
                <span>Source: <b style="color: #94a3b8;">{active_name}</b></span>
                <span>Preprocessed 2D Slice</span>
            </div>
            """, unsafe_allow_html=True)

        with action_col:
            st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Execute Neural Pipeline</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class="bezel-card" style="margin-bottom: 1rem;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.4rem;">
                    🚀 Automated Inference Stages
                </div>
                <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.55;">
                    Executing the pipeline will:<br>
                    &bull; <b>Preprocess</b>: Normalize tensor with ImageNet mean/std.<br>
                    &bull; <b>Predict</b>: Compute softmax output logits on EfficientNet-B0.<br>
                    &bull; <b>Grad-CAM</b>: Compute backpropagated gradients on layer <code>features[-1]</code>.<br>
                    &bull; <b>Explain</b>: Synthesize clinical context with Google Gemini.
                </div>
            </div>
            """, unsafe_allow_html=True)

            analyze_button = st.button("🔬 Run Inference & Explainability Pipeline", type="primary", use_container_width=True)

            if analyze_button:
                with st.spinner("Processing slice through EfficientNet-B0 and computing Grad-CAM..."):
                    # 1. Preprocess & Predict
                    input_tensor = preprocess_image(active_img, device=device)
                    result = predict_mri(model, input_tensor, idx_to_class)

                    pred_class = result["predicted_class"]
                    conf = result["confidence"]
                    probs = result["probabilities"]

                    # 2. Grad-CAM Generation on last conv block
                    heatmap = generate_gradcam(model, input_tensor, target_class=result["predicted_index"])

                    # 3. Privacy-Preserving LLM Result Explanation (ZERO images passed)
                    explainer_input = {
                        "predicted_class": pred_class,
                        "confidence": conf,
                        "probabilities": probs,
                    }
                    explanation_res = explain_result(explainer_input, language=lang_code)

                # Persist results in session state
                st.session_state["last_analysis"] = {
                    "pil_image": active_img,
                    "result": result,
                    "heatmap": heatmap,
                    "explanation_res": explanation_res,
                    "lang_code": lang_code,
                    "source_name": active_name,
                }

        # ---------------------------------------------------------------------
        # Render Classification, Grad-CAM & LLM Results
        # ---------------------------------------------------------------------
        if "last_analysis" in st.session_state:
            data = st.session_state["last_analysis"]
            res = data["result"]
            heatmap = data["heatmap"]
            exp = data["explanation_res"]

            pred_class = res["predicted_class"]
            conf = res["confidence"]
            probs = res["probabilities"]

            # Class Specific Theme Colors
            color_map = {
                "Non_Demented": {"accent": "#10b981", "bg": "rgba(16, 185, 129, 0.1)", "border": "rgba(16, 185, 129, 0.35)", "tag": "🟢 Non-Demented"},
                "Very_Mild_Demented": {"accent": "#38bdf8", "bg": "rgba(56, 189, 248, 0.1)", "border": "rgba(56, 189, 248, 0.35)", "tag": "🔵 Very Mild Demented"},
                "Mild_Demented": {"accent": "#f59e0b", "bg": "rgba(245, 158, 11, 0.1)", "border": "rgba(245, 158, 11, 0.35)", "tag": "🟠 Mild Demented"},
                "Moderate_Demented": {"accent": "#f43f5e", "bg": "rgba(244, 63, 94, 0.12)", "border": "rgba(244, 63, 94, 0.4)", "tag": "🔴 Moderate Demented"},
            }
            theme = color_map.get(pred_class, {"accent": "#38bdf8", "bg": "rgba(56, 189, 248, 0.1)", "border": "rgba(56, 189, 248, 0.35)", "tag": pred_class})
            clean_name = pred_class.replace("_", " ")

            st.markdown("<hr style='border: none; border-top: 1px solid rgba(51, 65, 85, 0.6); margin: 2rem 0;'>", unsafe_allow_html=True)
            st.markdown("""
            <div style="margin-bottom: 1.2rem;">
                <div class="badge-eyebrow"><span class="pulse-dot"></span> PIPELINE OUTPUT SUMMARY</div>
                <h2 style="font-size: 1.8rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.01em; margin: 0;">
                    Classification &amp; Explainability Telemetry
                </h2>
            </div>
            """, unsafe_allow_html=True)

            # Result Hero & Probability Grid
            res_col1, res_col2 = st.columns([1, 1], gap="large")

            with res_col1:
                # Dynamic Prediction Hero Card
                minority_note = ""
                if pred_class == "Moderate_Demented":
                    minority_note = """
                    <div style="margin-top: 0.9rem; padding: 0.6rem 0.8rem; border-radius: 8px; background: rgba(244, 63, 94, 0.15); border: 1px solid rgba(244, 63, 94, 0.3); font-size: 0.75rem; color: #fecdd3;">
                        ⚠️ <b>Extreme Minority Class Warning:</b> Moderate Demented represents only 1% (64 of 6,400 images) of the total dataset. Interpret confidence with caution.
                    </div>
                    """

                st.markdown(f"""
                <div style="background: #0f172a; border: 1px solid {theme['border']}; border-radius: 16px; padding: 1.5rem; box-shadow: 0 10px 30px rgba(0,0,0,0.4); position: relative;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.8rem;">
                        <div>
                            <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; letter-spacing: 0.1em; text-transform: uppercase;">CLASSIFIER PREDICTION</div>
                            <div style="font-size: 2.1rem; font-weight: 800; color: {theme['accent']}; margin-top: 0.2rem; letter-spacing: -0.01em;">{clean_name}</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; letter-spacing: 0.1em; text-transform: uppercase;">SOFTMAX CONFIDENCE</div>
                            <div style="font-size: 2.1rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #f8fafc; margin-top: 0.2rem;">{conf * 100:.2f}%</div>
                        </div>
                    </div>
                    <div style="font-size: 0.84rem; color: #cbd5e1; background: {theme['bg']}; padding: 0.75rem 1rem; border-radius: 10px; border: 1px solid {theme['border']}; line-height: 1.5;">
                        <b>Dataset Context:</b> {CLASS_DESCRIPTIONS.get(pred_class, 'Standard dataset category')}
                    </div>
                    {minority_note}
                    <div style="margin-top: 0.9rem; font-size: 0.72rem; color: #64748b;">
                        &bull; Confidence reflects the mathematical softmax distribution across 4 dataset categories, not a clinical probability of disease.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                # Custom Horizontal Animated Probability Bars
                st.markdown("""
                <div class="bezel-card">
                    <div style="font-size: 0.92rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.8rem; display: flex; justify-content: space-between;">
                        <span>Category Softmax Distribution</span>
                        <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 500;">Sum = 100.0%</span>
                    </div>
                """, unsafe_allow_html=True)

                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                for c_name, p_val in sorted_probs:
                    pct = p_val * 100
                    c_clean = c_name.replace("_", " ")
                    is_pred = (c_name == pred_class)

                    # Determine fill gradient
                    if c_name == "Moderate_Demented":
                        grad = "linear-gradient(90deg, #e11d48, #f43f5e)"
                    elif c_name == "Mild_Demented":
                        grad = "linear-gradient(90deg, #d97706, #f59e0b)"
                    elif c_name == "Very_Mild_Demented":
                        grad = "linear-gradient(90deg, #0284c7, #38bdf8)"
                    else:
                        grad = "linear-gradient(90deg, #059669, #10b981)"

                    highlight_border = f"border: 1px solid rgba(56, 189, 248, 0.4);" if is_pred else ""

                    st.markdown(f"""
                    <div class="prob-row" style="{highlight_border} padding: 4px 8px; border-radius: 8px;">
                        <div class="prob-header">
                            <span style="font-weight: {'700' if is_pred else '500'}; color: {'#f8fafc' if is_pred else '#94a3b8'};">
                                {'👉 ' if is_pred else ''}{c_clean}
                            </span>
                            <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: {'#38bdf8' if is_pred else '#cbd5e1'}; font-size: 0.82rem;">
                                {pct:.2f}%
                            </span>
                        </div>
                        <div class="prob-track">
                            <div class="prob-fill" style="width: {max(pct, 1.5):.2f}%; background: {grad};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

            # -----------------------------------------------------------------
            # Visual Explainability (Grad-CAM)
            # -----------------------------------------------------------------
            st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">🔍 Visual Explainability &bull; Grad-CAM</div>
                <span style="font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: #38bdf8; background: rgba(56, 189, 248, 0.12); padding: 3px 9px; border-radius: 9999px; border: 1px solid rgba(56, 189, 248, 0.25);">Target Layer: model.features[-1]</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="font-size: 0.84rem; color: #94a3b8; margin-bottom: 0.9rem; line-height: 1.5;">
                Grad-CAM computes the spatial gradients of the target class score with respect to feature maps in the final convolutional block.
                The overlay highlights regions that mathematically guided the network's classification.
            </div>
            """, unsafe_allow_html=True)

            # Render Matplotlib Figure in styled container
            fig = create_gradcam_figure(
                pil_image=data["pil_image"],
                heatmap=heatmap,
                predicted_class=res["predicted_class"],
                confidence=res["confidence"],
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown("""
            <div class="disclaimer-box">
                <b>🔬 Important Saliency Interpretation Notice:</b><br>
                The heatmap visualization reflects mathematical gradient influence within the convolutional layers.
                It visualizes image regions contributing to the mathematical score, <b>not</b> clinical lesion boundaries, cellular pathology, or definitive neurodegeneration.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

            # -----------------------------------------------------------------
            # Educational LLM Result Explanation
            # -----------------------------------------------------------------
            status_badge_html = (
                "<span class='badge-live'>🟢 Live LLM Explanation (Google Gemini 3.5 Flash)</span>"
                if exp["status"] == "success"
                else f"<span class='badge-fallback'>🛡️ Deterministic Local Fallback ({exp.get('error', 'API unavailable')})</span>"
            )

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #090e1a 0%, #111a2e 100%); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 16px; padding: 1.5rem; box-shadow: 0 10px 30px rgba(0,0,0,0.4); margin-bottom: 1.2rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid rgba(99, 102, 241, 0.2); padding-bottom: 0.75rem;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 1.4rem;">💬</span>
                        <span style="font-size: 1.15rem; font-weight: 700; color: #e0e7ff;">Clinical Context Explainer</span>
                    </div>
                    {status_badge_html}
                </div>
            """, unsafe_allow_html=True)

            st.markdown(exp["explanation"])

            st.markdown("</div>", unsafe_allow_html=True)

            # Mandatory Non-Diagnostic Disclaimer
            st.markdown("""
            <div class="warning-box">
                <b>⚠️ NON-DIAGNOSTIC NOTICE:</b> NeuroScan AI is an educational engineering research tool developed for university coursework.
                Predictions and explanations are not medical diagnoses. For clinical assessment, always consult certified physicians, neurologists, and radiologists.
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Please select an instant demo scan or upload an axial MRI image above to begin.")


# =============================================================================
# PAGE 3: ABOUT PROJECT & SAFETY
# =============================================================================
elif page == "ℹ️ About Project":
    st.markdown("""
    <div style="margin-bottom: 1.8rem;">
        <div class="badge-eyebrow"><span class="pulse-dot"></span> ARCHITECTURE &amp; METHODOLOGY</div>
        <h1 style="font-size: 2.3rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em; margin: 0 0 0.4rem 0;">
            About <span style="background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">NeuroScan AI</span>
        </h1>
        <p style="font-size: 1rem; color: #94a3b8; margin: 0; max-width: 850px;">
            Academic context, engineering methodology, technology stack, and known dataset constraints.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box" style="font-size: 0.92rem; padding: 1.1rem 1.3rem;">
        <b>⚠️ MANDATORY MEDICAL DISCLAIMER:</b><br>
        NeuroScan AI is strictly an undergraduate engineering academic demonstration.
        It has not undergone clinical trials, clinical validation, or regulatory approval by FDA/EMA/CDSCO.
        Model predictions must never be interpreted as clinical diagnoses or used to direct patient care.
        For cognitive, neurological, or memory concerns, always consult certified healthcare specialists.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("""
        <div class="bezel-card">
            <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.75rem;">
                🎯 Project Objectives
            </div>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                NeuroScan AI explores the intersection of deep learning computer vision, explainable AI (XAI),
                and large language model interpretation for medical image analysis:
                <ul style="margin-top: 0.4rem; padding-left: 1.2rem;">
                    <li><b>Transfer Learning Vision</b>: Fine-tuned EfficientNet-B0 on preprocessed 2D axial MRI slices.</li>
                    <li><b>Mathematical Saliency</b>: Native zero-dependency PyTorch Grad-CAM visualizing network attention layers.</li>
                    <li><b>Language Synthesis</b>: Translates quantitative outputs into accessible educational context in English and Hinglish.</li>
                    <li><b>Safe Architecture</b>: Zero image data transmission to external APIs, strict input validation, and automated fallback logic.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="bezel-card">
            <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.75rem;">
                🛠️ Technology Stack
            </div>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                &bull; <b>Deep Learning Engine</b>: PyTorch 2.x, Torchvision<br>
                &bull; <b>Model Backbone</b>: EfficientNet-B0 (5.3M parameters)<br>
                &bull; <b>Web Workstation</b>: Streamlit 1.x<br>
                &bull; <b>Language Explainer</b>: Google Gemini REST API (<code>gemini-3.5-flash</code>)<br>
                &bull; <b>Visualization</b>: Matplotlib, Pillow, NumPy
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="bezel-card">
            <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.75rem;">
                ⚠️ Critical Dataset &amp; Model Limitations
            </div>
            <ol style="font-size: 0.86rem; color: #cbd5e1; line-height: 1.6; padding-left: 1.2rem; margin: 0;">
                <li><b>2D Slice vs 3D Volume</b>: Scans are processed as single 2D axial slices rather than complete 3D volumetric T1/T2 MRI sequences.</li>
                <li><b>Patient-Level Leakage Risk</b>: The underlying dataset lacks patient IDs; multiple slices from the same subject may exist across splits.</li>
                <li><b>Severe Minority Imbalance</b>: The <code>Moderate_Demented</code> class comprises only 1% (64 scans) of the dataset. Despite class-weighted loss, statistical variance is high.</li>
                <li><b>Confidence vs Probability</b>: Softmax scores represent the model's mathematical score relative to training distributions, <b>not</b> clinical probability of disease.</li>
                <li><b>Grad-CAM Saliency Limits</b>: Heatmaps reflect mathematical gradient activations, which can be influenced by scan contrast, skull edges, or dataset artifacts.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border: none; border-top: 1px solid rgba(51, 65, 85, 0.5); margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.75rem; color: #64748b; text-align: center;'>NeuroScan AI &bull; CSE Final Year College Project &bull; Phase 6 Prototype UI</div>", unsafe_allow_html=True)
