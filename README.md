# NeuroScan AI — Educational Brain MRI Classification & Explainability Workstation

[![Python 3.10 | 3.13](https://img.shields.io/badge/Python-3.10%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.9.0%2Bcu130-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit 1.51](https://img.shields.io/badge/Streamlit-1.51.0-FF4B4B.svg)](https://streamlit.io/)
[![Architecture EfficientNet-B0](https://img.shields.io/badge/Backbone-EfficientNet--B0-green.svg)](https://arxiv.org/abs/1905.11946)
[![Tests Passing](https://img.shields.io/badge/Tests-59%2F59%20Passing-brightgreen.svg)]()
[![Status Educational Prototype](https://img.shields.io/badge/Status-College%20Project%20Prototype-orange.svg)]()

> **Undergraduate Computer Science & Engineering Final Year Project**  
> *Developed for academic research and educational demonstration. Not intended, certified, or approved for clinical diagnostic use.*

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [Main Features](#4-main-features)
5. [Technology Stack](#5-technology-stack)
6. [System Architecture](#6-system-architecture)
7. [Dataset Overview & Class Distribution](#7-dataset-overview--class-distribution)
8. [Model Architecture & Transfer Learning](#8-model-architecture--transfer-learning)
9. [Training & Evaluation Summary](#9-training--evaluation-summary)
10. [Grad-CAM Explainability](#10-grad-cam-explainability)
11. [RAG Knowledge Base Architecture](#11-rag-knowledge-base-architecture)
12. [Google Gemini Integration & Fallback](#12-google-gemini-integration--fallback)
13. [LangGraph Multi-Stage Reasoning Workflow](#13-langgraph-multi-stage-reasoning-workflow)
14. [Application Setup](#14-application-setup)
15. [Environment Setup](#15-environment-setup)
16. [How to Launch Streamlit](#16-how-to-launch-streamlit)
17. [How to Use the Application](#17-how-to-use-the-application)
18. [Critical Project & Dataset Limitations](#18-critical-project--dataset-limitations)
19. [Mandatory Medical & Clinical Disclaimer](#19-mandatory-medical--clinical-disclaimer)
20. [Future Improvements](#20-future-improvements)
21. [Repository Structure](#21-repository-structure)

---

## 1. Project Overview

**NeuroScan AI** is an educational, multi-modal medical AI workstation designed to demonstrate modern deep learning, visual explainability, and retrieval-augmented language models in the context of Alzheimer's disease neuroimaging.

The system processes 2D axial brain Magnetic Resonance Imaging (MRI) scans, classifies them into four dataset categories using a fine-tuned **EfficientNet-B0** convolutional neural network, visualizes mathematical feature attribution via **Grad-CAM**, and synthesizes transparent clinical and methodological context using **Retrieval-Augmented Generation (RAG)** and a **LangGraph** multi-stage reasoning workflow.

---

## 2. Problem Statement

Alzheimer's disease is a progressive neurodegenerative disorder affecting tens of millions of individuals globally. While deep neural networks have achieved remarkable visual classification benchmarks on neuroimaging datasets, medical AI tools frequently suffer from critical real-world limitations:
- **Black-box opacity**: Standard convolutional models output raw classification probabilities without spatial interpretability.
- **Overconfidence**: Softmax probabilities are frequently misinterpreted as clinical diagnostic likelihoods.
- **Methodological vulnerabilities**: Public datasets often contain class imbalances, slice-level patient leakage, and lack volumetric 3D context.
- **Generic LLM hallucinations**: Conventional chatbots lack grounding in verified neuropathology literature and risk generating speculative medical claims.

NeuroScan AI tackles these challenges within an academic demonstration setting by coupling PyTorch vision models with spatial saliency maps (Grad-CAM), verified RAG grounding, and multi-stage LangGraph safety guardrails.

---

## 3. Project Objectives

1. **Deep Learning Classification**: Fine-tune a lightweight, high-efficiency convolutional backbone (EfficientNet-B0) on 2D axial MRI slices using inverse-frequency class-weighted cross-entropy loss.
2. **Visual Interpretability (XAI)**: Implement native zero-dependency PyTorch Grad-CAM to expose spatial gradient activations in the final convolutional layers.
3. **Transparent Language Explanations**: Translate quantitative probabilities into understandable educational reports using Google Gemini, backed by an offline local synthesis engine.
4. **Authoritative RAG Assistant**: Ingest peer-reviewed neuropathology and radiological literature to answer student inquiries with verifiable citations.
5. **Multi-Stage Workflow Coordination**: Build a LangGraph workflow that orchestrates input validation, intent classification, context preparation, retrieval, generation, and safety auditing.
6. **Academic Rigor & Safety**: Emphasize technical constraints (data leakage, minority imbalance, 2D constraints) and enforce non-diagnostic boundaries.

---

## 4. Main Features

- 🧠 **Four-Class MRI Inference**: Real-time inference on uploaded 2D axial scans (PNG, JPG, JPEG) or pre-configured instant demo scans.
- 🔬 **Grad-CAM Attention Heatmaps**: Thermal overlays visualizing convolutional feature activations mapped to brain parenchyma.
- 💡 **RAG Knowledge Assistant**: Interactive chat grounded in 8 curated educational documents (NIA, Radiopaedia, ICML literature).
- 🔄 **LangGraph Reasoning Pipeline**: 6-stage operational graph coordinating validation, intent routing, retrieval, and safety.
- 🛡️ **Dual-Engine Fault Tolerance**: Native LangGraph and ChromaDB execution in compatible environments with seamless fallback to pure-Python state machines and lexical search in constrained setups.
- 🌐 **Bilingual Explanations**: Native support for English and conversational Hinglish (Latin script).
- 📊 **Interactive Evaluation Dashboard**: Live metrics, confusion matrix breakdowns, and minority class scrutiny ($N=9$ held-out test scans).

---

## 5. Technology Stack

| Domain | Technology / Library | Version | Role in Project |
| :--- | :--- | :--- | :--- |
| **Deep Learning** | PyTorch / Torchvision | `2.9.0+cu130` / `0.24.0` | Tensor operations, model loading, GPU inference, backward hooks |
| **Model Backbone** | EfficientNet-B0 | 5.3M parameters | Transfer learning feature extractor pretrained on ImageNet |
| **Web Interface** | Streamlit | `1.51.0` | Medical workstation UI, dark theme, interactive cards |
| **Visual XAI** | Native PyTorch Grad-CAM | Custom | Gradient feature extraction from `model.features[-1]` |
| **Vector Database** | ChromaDB | `1.5.9` | Dense vector indexing with cosine distance similarity |
| **Embeddings** | Sentence-Transformers | `all-MiniLM-L6-v2` | 384-dimensional dense semantic embeddings |
| **Reasoning Graph** | LangGraph | `1.2.20` | Multi-stage StateGraph routing and state management |
| **LLM Provider** | Google Gemini REST API | `gemini-1.5-flash` | Text synthesis with 15s timeout and transient retry backoff |
| **Data & Plotting** | NumPy / Matplotlib / PIL | `2.1.2` / `3.10.7` | Image preprocessing, colormap blending, visualization |

---

## 6. System Architecture

```text
+-----------------------------------------------------------------------------------+
|                                  USER INTERFACE                                   |
|              Streamlit 1.51 Web Application (app.py — Port 8501)                 |
+----------------------------------------+------------------------------------------+
                                         |
            +----------------------------+----------------------------+
            |                                                         |
            v                                                         v
+-------------------------------+                         +-------------------------+
|      VISION & XAI ENGINE      |                         |    LANGGRAPH WORKFLOW   |
| (PyTorch EfficientNet-B0)     |                         | (Multi-Stage Reasoning) |
+---------------+---------------+                         +------------+------------+
                |                                                      |
    +-----------+-----------+                      +-------------------+-------------------+
    |                       |                      |                   |                   |
    v                       v                      v                   v                   v
[224x224 Bilinear]    [Grad-CAM Hook]       [1. Validation]     [2. Intent]       [3. Context Prep]
    |                       |                      |                   |                   |
    v                       v                      +-------------------+-------------------+
[Inference Logits]    [Thermal Overlay]                                |
    |                                                                  v
    +----------------------------------------------------> [4. Knowledge Retrieval]
                                                                       | (Dense / Keyword)
                                                                       v
                                                           [5. Grounded Generation]
                                                                       | (Gemini / Offline)
                                                                       v
                                                           [6. Safety Validation]
                                                                       |
                                                                       v
                                                            [Synthesized Response]
```

---

## 7. Dataset Overview & Class Distribution

The project utilizes a preprocessed dataset comprising **6,400 standalone 2D axial brain MRI slices** across four categories:

| Class Name | Total Scans | Proportion | Held-Out Test ($N=960$) | Category Characterization in Dataset |
| :--- | :--- | :--- | :--- | :--- |
| `Non_Demented` | 3,200 | 50.0% | 480 | Scans labeled with no apparent structural dementia markers. |
| `Very_Mild_Demented` | 2,240 | 35.0% | 336 | Scans labeled with early, subtle structural indicators (resembling MCI). |
| `Mild_Demented` | 896 | 14.0% | 135 | Scans exhibiting noticeable cortical and ventricular changes. |
| `Moderate_Demented` | 64 | **1.0%** | **9** | **Extreme minority class** exhibiting marked anatomical atrophy. |

---

## 8. Model Architecture & Transfer Learning

1. **Backbone Selection**: **EfficientNet-B0** (Mingxing Tan & Quoc V. Le, ICML 2019).
2. **Compound Scaling**: Scales network depth ($d$), width ($w$), and input resolution ($r$) uniformly using a compound coefficient $\phi$:
   $$\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi$$
3. **Efficiency**: Leverages Mobile Inverted Bottleneck Convolution (**MBConv**) blocks with depthwise separable convolutions and Squeeze-and-Excitation (SE) optimization (~5.3M parameters).
4. **Transfer Learning Adaptation**:
   - Initialized with ImageNet-pretrained weights.
   - Replaced original 1,000-class linear layer with:
     ```python
     nn.Sequential(
         nn.Dropout(p=0.2),
         nn.Linear(in_features=1280, out_features=4)
     )
     ```
   - Input shape: $3 \times 224 \times 224$ normalized using standard ImageNet mean and standard deviation.

---

## 9. Training & Evaluation Summary

Trained using inverse-frequency class-weighted cross-entropy loss and Adam optimizer with cosine annealing:

### Quantitative Test Split Metrics ($N=960$)
- **Test Accuracy**: **95.52%** (917 / 960 held-out scans correctly classified)
- **Macro F1-Score**: **0.9486** (unweighted mean across all 4 classes)
- **Weighted F1-Score**: **0.9553**
- **Test Specificity**: **98.24%** (Macro One-vs-Rest)
- **Best Validation F1**: **0.9780** (Checkpoint: `artifacts/best_efficientnet_b0.pth`, Epoch 15)

### Per-Class Performance Breakdown
| Class Name | Precision | Recall | F1-Score | Specificity | Test Sample ($N$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Mild_Demented` | 89.86% | 98.52% | 0.9399 | 98.18% | 135 |
| `Moderate_Demented` | 100.00% | 88.89% | 0.9412 | 100.00% | **9** (8 TP, 1 FN, 0 FP) |
| `Non_Demented` | 96.85% | 96.04% | 0.9644 | 96.88% | 480 |
| `Very_Mild_Demented` | 96.04% | 93.75% | 0.9488 | 97.92% | 336 |

> **Academic Rigor Note**: While held-out test accuracy is 95.52%, the `Moderate_Demented` class contains only 9 test scans. Each single prediction shifts class recall by 11.1%. Furthermore, these results reflect the public dataset split and do not constitute certified clinical diagnostic performance.

---

## 10. Grad-CAM Explainability

**Gradient-weighted Class Activation Mapping** (Selvaraju et al., ICCV 2017) provides visual interpretability:
1. **Target Layer**: Final convolutional layer before pooling (`model.features[-1]`).
2. **Gradients**: Computes spatial gradients of the predicted class score $y^c$ with respect to feature activation maps $A^k$:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
3. **Linear Combination & ReLU**:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
4. **Visualization**: Bilinearly upsampled to $224 \times 224$ and overlaid with a thermal jet/turbo colormap.
5. **Interpretability Reality**: Highlights mathematical feature activations driving the network's logits; it does **not** trace biological lesion margins or clinical pathology.

---

## 11. RAG Knowledge Base Architecture

The RAG subpackage (`src/neuroscan/rag/`) grounds educational answers in authoritative sources:
- **Curated Knowledge Base (`data/knowledge_base/`)**:
  - `alzheimers_overview.md` (National Institute on Aging / Mayo Clinic)
  - `dementia_stages.md` (Alzheimer's Association / Reisberg GDS)
  - `brain_mri_basics.md` (Radiopaedia / NINDS)
  - `axial_mri_slices.md` (Radiopaedia / RadiologyInfo.org)
  - `efficientnet_b0.md` (Tan & Le ICML 2019 / PyTorch)
  - `gradcam_explainability.md` (Selvaraju et al. ICCV 2017)
  - `dataset_and_model_limitations.md` (NeuroScan AI Technical Audits)
  - `medical_safety_and_disclaimers.md` (WHO / American College of Radiology)
- **Sentence-Aware Semantic Chunking**: Paragraph-level chunking preserving complete grammatical sentences without cutting mid-word.
- **Dual Retrieval Engine**:
  - Dense semantic search using ChromaDB & `all-MiniLM-L6-v2`.
  - In-memory lexical search with stopword filtering, domain synonym expansion, and Jaccard deduplication when vector libraries are missing.

---

## 12. Google Gemini Integration & Fallback

- **API Interface**: Google Generative Language REST API (`gemini-1.5-flash`).
- **Safety Boundary**: **Zero image transmission**. Only structured text strings (prediction labels, percentages) or educational text are transmitted.
- **Resilience**: 15-second interactive timeout with automatic single-retry exponential backoff for transient HTTP 503/429 errors.
- **Local Fallback Engine**: If credentials are unset or network requests fail, a topic-aware synthesis engine synthesizes human-readable Markdown explanations with complete tables and disclaimers directly from the knowledge base.

---

## 13. LangGraph Multi-Stage Reasoning Workflow

The workflow subpackage (`src/neuroscan/workflow/`) coordinates end-to-end processing across six sequential stages:

```text
[Input Query] ---> [1. Input Validation]
                           |
                           v
                  [2. Intent Routing]
                           |
                           v
                [3. Context Preparation]
                           |
                           v
                [4. Knowledge Retrieval]
                           |
                           v
                 [5. Answer Generation]
                           |
                           v
                 [6. Safety Validation] ---> [Final Response]
```

### Supported Query Intents
1. `medical_advice`: Triggers immediate clinical disclaimer directing user to licensed neurologists.
2. `mri_explanation`: Explains structured model predictions without altering quantitative values.
3. `gradcam_explanation`: Explains visual heatmaps and spatial feature activations.
4. `limitations`: Explains dataset imbalance, patient leakage, and 2D constraints.
5. `educational_question`: Answers general pathology and technical inquiries via RAG.
6. `out_of_scope`: Politely explains that the knowledge base does not cover unrelated topics.

---

## 14. Application Setup

### Prerequisites
- Windows 10/11, macOS, or Linux.
- Conda (Miniconda / Anaconda) installed.
- NVIDIA GPU with CUDA support recommended (CPU mode supported).

### Clone Repository
```powershell
git clone https://github.com/MohitJaiswal2507/Alzheimer-Detection.git
cd Alzheimer-Detection
```

---

## 15. Environment Setup

### Recommended Setup (Conda Environment)
```powershell
# Create environment with Python 3.10
conda create -n tgpu python=3.10.19 -y
conda activate tgpu

# Install PyTorch matching your CUDA runtime (example: CUDA 12.1 / 13.0)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### Configure API Keys (Optional for Live LLM)
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```
Add your Google Gemini API key:
```env
GEMINI_API_KEY="your_actual_api_key_here"
GEMINI_MODEL=gemini-1.5-flash
```
*(Note: If no API key is provided, the application automatically runs in deterministic local knowledge base fallback mode).*

---

## 16. How to Launch Streamlit

Run the Streamlit application using the environment's Python executable:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 17. How to Use the Application

1. **Dashboard (`📊 Dashboard`)**:
   - Inspect held-out test metrics, confusion matrix, precision/recall, and minority class distributions.
2. **MRI Analysis (`🔬 MRI Analysis`)**:
   - Select an **Instant Demo Scan** from the dropdown (`Moderate_Demented`, `Mild_Demented`, etc.) or upload a 2D axial MRI scan (PNG, JPG).
   - Click **🚀 Analyze MRI**.
   - View predicted class, confidence meters, Grad-CAM attention heatmap overlay, and clinical context explainer.
3. **Knowledge Assistant (`💡 Knowledge Assistant`)**:
   - Select response language (English or Hindi/Hinglish).
   - Click quick suggestion buttons or enter custom questions in the chat input.
   - Review grounded answers, citations expander, and workflow telemetry.
4. **About Project (`ℹ️ About Project`)**:
   - Review architectural objectives, technology stack, and full technical limitations.

---

## 18. Critical Project & Dataset Limitations

1. **Patient-Level Data Leakage**: The source Kaggle dataset contains sequential numeric filenames (`mild_1.jpg`) without patient identifiers. While train/val/test splits have zero identical image overlap, multiple slices from the same subject may exist across splits.
2. **2D Slices vs. 3D Volumetric Imaging**: The model evaluates standalone 2D axial slices. Clinical neuroimaging relies on multi-planar 3D volumetric sequences (T1/T2/FLAIR) comprising hundreds of slices to reconstruct volumetric atrophy.
3. **Extreme Class Imbalance**: The `Moderate_Demented` class represents only 1% (64 of 6,400 scans) of the dataset. Despite class weighting, sample variance is elevated.
4. **Softmax Score vs. Clinical Probability**: Classifier confidence is a normalized mathematical score relative to training distributions, **not** a clinical probability that an individual has dementia.

---

## 19. Mandatory Medical & Clinical Disclaimer

> **⚠️ IMPORTANT MEDICAL NOTICE**:  
> **NeuroScan AI is strictly an undergraduate engineering coursework demonstration and educational research project.**  
> It has **not** been evaluated in clinical trials, approved by the FDA, EMA, CDSCO, or any regulatory body, and is **not Software as a Medical Device (SaMD)**.  
> Outputs must **never** be interpreted as clinical medical diagnoses, prognoses, or treatment plans. Individuals with cognitive symptoms must consult certified physicians and neurologists.

---

## 20. Future Improvements

- **3D Volumetric CNN / Vision Transformers**: Transitioning from 2D slices to 3D volumetric ResNet or Swin UNETR models using ADNI / OASIS NIfTI datasets.
- **Patient-Level Splitting**: Ingesting datasets with verifiable Subject IDs to eliminate slice leakage.
- **Multi-Modal Biomarkers**: Integrating clinical tabular metrics (MMSE scores, APOE-e4 genetics) with imaging data.
- **Calibration**: Applying Platt scaling or temperature scaling against clinical cohorts.

---

## 21. Repository Structure

```text
Alzheimer-Detection/
├── app.py                            # Streamlit medical AI workstation UI
├── requirements.txt                  # Consolidated dependency manifest
├── README.md                         # Presentation-ready academic documentation
├── .env.example                      # Configuration and environment template
├── artifacts/
│   ├── best_efficientnet_b0.pth      # Fine-tuned PyTorch model weights (Epoch 15)
│   └── chroma_db/                    # Persistent local ChromaDB vector store
├── data/
│   ├── Alzheimer/                    # Raw Kaggle 4-class image dataset (6,400 scans)
│   └── knowledge_base/               # 8 Curated educational Markdown documents
├── docs/
│   ├── PROJECT_ARCHITECTURE.md       # Comprehensive system architecture guide
│   ├── DEMO_GUIDE.md                 # 10-Step presentation and viva demonstration guide
│   └── reports/                      # Engineering milestone and audit reports (Phases 0-9)
├── src/
│   └── neuroscan/
│       ├── config.py                 # Paths, taxonomies, and global constants
│       ├── vision/                   # Inference pipeline, preprocessing, model loader
│       ├── explainability/           # Native PyTorch Grad-CAM hook and visualizer
│       ├── llm/                      # Gemini REST API provider and structured explainer
│       ├── rag/                      # Loader, chunker, vector store, retriever, generator
│       └── workflow/                 # LangGraph StateGraph workflow and state nodes
└── tests/
    ├── test_vision.py                # Model inference and Grad-CAM tests (5 tests)
    ├── test_llm_explainer.py         # Classifier explainer & safety tests (14 tests)
    ├── test_rag.py                   # RAG chunking, retrieval, fallback tests (21 tests)
    ├── test_workflow.py              # LangGraph workflow & routing tests (18 tests)
    └── test_ui_helpers.py            # Streamlit visualization helper tests (1 test)
```

---
*NeuroScan AI &bull; Computer Science & Engineering Final Year Project &bull; Academic Year 2026*
