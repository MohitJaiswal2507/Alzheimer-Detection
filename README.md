# NeuroScan AI

An educational 2D brain MRI classification and explainability web app built for our final-year Computer Science & Engineering project.

> **Disclaimer:** This is a student research prototype for learning and demonstration only. It is **not** a clinical diagnostic tool and should never be used for medical decisions.

---

## What It Does

- **MRI Classification:** Classifies 2D axial brain MRI slices into four dataset categories using a fine-tuned **EfficientNet-B0** model.
- **Visual Attention Maps:** Uses native PyTorch **Grad-CAM** overlays to highlight which regions of the slice influenced the prediction.
- **Educational Summaries:** Explains the model output in plain English or Hinglish using Google Gemini (with a deterministic offline fallback and zero MRI pixel transmission).
- **Knowledge Assistant:** Answers follow-up questions about Alzheimer's symptoms, daily care, and MRI concepts using curated educational documents.

---

## Dataset Overview

The model was trained on **6,400** preprocessed 2D axial brain MRI slices (`128×128` grayscale, resized to `224×224` RGB for EfficientNet-B0), split using a stratified **70% / 15% / 15%** partition (`4,480` train / `960` val / `960` test):

| Dataset Class | Total Images | Share | Held-Out Test (`N=960`) |
| :--- | :---: | :---: | :---: |
| `Non_Demented` | 3,200 | 50.0% | 480 |
| `Very_Mild_Demented` | 2,240 | 35.0% | 336 |
| `Mild_Demented` | 896 | 14.0% | 135 |
| `Moderate_Demented` | 64 | 1.0% | 9 |

**Key Model Results (Epoch 15 checkpoint):**
- **Test Accuracy:** `95.52%` (`917 / 960`)
- **Macro F1-Score:** `0.9486`
- **Average Specificity:** `98.24%`

**Known Limitations:** The dataset contains standalone 2D slices without patient IDs, so patient-level data leakage across splits cannot be ruled out. `Moderate_Demented` also has only 9 test samples, making its recall sensitive to single-image errors.

---

## High-Level Flow

1. Upload a 2D axial MRI image (or pick a preset scan) in the Streamlit UI (`app.py`).
2. `src/neuroscan/vision/inference.py` loads `artifacts/best_efficientnet_b0.pth` and computes 4-class probabilities.
3. `src/neuroscan/explainability/gradcam.py` generates the Grad-CAM heatmap overlay.
4. `src/neuroscan/llm/explainer.py` explains the structured result (without sending any image data to the API).
5. `src/neuroscan/workflow/` + `src/neuroscan/rag/` power the Knowledge Assistant with ChromaDB and an offline fallback.

---

## How to Run Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Configure Gemini API key:**
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
   *(The app still works offline without a key using its built-in fallback summaries.)*

3. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Run the test suite:**
   ```bash
   pytest -q
   ```
