"""
NeuroScan AI — Configuration Module
Constants, paths, and environment settings for the project.
"""
import os
from pathlib import Path

# Load local .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

# Verified 4-class taxonomy from Phases 1-4
CLASS_NAMES = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented",
]

# Neutral educational descriptions of each dataset category
CLASS_DESCRIPTIONS = {
    "Non_Demented": "Dataset category labeled 'Non_Demented' in this 4-class collection.",
    "Very_Mild_Demented": "Dataset category labeled 'Very_Mild_Demented' in this 4-class collection.",
    "Mild_Demented": "Dataset category labeled 'Mild_Demented' in this 4-class collection.",
    "Moderate_Demented": "Dataset category labeled 'Moderate_Demented' in this 4-class collection (a minority class comprising 64 of 6,400 total images).",
}

# Configured LLM provider settings (Google Gemini)
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
