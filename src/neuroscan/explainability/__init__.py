"""
NeuroScan AI — Explainability Package
Native PyTorch Grad-CAM visualizer and figure generator.
"""

from .gradcam import (
    generate_gradcam,
    create_gradcam_figure,
)

__all__ = [
    "generate_gradcam",
    "create_gradcam_figure",
]

