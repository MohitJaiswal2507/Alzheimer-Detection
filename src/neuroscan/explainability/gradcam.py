"""
NeuroScan AI — Grad-CAM Explainability Module
Generates gradient-weighted class activation heatmaps using native PyTorch hooks.
Labels visualizations as mathematical model influence, not diagnostic pathology.
"""
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """
    Computes a normalized Grad-CAM heatmap for the final convolutional layer
    of EfficientNet-B0 (model.features[-1]).

    Args:
        model: Trained EfficientNet-B0 model in eval mode.
        input_tensor: Preprocessed tensor of shape (1, 3, 224, 224).
        target_class: Target class index. If None, uses model's top prediction.

    Returns:
        2D numpy array of shape (224, 224) normalized between 0.0 and 1.0.
    """
    target_layer = model.features[-1]
    activations = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    f_handle = target_layer.register_forward_hook(forward_hook)
    b_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad()
        x = input_tensor.clone().detach().requires_grad_(True)
        logits = model(x)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[0, target_class]
        score.backward()

        # Global average pooling over spatial dimensions
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # Min-max normalization
        c_min, c_max = float(cam.min()), float(cam.max())
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam)

        return cam
    finally:
        f_handle.remove()
        b_handle.remove()


def create_gradcam_figure(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    predicted_class: str,
    confidence: float,
) -> plt.Figure:
    """
    Renders a 3-panel figure showing Original MRI, Grad-CAM Heatmap, and Overlay.

    Args:
        pil_image: Original uploaded PIL image.
        heatmap: 2D numpy array of shape (224, 224).
        predicted_class: Name of predicted class.
        confidence: Confidence score between 0.0 and 1.0.

    Returns:
        Matplotlib Figure object.
    """
    resized_img = pil_image.convert("RGB").resize((224, 224))
    clean_class = predicted_class.replace("_", " ")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), facecolor="#0e1117")

    # Panel 1: Original MRI
    axes[0].imshow(resized_img, cmap="gray")
    axes[0].set_title("1. Original Scan (224x224)", color="#e0e0e0", fontsize=10, pad=8)
    axes[0].axis("off")

    # Panel 2: Grad-CAM Heatmap
    im = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(f"2. Saliency Heatmap\n{clean_class} ({confidence * 100:.1f}%)", color="#e0e0e0", fontsize=10, pad=8)
    axes[1].axis("off")
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#e0e0e0", labelcolor="#e0e0e0")

    # Panel 3: Alpha-Blended Overlay
    axes[2].imshow(resized_img, cmap="gray")
    axes[2].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[2].set_title("3. Overlay Visualizer\n(Model Attention Influence)", color="#e0e0e0", fontsize=10, pad=8)
    axes[2].axis("off")

    plt.tight_layout()
    return fig

