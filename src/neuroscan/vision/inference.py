"""
NeuroScan AI — Vision Inference Module
Loads the trained EfficientNet-B0 checkpoint, handles image preprocessing,
and generates class predictions with calibrated confidence scores.
"""
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import torchvision.models as models
from PIL import Image

from ..config import PROJECT_ROOT, ARTIFACTS_DIR, CLASS_NAMES


# Standard ImageNet normalization used during Phase 2-4
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform() -> T.Compose:
    """Returns the deterministic evaluation transform from Phase 2."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_trained_model(
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[int, str], Dict[str, int], Dict[str, Any]]:
    """
    Loads the trained EfficientNet-B0 classifier from disk.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
                         Defaults to artifacts/best_efficientnet_b0.pth.
        device: Torch device (defaults to auto-detected device).

    Returns:
        Tuple of (model, idx_to_class, class_to_idx, metadata_dict).

    Raises:
        FileNotFoundError: If the checkpoint file cannot be found.
    """
    if checkpoint_path is None:
        checkpoint_path = ARTIFACTS_DIR / "best_efficientnet_b0.pth"

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Trained model checkpoint not found at: {checkpoint_path.resolve()}\n"
            "Please ensure artifacts/best_efficientnet_b0.pth exists before running the app."
        )

    if device is None:
        device = get_device()

    # Load weights onto device safely
    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_to_idx = checkpoint.get("class_to_idx", {name: i for i, name in enumerate(CLASS_NAMES)})
    idx_to_class = checkpoint.get("idx_to_class", {i: name for i, name in enumerate(CLASS_NAMES)})
    num_classes = len(idx_to_class)

    # Instantiate EfficientNet-B0 without downloading weights
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    metadata = {
        "epoch": checkpoint.get("epoch", 15),
        "val_f1": checkpoint.get("val_f1", 0.9780),
        "val_acc": checkpoint.get("val_acc", 0.9708),
        "device": str(device),
        "checkpoint_path": str(checkpoint_path.resolve()),
    }

    return model, idx_to_class, class_to_idx, metadata


def preprocess_image(pil_image: Image.Image, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Prepares a PIL Image for model inference.

    Args:
        pil_image: PIL Image in any color mode.
        device: Torch device.

    Returns:
        Tensor of shape (1, 3, 224, 224) ready for inference.
    """
    if device is None:
        device = get_device()

    # Convert grayscale or RGBA to RGB for EfficientNet-B0
    rgb_image = pil_image.convert("RGB")
    transform = get_transform()
    tensor = transform(rgb_image).unsqueeze(0).to(device)
    return tensor


def predict_mri(
    model: nn.Module,
    input_tensor: torch.Tensor,
    idx_to_class: Dict[int, str],
) -> Dict[str, Any]:
    """
    Executes model inference on a preprocessed tensor.

    Args:
        model: Trained EfficientNet-B0 model.
        input_tensor: Tensor of shape (1, 3, 224, 224).
        idx_to_class: Mapping from integer indices to class strings.

    Returns:
        Structured classifier result dictionary:
        - predicted_class: string label
        - confidence: float score between 0.0 and 1.0
        - probabilities: dict mapping all 4 classes to probabilities
        - predicted_index: integer index
    """
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    predicted_class = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx].item())

    probabilities = {
        idx_to_class[i]: round(float(probs[i].item()), 4)
        for i in range(len(idx_to_class))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "predicted_index": pred_idx,
    }

