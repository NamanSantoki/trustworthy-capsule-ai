"""Utility functions for explainability module in Trustworthy Capsule AI."""

import os
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from src.config import AppConfig


def get_target_layer(model: nn.Module, target_layer_path: Optional[str] = None) -> nn.Module:
    """Find and return the target convolutional layer in the model.

    If target_layer_path is specified as a dot-separated string (e.g. 'backbone.features.7'), it resolves that attribute.
    Otherwise, automatically locates the last Conv2d or feature block of EfficientNet in both standalone and hybrid architectures.

    Args:
        model: PyTorch model instance.
        target_layer_path: Optional dot-separated attribute path string.

    Returns:
        Target target layer module.

    Raises:
        ValueError: If layer path is invalid or target layer cannot be found.
    """
    if target_layer_path:
        curr = model
        for name in target_layer_path.split("."):
            if name.isdigit():
                curr = curr[int(name)]
            elif hasattr(curr, name):
                curr = getattr(curr, name)
            else:
                raise ValueError(f"Target layer path '{target_layer_path}' not found on model.")
        return curr

    # Automatic selection strategy:
    # 1. Standalone EfficientNet (EfficientNetV2SClassifier)
    if hasattr(model, "backbone") and hasattr(model.backbone, "features"):
        return model.backbone.features[-1]

    # 2. Direct features attribute
    if hasattr(model, "features"):
        return model.features[-1]

    # 3. Hybrid model (HybridCapsuleModel) -> cnn_backbone.features
    if hasattr(model, "cnn_backbone") and hasattr(model.cnn_backbone, "features"):
        return model.cnn_backbone.features[-1]

    # 4. Search recursively for the last Conv2d layer
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is not None:
        return last_conv

    raise ValueError("Could not automatically determine target layer for Grad-CAM. Please pass target_layer_path.")


def load_model_checkpoint(model: nn.Module, checkpoint_path: str, device: Union[str, torch.device] = "cpu") -> nn.Module:
    """Load trained weights from checkpoint into model.

    Args:
        model: PyTorch model instance.
        checkpoint_path: File path to .pth checkpoint.
        device: Device to map storage to.

    Returns:
        Model loaded with state dict in eval mode.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Invalid checkpoint format.")

    model.to(device)
    model.eval()
    return model


def preprocess_image(
    image_input: Union[str, Image.Image, np.ndarray],
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.57362, 0.36142, 0.18508),
    std: Tuple[float, float, float] = (0.20614, 0.15146, 0.10351),
) -> Tuple[torch.Tensor, np.ndarray]:
    """Preprocess image for input into PyTorch model.

    Args:
        image_input: File path string, PIL Image, or numpy array (H, W, 3).
        img_size: Target square dimension.
        mean: Normalization RGB mean.
        std: Normalization RGB std.

    Returns:
        Tuple of (preprocessed_tensor (1, 3, img_size, img_size), original_rgb_array (img_size, img_size, 3) in [0, 255]).
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image path not found: {image_input}")
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    pil_img = pil_img.resize((img_size, img_size), Image.BILINEAR)
    orig_rgb = np.array(pil_img)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    tensor_img = transform(pil_img).unsqueeze(0) # (1, 3, H, W)
    return tensor_img, orig_rgb


def overlay_heatmap(
    heatmap: Union[torch.Tensor, np.ndarray],
    original_rgb: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "COLORMAP_JET",
) -> Tuple[np.ndarray, np.ndarray]:
    """Overlay normalized 2D heatmap on RGB image.

    Args:
        heatmap: 2D heatmap array or tensor (H, W) scaled in [0, 1].
        original_rgb: Original image array (H, W, 3) with values in range [0, 255] uint8.
        alpha: Blending weight for heatmap overlay.
        colormap: OpenCV colormap string or int (e.g. COLORMAP_JET, COLORMAP_VIRIDIS).

    Returns:
        Tuple of (overlay_rgb, heatmap_colored_rgb).
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.squeeze().detach().cpu().numpy()

    if heatmap.ndim == 3 and heatmap.shape[0] == 1:
        heatmap = heatmap.squeeze(0)

    # Ensure heatmap is scaled [0, 255] uint8
    heatmap_norm = np.clip(heatmap, 0, 1)
    heatmap_uint8 = np.uint8(255 * heatmap_norm)

    if isinstance(colormap, str):
        cmap_code = getattr(cv2, colormap, cv2.COLORMAP_JET)
    else:
        cmap_code = colormap

    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, cmap_code)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

    if original_rgb.shape[:2] != heatmap_colored_rgb.shape[:2]:
        heatmap_colored_rgb = cv2.resize(heatmap_colored_rgb, (original_rgb.shape[1], original_rgb.shape[0]))

    overlay = cv2.addWeighted(original_rgb, 1.0 - alpha, heatmap_colored_rgb, alpha, 0)
    return overlay, heatmap_colored_rgb


def save_explainability_output(
    original_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    save_path: str,
    save_overlay: bool = True,
    save_side_by_side: bool = True,
    metadata: Optional[dict] = None,
) -> None:
    """Save visualization outputs to file system.

    Args:
        original_rgb: Original RGB array (H, W, 3).
        overlay_rgb: Heatmap overlay RGB array (H, W, 3).
        heatmap_rgb: Pure heatmap RGB array (H, W, 3).
        save_path: Base file path or target file path.
        save_overlay: If True, saves overlay image.
        save_side_by_side: If True, saves side-by-side comparison (Original | Heatmap | Overlay).
        metadata: Optional dictionary of explanation metadata (predicted class, target class, layer name, etc.).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    base, ext = os.path.splitext(save_path)
    if not ext:
        ext = ".png"

    if save_overlay:
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{base}_overlay{ext}", overlay_bgr)

    if save_side_by_side:
        combined_rgb = np.hstack([original_rgb, heatmap_rgb, overlay_rgb])
        combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{base}_side_by_side{ext}", combined_bgr)

    if metadata is not None:
        import json
        with open(f"{base}_meta.json", "w") as f:
            json.dump(metadata, f, indent=2)

