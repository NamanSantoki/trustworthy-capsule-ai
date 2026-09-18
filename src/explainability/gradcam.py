"""Grad-CAM implementation for PyTorch CNN backbones and hybrid models.

Supports EfficientNet baseline as well as Hybrid Concat and Hybrid Adaptive models.
"""

from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.explainability.base import BaseExplainer


class GradCAM(BaseExplainer):
    """Gradient-weighted Class Activation Mapping (Grad-CAM).

    Computes activation heatmaps by combining feature maps with target gradients
    with respect to target layer activations.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Initialize Grad-CAM.

        Args:
            model: PyTorch model (EfficientNet baseline or Hybrid model).
            target_layer: The target convolutional layer (e.g. model.features[-1] or model.cnn_backbone.features[-1]).
        """
        super().__init__(model=model, target_layer=target_layer)
        if target_layer is None:
            raise ValueError("Grad-CAM requires a valid target_layer.")

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles: list = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        if self.target_layer is not None:
            self.handles.append(self.target_layer.register_forward_hook(forward_hook))
            self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        """Remove PyTorch hooks to avoid memory leaks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def attribute(self, input_tensor: torch.Tensor, target_category: Optional[Union[int, List[int]]] = None) -> torch.Tensor:
        """Generate Grad-CAM heatmaps for input_tensor.

        Args:
            input_tensor: Input image tensor of shape (B, C, H, W).
            target_category: Target class index (int) or list of indices. If None, argmax of model logits is used.

        Returns:
            Heatmap tensor of shape (B, 1, H, W) normalized to [0, 1].
        """
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        was_training = self.model.training
        self.model.eval()

        # Ensure input tensor requires grad or backprop propagates
        input_tensor = input_tensor.detach().requires_grad_(True)

        self.activations = None
        self.gradients = None

        # Enable gradient computation locally even if called in torch.no_grad() context
        with torch.enable_grad():
            logits = self.model(input_tensor) # (B, num_classes)
            batch_size, num_classes = logits.shape

            if target_category is None:
                target_categories = torch.argmax(logits, dim=1).tolist()
            elif isinstance(target_category, int):
                target_categories = [target_category] * batch_size
            elif isinstance(target_category, (list, tuple)):
                target_categories = list(target_category)
                if len(target_categories) != batch_size:
                    raise ValueError(f"Length of target_category list ({len(target_categories)}) must match batch size ({batch_size}).")
            else:
                raise TypeError("target_category must be None, int, or list of ints.")

            # Compute score sum for targeted categories across batch
            self.model.zero_grad()
            score = sum(logits[i, target_categories[i]] for i in range(batch_size))
            score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            self.model.train(was_training)
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        activations = self.activations.detach() # (B, C, H_feat, W_feat)
        gradients = self.gradients.detach()    # (B, C, H_feat, W_feat)

        # Global average pooling of gradients per channel -> weights (B, C, 1, 1)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True) # (B, 1, H_feat, W_feat)

        # Apply ReLU to retain positive contributions
        cam = F.relu(cam)

        # Interpolate to match original image dimensions (B, 1, H, W)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False)

        # Min-max normalization per sample in batch
        B, C, H, W = cam.shape
        cam_flat = cam.view(B, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        cam_max = cam_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)

        denominator = cam_max - cam_min
        denominator[denominator == 0] = 1e-7 # Prevent division by zero
        cam_normalized = (cam - cam_min) / denominator

        self.model.train(was_training)
        return cam_normalized

    def __del__(self) -> None:
        self.remove_hooks()
