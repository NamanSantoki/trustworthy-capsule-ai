"""Abstract base class for all explainability algorithms in Trustworthy Capsule AI."""

from abc import ABC, abstractmethod
from typing import Optional, Union, List
import torch
import torch.nn as nn


class BaseExplainer(ABC):
    """Base interface for model explainer algorithms (Grad-CAM, Score-CAM, Attention Rollout, etc.)."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        """Initialize the explainer.

        Args:
            model: PyTorch classification model.
            target_layer: Specific target layer module for feature extraction / attribution.
        """
        self.model = model
        self.target_layer = target_layer

    @abstractmethod
    def attribute(self, input_tensor: torch.Tensor, target_category: Optional[Union[int, List[int]]] = None) -> torch.Tensor:
        """Generate attribution map / heatmap tensor for given input.

        Args:
            input_tensor: Input image tensor of shape (B, C, H, W).
            target_category: Target class index or list of target indices. If None, uses top predicted class.

        Returns:
            Attribution heatmap tensor scaled in range [0, 1] of shape (B, 1, H, W) or (B, H, W).
        """
        pass

    def __call__(self, input_tensor: torch.Tensor, target_category: Optional[Union[int, List[int]]] = None) -> torch.Tensor:
        """Convenience caller forwarding to attribute()."""
        return self.attribute(input_tensor, target_category=target_category)
