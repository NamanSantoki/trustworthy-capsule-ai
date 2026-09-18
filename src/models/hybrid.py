"""Hybrid Capsule Model combining CNN and Vision Transformer backbones with a fusion module."""

from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn


class HybridCapsuleModel(nn.Module):
    """Composable Hybrid Architecture for Trustworthy-Capsule-AI.

    Fuses feature representations from a convolutional neural network (EfficientNetV2-S)
    and a Vision Transformer (ViT-Small/16) using a pluggable fusion module
    (e.g., ConcatFusion, AdaptiveCrossAttentionFusion, ClassAwareAdaptiveFusion).
    """

    def __init__(
        self,
        cnn_backbone: nn.Module,
        transformer_backbone: nn.Module,
        fusion_module: nn.Module,
        freeze_backbone_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.cnn_backbone = cnn_backbone
        self.transformer_backbone = transformer_backbone
        self.fusion_module = fusion_module
        self.freeze_backbone_epochs = freeze_backbone_epochs

        # Freeze backbones initially if configured
        if self.freeze_backbone_epochs > 0:
            self.freeze_backbones()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-modal representations and predict class logits.

        Args:
            x: Input image batch of shape (B, 3, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        cnn_feats = self.cnn_backbone(x)
        vit_feats = self.transformer_backbone(x)
        return self.fusion_module(cnn_feats, vit_feats)

    def freeze_backbones(self) -> None:
        """Freeze parameters in both CNN and Transformer backbones."""
        if hasattr(self.cnn_backbone, "freeze"):
            self.cnn_backbone.freeze()
        else:
            for param in self.cnn_backbone.parameters():
                param.requires_grad = False

        if hasattr(self.transformer_backbone, "freeze"):
            self.transformer_backbone.freeze()
        else:
            for param in self.transformer_backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbones(self) -> None:
        """Unfreeze parameters in both CNN and Transformer backbones."""
        if hasattr(self.cnn_backbone, "unfreeze"):
            self.cnn_backbone.unfreeze()
        else:
            for param in self.cnn_backbone.parameters():
                param.requires_grad = True

        if hasattr(self.transformer_backbone, "unfreeze"):
            self.transformer_backbone.unfreeze()
        else:
            for param in self.transformer_backbone.parameters():
                param.requires_grad = True

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters across all components."""
        cnn_trainable = sum(p.numel() for p in self.cnn_backbone.parameters() if p.requires_grad)
        transformer_trainable = sum(p.numel() for p in self.transformer_backbone.parameters() if p.requires_grad)
        fusion_trainable = sum(p.numel() for p in self.fusion_module.parameters() if p.requires_grad)
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "cnn_backbone_trainable": cnn_trainable,
            "transformer_backbone_trainable": transformer_trainable,
            "fusion_module_trainable": fusion_trainable,
            "total_trainable": total_trainable,
        }
