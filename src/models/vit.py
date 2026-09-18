"""Vision Transformer (ViT-Small/16) backbone and standalone classifier for Trustworthy-Capsule-AI."""

from __future__ import annotations
from typing import Dict

import timm
import torch
import torch.nn as nn


class ViTSmallBackbone(nn.Module):
    """ViT-Small/16 feature extractor backbone.

    Extracts a 384-dimensional feature embedding from input images using timm.
    """

    OUT_FEATURES: int = 384

    def __init__(self, pretrained: bool = True, freeze: bool = False) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0,
        )

        if freeze:
            self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global feature vector (B, 384) from input image batch (B, 3, H, W)."""
        return self.backbone(x)

    def freeze(self) -> None:
        """Freeze all backbone parameters (disable gradients)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters (enable gradients)."""
        for param in self.parameters():
            param.requires_grad = True


class ViTSmallClassifier(nn.Module):
    """Standalone ViT-Small/16 classification baseline."""

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 8,
        dropout_rate: float = 0.2,
        freeze_backbone_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.backbone = ViTSmallBackbone(
            pretrained=pretrained,
            freeze=(freeze_backbone_epochs > 0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(ViTSmallBackbone.OUT_FEATURES, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits (B, num_classes) for input images (B, 3, H, W)."""
        feats = self.backbone(x)
        return self.classifier(feats)

    def freeze_backbones(self) -> None:
        """Freeze backbone parameters for initial warm-up epochs."""
        self.backbone.freeze()

    def unfreeze_backbones(self) -> None:
        """Unfreeze backbone parameters for fine-tuning."""
        self.backbone.unfreeze()

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters across components."""
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        classifier_trainable = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "backbone_trainable": backbone_trainable,
            "classifier_trainable": classifier_trainable,
            "total_trainable": total_trainable,
        }
