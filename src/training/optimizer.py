"""Optimizer factory for Trustworthy-Capsule-AI.

Provides ``build_optimizer(cfg, model)`` which constructs an AdamW optimizer
with differential learning rates for backbones vs. classification/fusion heads.

Why Differential Learning Rates are Essential
---------------------------------------------
Pretrained Vision Transformers (ViT) and CNNs already contain rich visual
representations from large-scale pretraining (e.g., ImageNet).
- The newly initialized classification head (or fusion module) requires a standard
  learning rate (e.g. 5e-4 to 1e-3) to learn class decision boundaries quickly.
- In contrast, the pretrained ViT self-attention blocks are sensitive to large
  gradient updates; applying 1e-3 directly causes gradient shock and catastrophic
  forgetting upon unfreezing.
- By assigning a lower learning rate (e.g., 5e-5) to the backbone parameter group,
  the model preserves pretrained features while fine-tuning smoothly.
"""

from __future__ import annotations

import logging
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from src.config import AppConfig

logger = logging.getLogger(__name__)


def build_optimizer(cfg: AppConfig, model: nn.Module) -> Optimizer:
    """Construct an AdamW optimizer with differential learning rates.

    Partitions model parameters into two distinct groups:
    1. **Head / Fusion Group**: Parameters in classifier heads and fusion layers,
       assigned ``lr = cfg.training.learning_rate`` (e.g., 5e-4).
    2. **Backbone Group**: Parameters in CNN and Transformer feature extractors,
       assigned ``lr = cfg.training.backbone_lr`` (e.g., 5e-5).

    Args:
        cfg:   Validated :class:`~src.config.AppConfig` instance.
        model: PyTorch model instance (e.g., ``ViTSmallClassifier``,
               ``EfficientNetV2SClassifier``, or ``HybridCapsuleModel``).

    Returns:
        A configured ``AdamW`` optimizer instance with distinct parameter groups.
    """
    head_lr = cfg.training.learning_rate
    backbone_lr = getattr(cfg.training, "backbone_lr", head_lr * 0.1)
    weight_decay = cfg.training.weight_decay

    backbone_params = []
    head_params = []

    # Identify backbone sub-modules via known architecture attributes
    if hasattr(model, "cnn_backbone") and hasattr(model, "transformer_backbone"):
        # HybridCapsuleModel
        backbone_params.extend(list(model.cnn_backbone.parameters()))
        backbone_params.extend(list(model.transformer_backbone.parameters()))
    elif hasattr(model, "backbone"):
        # Standalone ViTSmallClassifier or EfficientNetV2SClassifier
        backbone_params.extend(list(model.backbone.parameters()))
    else:
        # Fallback: check named modules containing 'backbone'
        for name, module in model.named_children():
            if "backbone" in name.lower():
                backbone_params.extend(list(module.parameters()))

    backbone_param_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]

    # Build parameter groups
    param_groups = []
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay,
            "name": "head",
        })

    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
            "name": "backbone",
        })

    optimizer = AdamW(param_groups)

    head_count = sum(p.numel() for p in head_params)
    backbone_count = sum(p.numel() for p in backbone_params)
    logger.info(
        f"Optimizer initialized: Head group ({head_count:,} params, lr={head_lr:.2e}), "
        f"Backbone group ({backbone_count:,} params, lr={backbone_lr:.2e}), "
        f"weight_decay={weight_decay}"
    )

    return optimizer
