"""Model factory for Trustworthy-Capsule-AI.

This module is the single entry point for instantiating any model variant.
It reads ``cfg.model.variant`` and routes to the appropriate architecture,
assembling backbones and fusion modules as needed.

Supported variants
------------------
  efficientnet_only    → EfficientNetV2SClassifier (standalone CNN baseline)
  vit_only             → ViTSmallClassifier        (standalone ViT baseline)
  hybrid_concat        → HybridCapsuleModel + ConcatFusion (hybrid baseline)
  hybrid_adaptive      → HybridCapsuleModel + AdaptiveCrossAttentionFusion
                         (Phase 5 — class-agnostic adaptive gate)
  hybrid_class_aware   → HybridCapsuleModel + ClassAwareAdaptiveFusion
                         (Phase 6 proposed — class prototype-conditioned gate)

Switching variants
------------------
To switch between variants, change ``cfg.model.variant`` in model.yaml and
re-run ``train.py``.  No other code changes required.

    hybrid_concat   → baseline ablation (ConcatFusion, unchanged)
    hybrid_adaptive → proposed model (AdaptiveCrossAttentionFusion)

AdaptiveCrossAttentionFusion parameters read from config
--------------------------------------------------------
    cfg.model.fusion_d_model    — shared projection space dimension
    cfg.model.fusion_num_heads  — attention heads in MultiheadAttention
    cfg.model.fusion_hidden_dim — classification head intermediate dimension
    cfg.model.fusion_dropout    — attention-weight dropout rate
    cfg.model.dropout_rate      — general dropout (head + post-attention)

Usage
-----
    from src.models.factory import build_model
    model = build_model(cfg)   # returns nn.Module, device-agnostic
"""

import torch.nn as nn
from src.config import AppConfig
from src.models.efficientnet import EfficientNetV2SBackbone, EfficientNetV2SClassifier
from src.models.vit import ViTSmallBackbone, ViTSmallClassifier
from src.models.fusion.concat import ConcatFusion
from src.models.fusion.adaptive import AdaptiveCrossAttentionFusion
from src.models.fusion.class_aware import ClassAwareAdaptiveFusion
from src.models.hybrid import HybridCapsuleModel


def build_model(cfg: AppConfig) -> nn.Module:
    """Instantiate and return the model specified by ``cfg.model.variant``.

    The returned model is always placed on CPU; move it to the target device
    in the calling script with ``model.to(device)``.

    Args:
        cfg: Validated :class:`~src.config.AppConfig` instance.

    Returns:
        An ``nn.Module`` that:
        - Accepts image tensors ``(B, 3, H, W)`` in ``forward()``.
        - Optionally exposes ``freeze_backbones()`` / ``unfreeze_backbones()``
          and the ``freeze_backbone_epochs`` property (used by the Trainer).

    Raises:
        ValueError: If ``cfg.model.variant`` is not a recognised string.
    """
    variant      = cfg.model.variant
    pretrained   = cfg.model.pretrained
    num_classes  = cfg.model.num_classes
    dropout_rate = cfg.model.dropout_rate
    freeze_ep    = cfg.model.freeze_backbone_epochs
    hidden_dim   = cfg.model.fusion_hidden_dim

    # ── Standalone baselines ──────────────────────────────────────────────────

    if variant == "efficientnet_only":
        return EfficientNetV2SClassifier(
            pretrained=pretrained,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            freeze_backbone_epochs=freeze_ep,
        )

    if variant == "vit_only":
        return ViTSmallClassifier(
            pretrained=pretrained,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            freeze_backbone_epochs=freeze_ep,
        )

    # ── Hybrid models ─────────────────────────────────────────────────────────

    if variant == "hybrid_concat":
        cnn = EfficientNetV2SBackbone(pretrained=pretrained)
        vit = ViTSmallBackbone(pretrained=pretrained)
        fusion = ConcatFusion(
            cnn_dim=EfficientNetV2SBackbone.OUT_FEATURES,
            transformer_dim=ViTSmallBackbone.OUT_FEATURES,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
        return HybridCapsuleModel(
            cnn_backbone=cnn,
            transformer_backbone=vit,
            fusion_module=fusion,
            freeze_backbone_epochs=freeze_ep,
        )

    if variant == "hybrid_adaptive":
        cnn = EfficientNetV2SBackbone(pretrained=pretrained)
        vit = ViTSmallBackbone(pretrained=pretrained)
        fusion = AdaptiveCrossAttentionFusion(
            cnn_dim=EfficientNetV2SBackbone.OUT_FEATURES,
            transformer_dim=ViTSmallBackbone.OUT_FEATURES,
            num_classes=num_classes,
            d_model=cfg.model.fusion_d_model,
            num_heads=cfg.model.fusion_num_heads,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attention_dropout=cfg.model.fusion_dropout,
        )
        return HybridCapsuleModel(
            cnn_backbone=cnn,
            transformer_backbone=vit,
            fusion_module=fusion,
            freeze_backbone_epochs=freeze_ep,
        )

    if variant == "hybrid_class_aware":
        cnn = EfficientNetV2SBackbone(pretrained=pretrained)
        vit = ViTSmallBackbone(pretrained=pretrained)
        fusion = ClassAwareAdaptiveFusion(
            cnn_dim=EfficientNetV2SBackbone.OUT_FEATURES,
            transformer_dim=ViTSmallBackbone.OUT_FEATURES,
            num_classes=num_classes,
            d_model=cfg.model.fusion_d_model,
            num_heads=cfg.model.fusion_num_heads,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attention_dropout=cfg.model.fusion_dropout,
            prototype_temperature=cfg.model.class_aware_prototype_temperature,
            normalize_prototypes=cfg.model.class_aware_prototype_normalize,
        )
        return HybridCapsuleModel(
            cnn_backbone=cnn,
            transformer_backbone=vit,
            fusion_module=fusion,
            freeze_backbone_epochs=freeze_ep,
        )

    raise ValueError(
        f"Unknown model variant: '{variant}'. "
        f"Must be one of: efficientnet_only, vit_only, hybrid_concat, "
        f"hybrid_adaptive, hybrid_class_aware."
    )
