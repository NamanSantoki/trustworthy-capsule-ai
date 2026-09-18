"""ConcatFusion — Baseline feature fusion module for Trustworthy-Capsule-AI.

Architecture
------------
CNN features (1280-d) + Transformer features (384-d)
        ↓  torch.cat  → (1664-d)
    LayerNorm(1664)
    Linear(1664 → hidden_dim)
    GELU
    Dropout(dropout_rate)
    Linear(hidden_dim → num_classes)
        ↓
    logits (num_classes)

Ablation role
-------------
This module serves as the **baseline hybrid fusion** strategy. It will be
compared against ``AdaptiveCrossAttentionFusion`` (Phase 6) to quantify the
research contribution of learned cross-modal attention gating.

Interchangeability
------------------
``ConcatFusion`` is passed as the ``fusion_module`` argument to
``HybridCapsuleModel``; swapping it for ``AdaptiveCrossAttentionFusion``
only requires changing one argument in ``build_model()``—no Trainer changes.
"""

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Simple concatenation-based fusion head.

    Concatenates CNN and Transformer feature vectors, then projects through a
    normalisation → hidden → activation → dropout → output pipeline.

    Args:
        cnn_dim:         Feature dimensionality from the CNN backbone.
        transformer_dim: Feature dimensionality from the Transformer backbone.
        hidden_dim:      Intermediate projection dimension (default 512).
        num_classes:     Number of output classes.
        dropout_rate:    Dropout probability before the output layer.
    """

    def __init__(
        self,
        cnn_dim: int,
        transformer_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 8,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        combined_dim = cnn_dim + transformer_dim  # 1280 + 384 = 1664

        self.head = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        cnn_features: torch.Tensor,
        transformer_features: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate features and compute class logits.

        Args:
            cnn_features:         Tensor ``(B, cnn_dim)``.
            transformer_features: Tensor ``(B, transformer_dim)``.

        Returns:
            Logit tensor ``(B, num_classes)``.
        """
        combined = torch.cat([cnn_features, transformer_features], dim=1)
        return self.head(combined)
