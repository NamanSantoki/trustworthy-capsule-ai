
"""AdaptiveCrossAttentionFusion — Phase 5 research contribution.

Architecture
------------
This module is **backbone-agnostic** — it accepts flat feature vectors from any
CNN backbone (B, cnn_dim) and any Transformer backbone (B, transformer_dim).
Swapping EfficientNet for ConvNeXt, or ViT-Small for Swin, requires no changes
here; only the dim arguments at construction need updating.

Data flow::

    CNN backbone     → (B, cnn_dim)          [e.g. 1280 for EfficientNet-V2-S]
    Trans backbone   → (B, transformer_dim)  [e.g. 384 for ViT-Small]

    Step 1 — Project both modalities to shared d_model space:
        Q  = LayerNorm(Linear(cnn_dim       → d_model))   [CNN  → query]
        KV = LayerNorm(Linear(trans_dim     → d_model))   [ViT  → key/value]

    Step 2 — Cross-Attention (CNN queries ViT global context):
        Q3D, KV3D = Q.unsqueeze(1), KV.unsqueeze(1)
        attn_out, weights = MultiheadAttention(Q3D, KV3D, KV3D)
        attn_out = Dropout(attn_out.squeeze(1))            (B, d_model)

    Step 3 — Residual connection + LayerNorm:
        attn_residual = LayerNorm(Q + attn_out)            (B, d_model)

    Step 4 — Adaptive gating (per-sample, per-feature-dim):
        gate_input = cat([Q, attn_residual], dim=1)        (B, 2·d_model)
        alpha      = sigmoid(Linear(2·d_model → d_model))  (B, d_model) ∈ (0,1)
        fused      = alpha⊙attn_residual + (1−alpha)⊙Q    (B, d_model)

        Interpretation: alpha → 1 means "trust the cross-modal ViT context";
        alpha → 0 means "rely on the CNN features alone". The gate is a
        per-dimension learnable switch, not a scalar, giving the model fine-
        grained control over information routing.

    Step 5 — Classification head:
        LayerNorm → Linear(d_model→hidden_dim) → GELU
            → Dropout → Linear(hidden_dim→num_classes)
        → logits (B, num_classes)

Research Claim
--------------
AdaptiveCrossAttentionFusion is designed to:
  (a) outperform ConcatFusion on balanced accuracy and macro-F1 by letting the
      model attend to globally coherent ViT representations while preserving
      local CNN texture evidence;
  (b) produce interpretable gating maps α that can be visualised and cross-
      validated against Grad-CAM / Attention-Rollout heatmaps (Phase 6).

Ablation table
--------------
  efficientnet_only   → baseline_1
  vit_only            → baseline_2
  hybrid_concat       → baseline_3  (ConcatFusion, this phase, untouched)
  hybrid_adaptive     → proposed    (AdaptiveCrossAttentionFusion, Phase 5)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveCrossAttentionFusion(nn.Module):
    """Backbone-agnostic Adaptive Cross-Attention Fusion module.

    Accepts flat feature vectors from **any** CNN and Transformer backbone,
    fuses them via learned cross-attention and per-dimension gating, then
    produces classification logits.

    Args:
        cnn_dim:            Feature dimensionality from the CNN backbone
                            (e.g. 1280 for EfficientNetV2-S).
        transformer_dim:    Feature dimensionality from the Transformer backbone
                            (e.g. 384 for ViT-Small/16).
        num_classes:        Number of output classes.
        d_model:            Shared projection space dimension.  Must be divisible
                            by ``num_heads``.  Default: 256.
        num_heads:          Number of attention heads in ``MultiheadAttention``.
                            Default: 8.
        hidden_dim:         Intermediate dimension in the classification head.
                            Default: 512.
        dropout_rate:       Dropout probability applied after attention and in
                            the classification head.  Default: 0.2.
        attention_dropout:  Dropout on attention weights inside
                            ``MultiheadAttention``.  Default: 0.0.

    Raises:
        ValueError: If ``d_model`` is not divisible by ``num_heads``.

    Example::

        fusion = AdaptiveCrossAttentionFusion(
            cnn_dim=1280, transformer_dim=384,
            num_classes=4, d_model=256, num_heads=8,
        )
        logits = fusion(cnn_feats, vit_feats)   # (B, 4)
    """

    def __init__(
        self,
        cnn_dim: int,
        transformer_dim: int,
        num_classes: int,
        d_model: int = 256,
        num_heads: int = 8,
        hidden_dim: int = 512,
        dropout_rate: float = 0.2,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads}). "
                f"Choose d_model that satisfies d_model % num_heads == 0."
            )

        # ── Step 1: Modality projection layers ───────────────────────────────
        # Both project into the same d_model space so cross-attention is valid.
        # LayerNorm after projection normalises scale differences between CNN
        # and Transformer representations.
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.transformer_proj = nn.Sequential(
            nn.Linear(transformer_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Step 2: Cross-attention (CNN queries ViT context) ─────────────────
        # batch_first=True: expects (B, seq_len, embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(p=dropout_rate)

        # ── Step 3: Residual LayerNorm ─────────────────────────────────────────
        self.residual_norm = nn.LayerNorm(d_model)

        # ── Step 4: Adaptive gating MLP ───────────────────────────────────────
        # Input: concatenation of projected CNN features and attended features.
        # Output: per-dimension gating coefficient α ∈ (0, 1).
        self.gate = nn.Linear(2 * d_model, d_model)

        # ── Step 5: Classification head ───────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

        # Store for diagnostics / repr
        self._cnn_dim = cnn_dim
        self._transformer_dim = transformer_dim
        self._d_model = d_model
        self._num_heads = num_heads
        self._hidden_dim = hidden_dim
        self._num_classes = num_classes

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        cnn_features: torch.Tensor,
        transformer_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute class logits via adaptive cross-attention fusion.

        Args:
            cnn_features:          Tensor ``(B, cnn_dim)`` from any CNN backbone.
            transformer_features:  Tensor ``(B, transformer_dim)`` from any
                                   Transformer backbone.

        Returns:
            Logit tensor ``(B, num_classes)``.
        """
        # Step 1 — Project to shared d_model space
        Q  = self.cnn_proj(cnn_features)            # (B, d_model)
        KV = self.transformer_proj(transformer_features)  # (B, d_model)

        # Step 2 — Cross-attention: CNN features query ViT global context
        # MultiheadAttention expects (B, seq_len, embed_dim) with batch_first=True
        Q3D  = Q.unsqueeze(1)   # (B, 1, d_model)
        KV3D = KV.unsqueeze(1)  # (B, 1, d_model)
        attn_out, _ = self.cross_attention(Q3D, KV3D, KV3D)  # (B, 1, d_model)
        attn_out = self.attn_dropout(attn_out.squeeze(1))     # (B, d_model)

        # Step 3 — Residual connection + LayerNorm
        # Preserves the original CNN query information through the attention block.
        attn_residual = self.residual_norm(Q + attn_out)      # (B, d_model)

        # Step 4 — Adaptive gating (per-sample, per-feature-dimension)
        gate_input = torch.cat([Q, attn_residual], dim=1)     # (B, 2·d_model)
        alpha = torch.sigmoid(self.gate(gate_input))          # (B, d_model) ∈ (0,1)
        fused = alpha * attn_residual + (1.0 - alpha) * Q    # (B, d_model)

        # Step 5 — Classification
        return self.head(fused)                               # (B, num_classes)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def count_parameters(self) -> dict[str, int]:
        """Return trainable parameter counts broken down by sub-component.

        This breakdown is particularly useful for quantifying the computational
        overhead of the cross-attention mechanism relative to ConcatFusion in
        the ablation study section of the final evaluation.

        Returns:
            Dictionary with keys:
            ``cnn_projection``, ``transformer_projection``,
            ``cross_attention``, ``residual_norm``, ``gating_mlp``,
            ``classification_head``, ``total``.
        """
        def _c(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "cnn_projection":          _c(self.cnn_proj),
            "transformer_projection":  _c(self.transformer_proj),
            "cross_attention":         _c(self.cross_attention),
            "residual_norm":           _c(self.residual_norm),
            "gating_mlp":              _c(self.gate),
            "classification_head":     _c(self.head),
            "total":                   _c(self),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"cnn_dim={self._cnn_dim}, "
            f"transformer_dim={self._transformer_dim}, "
            f"d_model={self._d_model}, "
            f"num_heads={self._num_heads}, "
            f"hidden_dim={self._hidden_dim}, "
            f"num_classes={self._num_classes})"
        )
