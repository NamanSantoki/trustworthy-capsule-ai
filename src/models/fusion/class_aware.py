"""ClassAwareAdaptiveFusion — Phase 6 research contribution.

This module extends ``AdaptiveCrossAttentionFusion`` by conditioning the
adaptive gate on **class prototype similarity**, turning a class-agnostic
gating mechanism into a class-aware one.

Architecture overview
---------------------
The forward data flow reuses all five steps of the base adaptive module and
adds a prototype-similarity step between residual normalisation (Step 3) and
adaptive gating (Step 4).

Shared steps (identical to AdaptiveCrossAttentionFusion)::

    Step 1 — Project both modalities to shared d_model space:
        Q  = LayerNorm(Linear(cnn_dim       → d_model))   CNN  → query
        KV = LayerNorm(Linear(trans_dim     → d_model))   ViT  → key/value

    Step 2 — Cross-Attention (CNN queries ViT global context):
        Q3D, KV3D = Q.unsqueeze(1), KV.unsqueeze(1)
        attn_out, weights = MultiheadAttention(Q3D, KV3D, KV3D)
        attn_out = Dropout(attn_out.squeeze(1))            (B, d_model)

    Step 3 — Residual connection + LayerNorm:
        attn_residual = LayerNorm(Q + attn_out)            (B, d_model)

Novel step (Class-Aware Prototype Similarity)::

    Prototypes:
        P  ∈ ℝ^(num_classes × d_model)          learnable parameters

    Cosine similarity per class:
        sim_c  = (attn_residual · P_c) / (‖attn_residual‖ · ‖P_c‖ + ε)
        sim    ∈ (B, num_classes)

    Temperature-scaled softmax → class-aware weights:
        w      = softmax(sim / τ)                (B, num_classes)  Σ_c w_c = 1

    Class-aware prototype representation:
        P_x    = Σ_c  w_c · P_c                 (B, d_model)

    New gate input — expanded from 512-d to 768-d:
        gate_input = cat([Q, attn_residual, P_x], dim=1)   (B, 3·d_model)
        alpha      = sigmoid(Linear(3·d_model → d_model))  (B, d_model)

    Final fusion (unchanged):
        fused  = alpha⊙attn_residual + (1−alpha)⊙Q         (B, d_model)

    Step 5 — Classification head (unchanged):
        LayerNorm → Linear(d_model→hidden_dim) → GELU
            → Dropout → Linear(hidden_dim→num_classes)
        → logits (B, num_classes)

Research claim
--------------
By conditioning the per-feature gate on class prototype similarity, the model
can route information more selectively depending on which class the current
sample resembles.  The prototype weights ``w`` are interpretable: they indicate
which class the fused representation is most similar to *before* the final
classification layer, providing an additional XAI signal.

This is the fifth ablation point:
  efficientnet_only     → baseline_1
  vit_only              → baseline_2
  hybrid_concat         → baseline_3
  hybrid_adaptive       → baseline_4  (class-agnostic gate)
  hybrid_class_aware    → proposed    (class-aware prototype gate, this module)

Diagnostics exposed at inference
---------------------------------
Call ``forward_with_diagnostics()`` instead of ``forward()`` to obtain:
    - ``prototype_similarities``  (B, num_classes) — raw cosine similarities
    - ``prototype_weights``       (B, num_classes) — after temperature softmax
    - ``prototype_representation`` (B, d_model)   — weighted prototype P_x
    - ``gate_alpha``              (B, d_model)    — per-feature gate value
    - ``logits``                  (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassAwareAdaptiveFusion(nn.Module):
    """Backbone-agnostic Class-Aware Adaptive Cross-Attention Fusion module.

    Accepts flat feature vectors from any CNN and Transformer backbone, fuses
    them via cross-attention and a class-prototype-conditioned adaptive gate,
    and returns classification logits.

    Args:
        cnn_dim:                Feature dimensionality from CNN backbone
                                (e.g. 1280 for EfficientNetV2-S).
        transformer_dim:        Feature dimensionality from Transformer backbone
                                (e.g. 384 for ViT-Small/16).
        num_classes:            Number of output classes.
        d_model:                Shared projection space dimension.  Must be
                                divisible by ``num_heads``.  Default: 256.
        num_heads:              Attention heads in ``MultiheadAttention``.
                                Default: 8.
        hidden_dim:             Intermediate dimension in classification head.
                                Default: 512.
        dropout_rate:           Dropout probability after attention and in the
                                classification head.  Default: 0.2.
        attention_dropout:      Dropout on attention weights inside
                                ``MultiheadAttention``.  Default: 0.0.
        prototype_temperature:  Temperature τ for softmax over cosine
                                similarities.  Lower = sharper selection.
                                Default: 0.1.
        normalize_prototypes:   If True, L2-normalise prototype vectors before
                                computing cosine similarity (numerically
                                stable).  Default: True.

    Raises:
        ValueError: If ``d_model`` is not divisible by ``num_heads``.

    Example::

        fusion = ClassAwareAdaptiveFusion(
            cnn_dim=1280, transformer_dim=384,
            num_classes=8, d_model=256, num_heads=8,
        )
        logits = fusion(cnn_feats, vit_feats)   # (B, 8)

        # Rich diagnostics at inference:
        result = fusion.forward_with_diagnostics(cnn_feats, vit_feats)
        print(result["prototype_weights"])       # (B, 8) sums to 1
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
        prototype_temperature: float = 0.1,
        normalize_prototypes: bool = True,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads}). "
                f"Choose d_model that satisfies d_model % num_heads == 0."
            )
        if prototype_temperature <= 0:
            raise ValueError(
                f"prototype_temperature must be > 0, got {prototype_temperature}"
            )

        self._cnn_dim            = cnn_dim
        self._transformer_dim    = transformer_dim
        self._d_model            = d_model
        self._num_heads          = num_heads
        self._hidden_dim         = hidden_dim
        self._num_classes        = num_classes
        self._prototype_temperature  = prototype_temperature
        self._normalize_prototypes   = normalize_prototypes

        # ── Step 1: Modality projection layers ───────────────────────────────
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.transformer_proj = nn.Sequential(
            nn.Linear(transformer_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Step 2: Cross-attention (CNN queries ViT context) ─────────────────
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(p=dropout_rate)

        # ── Step 3: Residual LayerNorm ─────────────────────────────────────────
        self.residual_norm = nn.LayerNorm(d_model)

        # ── Prototype bank ────────────────────────────────────────────────────
        # Shape: (num_classes, d_model)
        # Initialised with small random values (Xavier-uniform style via kaiming).
        self.class_prototypes = nn.Parameter(
            torch.empty(num_classes, d_model)
        )
        nn.init.kaiming_uniform_(self.class_prototypes, a=1.0)

        # ── Step 4: Class-aware adaptive gating MLP ───────────────────────────
        # Input: cat([Q, attn_residual, P_x]) = 3 * d_model
        # Output: per-dimension gating coefficient α ∈ (0, 1)
        self.gate = nn.Linear(3 * d_model, d_model)

        # ── Step 5: Classification head ───────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _project_and_attend(
        self,
        cnn_features: torch.Tensor,
        transformer_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Steps 1–3 and return (Q, attn_residual)."""
        # Step 1 — Project to shared d_model space
        Q  = self.cnn_proj(cnn_features)               # (B, d_model)
        KV = self.transformer_proj(transformer_features)  # (B, d_model)

        # Step 2 — Cross-attention
        Q3D  = Q.unsqueeze(1)   # (B, 1, d_model)
        KV3D = KV.unsqueeze(1)  # (B, 1, d_model)
        attn_out, _ = self.cross_attention(Q3D, KV3D, KV3D)  # (B, 1, d_model)
        attn_out = self.attn_dropout(attn_out.squeeze(1))     # (B, d_model)

        # Step 3 — Residual + LayerNorm
        attn_residual = self.residual_norm(Q + attn_out)      # (B, d_model)
        return Q, attn_residual

    def _compute_prototype_similarity(
        self,
        attn_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute class-aware prototype representation.

        Args:
            attn_residual: Cross-attention output after residual, (B, d_model).

        Returns:
            Tuple of:
            - ``similarities``   (B, num_classes) — raw cosine similarities ∈ (−1, 1)
            - ``weights``        (B, num_classes) — softmax-normalised, sums to 1
            - ``P_x``            (B, d_model)    — weighted prototype aggregate
        """
        prototypes = self.class_prototypes  # (num_classes, d_model)

        if self._normalize_prototypes:
            prototypes = F.normalize(prototypes, p=2, dim=1)  # unit-norm rows

        # Normalise the query representation
        A_norm = F.normalize(attn_residual, p=2, dim=1)       # (B, d_model)

        # Cosine similarity: (B, d_model) @ (d_model, num_classes) = (B, num_classes)
        # Clamped to [-1.0, 1.0] for numerical stability
        similarities = torch.clamp(A_norm @ prototypes.t(), min=-1.0, max=1.0)  # (B, num_classes)

        # Temperature-scaled softmax → class-aware weights
        # Lower temperature (e.g. τ=0.1) yields sharper prototype assignment;
        # higher values (e.g. τ=1.0) yield softer weights across classes.
        temp = max(float(self._prototype_temperature), 1e-6)
        weights = F.softmax(similarities / temp, dim=-1)  # (B, C)

        # Weighted prototype aggregate P_x: (B, C) × (C, d_model) = (B, d_model)
        P_x = weights @ prototypes                  # (B, d_model)

        return similarities, weights, P_x

    # ── Public API ────────────────────────────────────────────────────────────

    def forward(
        self,
        cnn_features: torch.Tensor,
        transformer_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute class logits via class-aware adaptive cross-attention fusion.

        Args:
            cnn_features:          Tensor ``(B, cnn_dim)`` from any CNN backbone.
            transformer_features:  Tensor ``(B, transformer_dim)`` from any
                                   Transformer backbone.

        Returns:
            Logit tensor ``(B, num_classes)``.
        """
        Q, attn_residual = self._project_and_attend(cnn_features, transformer_features)

        # Class-aware prototype representation
        _, _, P_x = self._compute_prototype_similarity(attn_residual)

        # Step 4 — Class-aware adaptive gate (768-d input)
        gate_input = torch.cat([Q, attn_residual, P_x], dim=1)  # (B, 3·d_model)
        alpha = torch.sigmoid(self.gate(gate_input))             # (B, d_model)
        fused = alpha * attn_residual + (1.0 - alpha) * Q        # (B, d_model)

        # Step 5 — Classification
        return self.head(fused)                                   # (B, num_classes)

    def forward_with_diagnostics(
        self,
        cnn_features: torch.Tensor,
        transformer_features: torch.Tensor,
    ) -> dict:
        """Forward pass returning logits plus intermediate diagnostic tensors.

        Use this during inference / debugging to inspect prototype similarities,
        gate statistics, and class-aware weights without a second forward pass.

        Args:
            cnn_features:          Tensor ``(B, cnn_dim)``.
            transformer_features:  Tensor ``(B, transformer_dim)``.

        Returns:
            Dictionary with keys:

            ``logits``                   — ``(B, num_classes)``
            ``prototype_similarities``   — ``(B, num_classes)`` cosine sims
            ``prototype_weights``        — ``(B, num_classes)`` sums to 1
            ``prototype_representation`` — ``(B, d_model)``  weighted P_x
            ``gate_alpha``               — ``(B, d_model)``  ∈ (0, 1)
        """
        Q, attn_residual = self._project_and_attend(cnn_features, transformer_features)

        similarities, weights, P_x = self._compute_prototype_similarity(attn_residual)

        gate_input = torch.cat([Q, attn_residual, P_x], dim=1)
        alpha = torch.sigmoid(self.gate(gate_input))
        fused = alpha * attn_residual + (1.0 - alpha) * Q
        logits = self.head(fused)

        return {
            "logits":                   logits,
            "prototype_similarities":   similarities,
            "prototype_weights":        weights,
            "prototype_representation": P_x,
            "gate_alpha":               alpha,
        }

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def count_parameters(self) -> dict[str, int]:
        """Return trainable parameter counts broken down by sub-component.

        Returns:
            Dictionary with keys:
            ``cnn_projection``, ``transformer_projection``,
            ``cross_attention``, ``residual_norm``,
            ``class_prototypes``, ``gating_mlp``,
            ``classification_head``, ``total``.
        """
        def _c(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        prototype_params = self.class_prototypes.numel() if self.class_prototypes.requires_grad else 0

        return {
            "cnn_projection":          _c(self.cnn_proj),
            "transformer_projection":  _c(self.transformer_proj),
            "cross_attention":         _c(self.cross_attention),
            "residual_norm":           _c(self.residual_norm),
            "class_prototypes":        prototype_params,
            "gating_mlp":              _c(nn.ModuleList([self.gate])),
            "classification_head":     _c(self.head),
            "total":                   sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"cnn_dim={self._cnn_dim}, "
            f"transformer_dim={self._transformer_dim}, "
            f"d_model={self._d_model}, "
            f"num_heads={self._num_heads}, "
            f"hidden_dim={self._hidden_dim}, "
            f"num_classes={self._num_classes}, "
            f"prototype_temperature={self._prototype_temperature})"
        )
