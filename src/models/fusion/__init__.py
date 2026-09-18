"""Fusion modules package for Trustworthy-Capsule-AI.

Contents
--------
ConcatFusion                   (concat.py)   — Baseline: simple feature concatenation.
AdaptiveCrossAttentionFusion   (adaptive.py) — Phase 5 proposed: cross-attention gating.

Both modules are **backbone-agnostic** — they accept flat feature tensors
``(B, cnn_dim)`` and ``(B, transformer_dim)`` from any CNN and Transformer
backbone pair, making them interchangeable via the factory without touching
``HybridCapsuleModel`` or the Trainer.

Usage
-----
    from src.models.fusion import ConcatFusion
    from src.models.fusion import AdaptiveCrossAttentionFusion
"""

from src.models.fusion.concat import ConcatFusion
from src.models.fusion.adaptive import AdaptiveCrossAttentionFusion

__all__ = ["ConcatFusion", "AdaptiveCrossAttentionFusion"]
