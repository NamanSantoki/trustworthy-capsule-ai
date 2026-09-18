"""Monte Carlo Dropout uncertainty estimation for Trustworthy-Capsule-AI.

Theory
------
At test time, standard neural networks are fully deterministic when dropout
is disabled (``model.eval()``).  Monte Carlo (MC) Dropout re-enables dropout
at inference to obtain a Bayesian approximation to the posterior predictive
distribution [Gal & Ghahramani, 2016].

For ``N`` stochastic forward passes, let ``p_i ∈ ℝ^C`` be the softmax
output of pass ``i``.  Then:

    p̄_c   = (1/N) Σ_{i=1}^{N} p_{i,c}      mean class probability
    ŷ     = argmax_c p̄_c                    predicted class
    conf  = p̄_{ŷ}                           mean confidence

Predictive entropy (total uncertainty):

    H(p̄) = -Σ_c  p̄_c · log(p̄_c + ε)

Variation ratio (classification uncertainty):

    VR    = 1 - (mode_count / N)

where ``mode_count`` is the number of passes that agreed with the majority
class.

Implementation note — BatchNorm safety
---------------------------------------
Simply calling ``model.train()`` would set **all** sub-modules to training
mode, including BatchNorm layers.  During training BatchNorm uses the
per-batch running statistics, which changes predictions non-trivially and
introduces spurious uncertainty.

We instead use a context manager that:
  1. Calls ``model.train()`` to activate Dropout.
  2. Iterates over all sub-modules and sets any ``BatchNorm``
     (``BatchNorm1d``, ``BatchNorm2d``, ``GroupNorm``, ``LayerNorm``)
     back to ``eval()`` so they continue using running statistics.

This is the standard, safe way to implement MC Dropout in PyTorch.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class UncertaintyResult:
    """Result container for MC Dropout uncertainty estimation.

    Attributes:
        predicted_class:    Integer class index of the predicted label.
        mean_proba:         Mean class probability distribution ``(num_classes,)``.
        confidence:         Mean probability assigned to the predicted class.
        predictive_entropy: H(p̄) = -Σ p̄_c log(p̄_c) — total uncertainty.
        variation_ratio:    1 - mode_count / n_samples — classification spread.
        all_probas:         All N stochastic softmax outputs ``(N, num_classes)``.
        n_samples:          Number of MC forward passes used.
    """
    predicted_class:    int
    mean_proba:         np.ndarray
    confidence:         float
    predictive_entropy: float
    variation_ratio:    float
    all_probas:         np.ndarray
    n_samples:          int

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation (no numpy arrays)."""
        return {
            "predicted_class":    self.predicted_class,
            "confidence":         self.confidence,
            "predictive_entropy": self.predictive_entropy,
            "variation_ratio":    self.variation_ratio,
            "n_samples":          self.n_samples,
            "mean_proba":         self.mean_proba.tolist(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Context manager — safe MC Dropout mode
# ──────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def mc_dropout_mode(model: nn.Module):
    """Context manager that puts model in eval() mode and selectively enables only Dropout layers.

    This ensures BatchNorm, LayerNorm, and other normalisation running statistics
    remain frozen at test time, avoiding spurious noise while enabling Monte Carlo
    stochasticity through Dropout layers.

    Usage::

        with mc_dropout_mode(model):
            logits = model(x)   # Dropout is active; BN/LN remain in eval mode

    On exit the model is restored to whatever state it was in before entry.
    """
    # Remember original training state for each module
    original_states: dict[nn.Module, bool] = {
        m: m.training for m in model.modules()
    }

    # First set entire model to eval mode (safely freezes BatchNorm / LayerNorm etc.)
    model.eval()

    # Selectively enable dropout modules
    _DROPOUT_TYPES = (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )
    for m in model.modules():
        if isinstance(m, _DROPOUT_TYPES) or "Dropout" in m.__class__.__name__:
            m.train(True)

    try:
        yield model
    finally:
        # Restore original training states exactly
        for m, was_training in original_states.items():
            m.training = was_training


# ──────────────────────────────────────────────────────────────────────────────
# Predictor
# ──────────────────────────────────────────────────────────────────────────────

class MCDropoutPredictor:
    """Uncertainty-aware predictor using Monte Carlo Dropout.

    Args:
        model:  Any ``nn.Module`` that accepts ``(B, 3, H, W)`` tensors and
                returns logit tensors ``(B, num_classes)``.  The model must
                contain at least one ``nn.Dropout`` layer; without dropout
                MC inference degenerates to repeated identical passes.

    Example::

        predictor = MCDropoutPredictor(model)
        result = predictor.predict(tensor_img, n_samples=20, device=device)
        print(f"Entropy: {result.predictive_entropy:.4f}")
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def predict(
        self,
        tensor: torch.Tensor,
        n_samples: int = 20,
        device: torch.device | None = None,
    ) -> UncertaintyResult:
        """Run MC Dropout inference and return uncertainty estimates.

        If ``n_samples == 0``, a single deterministic forward pass is run
        (equivalent to standard ``model.eval()`` inference).  The returned
        ``UncertaintyResult`` will have ``n_samples=1``, ``predictive_entropy``
        computed from the single softmax distribution, and
        ``variation_ratio=0.0``.

        Args:
            tensor:    Input image tensor ``(1, 3, H, W)`` or ``(B, 3, H, W)``.
            n_samples: Number of stochastic forward passes.  ``0`` → deterministic.
            device:    Device to run inference on.  Defaults to model's device.

        Returns:
            :class:`UncertaintyResult` with all fields populated.
        """
        if device is not None:
            tensor = tensor.to(device)

        if n_samples == 0:
            # Deterministic mode
            self.model.eval()
            with torch.no_grad():
                logits = self.model(tensor)
                proba = F.softmax(logits, dim=1).cpu().numpy()   # (B, C)
            # Use first sample for scalar stats
            p = proba[0] if proba.ndim == 2 else proba
            return self._build_result(np.array([p]), n_actual=1, deterministic=True)

        # Stochastic MC Dropout passes
        all_probas: list[np.ndarray] = []
        with mc_dropout_mode(self.model):
            with torch.no_grad():
                for _ in range(n_samples):
                    logits = self.model(tensor)
                    proba = F.softmax(logits, dim=1).cpu().numpy()  # (B, C)
                    # Take first item if single-image input
                    all_probas.append(proba[0] if proba.ndim == 2 else proba)

        stacked = np.stack(all_probas, axis=0)   # (N, C)
        return self._build_result(stacked, n_actual=n_samples, deterministic=False)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_result(
        stacked: np.ndarray,
        n_actual: int,
        deterministic: bool,
    ) -> UncertaintyResult:
        """Build ``UncertaintyResult`` from stacked probability matrix (N, C)."""
        mean_proba = stacked.mean(axis=0)                         # (C,)
        predicted_class = int(np.argmax(mean_proba))
        confidence = float(mean_proba[predicted_class])

        # Predictive entropy: H(p̄) = -Σ p̄_c log(p̄_c + ε)
        eps = 1e-8
        predictive_entropy = float(
            -np.sum(mean_proba * np.log(mean_proba + eps))
        )

        if deterministic or n_actual == 1:
            variation_ratio = 0.0
        else:
            # Mode class across all N passes
            per_pass_classes = np.argmax(stacked, axis=1)         # (N,)
            mode_class = np.bincount(per_pass_classes).argmax()
            mode_count = int(np.sum(per_pass_classes == mode_class))
            variation_ratio = float(1.0 - mode_count / n_actual)

        return UncertaintyResult(
            predicted_class=predicted_class,
            mean_proba=mean_proba,
            confidence=confidence,
            predictive_entropy=predictive_entropy,
            variation_ratio=variation_ratio,
            all_probas=stacked,
            n_samples=n_actual,
        )
