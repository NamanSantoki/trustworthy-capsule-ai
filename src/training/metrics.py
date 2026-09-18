"""Evaluation metrics for Trustworthy-Capsule-AI.

All computation is stateless and PyTorch-free — methods accept plain NumPy
arrays so the module can be used outside a training loop (e.g., post-hoc
analysis scripts) without importing torch.

Classes
-------
MetricsResult
    Typed dataclass holding every computed metric.  Serialisable to dict/JSON
    via ``to_dict()`` for checkpointing and history tracking.

MetricsCalculator
    Stateless utility class.  The single public method ``compute_all()``
    returns a fully-populated ``MetricsResult`` in one call.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix as sk_confusion_matrix,
)


# ──────────────────────────────────────────────────────────────────────────────
# Result Container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricsResult:
    """Container for all evaluation metrics produced in one pass.

    Attributes:
        accuracy:            Top-1 accuracy (0–1).
        balanced_accuracy:   Macro-averaged recall; robust to class imbalance.
        precision_macro:     Macro-averaged precision.
        recall_macro:        Macro-averaged recall (= balanced accuracy).
        f1_macro:            Macro-averaged F1.
        f1_per_class:        Per-class F1 keyed by class name.
        roc_auc_macro:       One-vs-Rest macro-averaged ROC-AUC.
        roc_auc_per_class:   Per-class ROC-AUC keyed by class name.
        confusion_matrix:    Integer confusion matrix ``(num_classes, num_classes)``.
        num_samples:         Total number of samples evaluated.
    """

    accuracy:           float
    balanced_accuracy:  float
    precision_macro:    float
    recall_macro:       float
    f1_macro:           float
    f1_per_class:       dict[str, float]
    roc_auc_macro:      float
    roc_auc_per_class:  dict[str, float]
    confusion_matrix:   np.ndarray
    num_samples:        int

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary.

        ``confusion_matrix`` is converted to a nested list so it can be
        written to JSON without a custom encoder.
        """
        return {
            "accuracy":           self.accuracy,
            "balanced_accuracy":  self.balanced_accuracy,
            "precision_macro":    self.precision_macro,
            "recall_macro":       self.recall_macro,
            "f1_macro":           self.f1_macro,
            "f1_per_class":       self.f1_per_class,
            "roc_auc_macro":      self.roc_auc_macro,
            "roc_auc_per_class":  self.roc_auc_per_class,
            "confusion_matrix":   self.confusion_matrix.tolist(),
            "num_samples":        self.num_samples,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MetricsResult":
        """Reconstruct from a dictionary (e.g., loaded from JSON)."""
        d = dict(d)
        d["confusion_matrix"] = np.array(d["confusion_matrix"])
        return cls(**d)


# ──────────────────────────────────────────────────────────────────────────────
# Stateless Calculator
# ──────────────────────────────────────────────────────────────────────────────

class MetricsCalculator:
    """Stateless evaluation metrics calculator.

    All methods are ``@staticmethod`` — no instantiation required.

    Example::

        result = MetricsCalculator.compute_all(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            num_classes=4,
            class_names=["normal", "polyp", "ulcer", "cancer"],
        )
        print(f"F1: {result.f1_macro:.4f}")
    """

    @staticmethod
    def compute_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        num_classes: int,
        class_names: list[str],
    ) -> MetricsResult:
        """Compute the full suite of evaluation metrics in a single call.

        Args:
            y_true:       Ground-truth integer labels ``(N,)``.
            y_pred:       Predicted integer labels ``(N,)``.
            y_proba:      Predicted class probabilities ``(N, num_classes)``.
                          Rows must sum to 1.0 (i.e., post-softmax).
            num_classes:  Total number of classes.
            class_names:  List of class name strings, length ``num_classes``.

        Returns:
            :class:`MetricsResult` with all fields populated.
        """
        assert len(class_names) == num_classes, (
            f"class_names length {len(class_names)} != num_classes {num_classes}"
        )

        labels = list(range(num_classes))

        # ── Scalar metrics ────────────────────────────────────────────────────
        accuracy          = float(accuracy_score(y_true, y_pred))
        balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
        precision_macro   = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        recall_macro      = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        f1_macro          = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

        # ── Per-class F1 ─────────────────────────────────────────────────────
        f1_per_label: np.ndarray = f1_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        f1_per_class = {
            class_names[i]: float(f1_per_label[i])
            for i in range(num_classes)
        }

        # ── ROC-AUC (One-vs-Rest, multi-class) ───────────────────────────────
        # Gracefully handled when a class has no positive samples in the split.
        try:
            roc_auc_macro = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
            roc_auc_per_label: np.ndarray = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average=None
            )
            roc_auc_per_class = {
                class_names[i]: float(roc_auc_per_label[i])
                for i in range(num_classes)
            }
        except ValueError:
            # Raised when only one class is present in y_true (e.g. tiny val split)
            roc_auc_macro = float("nan")
            roc_auc_per_class = {name: float("nan") for name in class_names}

        # ── Confusion matrix ──────────────────────────────────────────────────
        cm = sk_confusion_matrix(y_true, y_pred, labels=labels)

        return MetricsResult(
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            f1_macro=f1_macro,
            f1_per_class=f1_per_class,
            roc_auc_macro=roc_auc_macro,
            roc_auc_per_class=roc_auc_per_class,
            confusion_matrix=cm,
            num_samples=int(len(y_true)),
        )
