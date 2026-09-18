"""Calibration evaluation for Trustworthy-Capsule-AI.

Model calibration measures how well predicted confidence corresponds to actual
accuracy.  A perfectly calibrated model that predicts 80% confidence is correct
exactly 80% of the time.

This module provides three calibration metrics:

Expected Calibration Error (ECE)
---------------------------------
ECE partitions samples into M equal-width confidence bins and measures the
weighted average gap between bin accuracy and bin confidence:

    ECE = Σ_{b=1}^{M} (|B_b| / N) · |acc(B_b) − conf(B_b)|

where:
    B_b     = set of samples whose max confidence ∈ ((b-1)/M, b/M]
    |B_b|   = number of samples in bin b
    N       = total number of samples
    acc(B_b) = fraction of B_b samples with correct prediction
    conf(B_b) = mean predicted confidence in B_b

Brier Score (multi-class)
--------------------------
    BS = (1/N) Σ_i Σ_c (p_{ic} − y_{ic})²

where y_{ic} = 1 iff sample i belongs to class c.  Range [0, 2] for multi-class
(lower is better).  The single-class Brier Score has range [0, 1].

Reliability Diagram
--------------------
A visual plot of bin confidence (x-axis) vs bin accuracy (y-axis).  A perfectly
calibrated model lies on the diagonal.  Bars show the gap (miscalibration).
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# ECE
# ──────────────────────────────────────────────────────────────────────────────

def compute_ece(
    y_true: np.ndarray,
    y_conf: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        y_true: Ground-truth integer labels ``(N,)``.
        y_conf: Predicted confidence (max softmax probability) ``(N,)``.
        y_pred: Predicted integer labels ``(N,)``.
        n_bins: Number of equal-width confidence bins.

    Returns:
        ECE as a float in ``[0, 1]``.
    """
    assert y_true.shape == y_conf.shape == y_pred.shape, (
        "y_true, y_conf, y_pred must all have the same shape"
    )
    assert n_bins >= 2, f"n_bins must be at least 2, got {n_bins}"

    n = len(y_true)
    if n == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        # Last bin is closed on the right
        if b < n_bins - 1:
            in_bin = (y_conf >= lo) & (y_conf < hi)
        else:
            in_bin = (y_conf >= lo) & (y_conf <= hi)

        n_b = int(in_bin.sum())
        if n_b == 0:
            continue

        acc_b  = float((y_pred[in_bin] == y_true[in_bin]).mean())
        conf_b = float(y_conf[in_bin].mean())
        ece   += (n_b / n) * abs(acc_b - conf_b)

    return float(ece)


# ──────────────────────────────────────────────────────────────────────────────
# Brier Score
# ──────────────────────────────────────────────────────────────────────────────

def compute_brier_score(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int,
) -> float:
    """Compute multi-class Brier Score.

    Args:
        y_true:      Ground-truth integer labels ``(N,)``.
        y_proba:     Predicted class probabilities ``(N, num_classes)``.
        num_classes: Total number of classes.

    Returns:
        Brier Score as a float in ``[0, 2]`` (lower is better).
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    # One-hot encode true labels
    y_one_hot = np.zeros((n, num_classes), dtype=np.float32)
    y_one_hot[np.arange(n), y_true.astype(int)] = 1.0

    brier = float(np.mean(np.sum((y_proba - y_one_hot) ** 2, axis=1)))
    return brier


# ──────────────────────────────────────────────────────────────────────────────
# Reliability diagram data
# ──────────────────────────────────────────────────────────────────────────────

def compute_reliability_data(
    y_true: np.ndarray,
    y_conf: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute per-bin accuracy and confidence for a reliability diagram.

    Args:
        y_true: Ground-truth integer labels ``(N,)``.
        y_conf: Predicted confidence ``(N,)``.
        y_pred: Predicted integer labels ``(N,)``.
        n_bins: Number of equal-width confidence bins.

    Returns:
        Dictionary with keys:
        ``bin_centers``  — bin midpoints on x-axis ``(n_bins,)``
        ``bin_accs``     — bin accuracy ``(n_bins,)``
        ``bin_confs``    — bin mean confidence ``(n_bins,)``
        ``bin_counts``   — sample count per bin ``(n_bins,)``
        ``ece``          — computed ECE
    """
    n = len(y_true)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])

    bin_accs   = np.zeros(n_bins)
    bin_confs  = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        if b < n_bins - 1:
            in_bin = (y_conf >= lo) & (y_conf < hi)
        else:
            in_bin = (y_conf >= lo) & (y_conf <= hi)

        n_b = int(in_bin.sum())
        bin_counts[b] = n_b
        if n_b == 0:
            continue
        bin_accs[b]  = float((y_pred[in_bin] == y_true[in_bin]).mean())
        bin_confs[b] = float(y_conf[in_bin].mean())

    ece = compute_ece(y_true, y_conf, y_pred, n_bins)

    return {
        "bin_centers":  bin_centers,
        "bin_accs":     bin_accs,
        "bin_confs":    bin_confs,
        "bin_counts":   bin_counts,
        "ece":          ece,
        "n_samples":    n,
        "n_bins":       n_bins,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Reliability diagram plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_reliability_diagram(
    reliability_data: dict,
    save_path: Optional[str] = None,
    title: str = "Reliability Diagram",
) -> None:
    """Plot and optionally save a reliability diagram.

    Requires ``matplotlib``.  The diagram shows:
    - Blue bars: per-bin accuracy (what the model achieves)
    - Orange bars: per-bin confidence gap (miscalibration)
    - Dashed diagonal: perfect calibration reference

    Args:
        reliability_data: Output of ``compute_reliability_data()``.
        save_path:        If provided, saves the figure to this path.
        title:            Plot title string.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError(
            "matplotlib is required for reliability diagrams. "
            "Install with: pip install matplotlib"
        )

    centers = reliability_data["bin_centers"]
    accs    = reliability_data["bin_accs"]
    confs   = reliability_data["bin_confs"]
    counts  = reliability_data["bin_counts"]
    ece     = reliability_data["ece"]
    n_bins  = reliability_data["n_bins"]

    width = 1.0 / n_bins

    fig, ax = plt.subplots(figsize=(7, 6))

    # Accuracy bars
    ax.bar(centers, accs, width=width * 0.9, alpha=0.85,
           color="steelblue", label="Accuracy", align="center")

    # Confidence gap bars (overconfidence)
    gap = confs - accs
    positive_gap = np.maximum(gap, 0)
    ax.bar(centers, positive_gap, width=width * 0.9, bottom=accs,
           alpha=0.5, color="darkorange", label="Confidence gap", align="center")

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
            linewidth=1.5, label="Perfect calibration")

    ax.set_xlabel("Predicted confidence", fontsize=12)
    ax.set_ylabel("Actual accuracy",      fontsize=12)
    ax.set_title(f"{title}\n(ECE = {ece:.4f}, N = {reliability_data['n_samples']})",
                 fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
