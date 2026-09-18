"""Tests for calibration evaluation module."""

import pytest
import numpy as np

from src.evaluation.calibration import (
    compute_ece,
    compute_brier_score,
    compute_reliability_data,
)


class TestCalibration:
    def test_perfect_calibration_ece(self):
        # 10 samples with 1.0 confidence and 100% accuracy
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_conf = np.ones(10)

        ece = compute_ece(y_true, y_conf, y_pred, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-5)

    def test_complete_miscalibration_ece(self):
        # 10 samples with 1.0 confidence and 0% accuracy
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_conf = np.ones(10)

        ece = compute_ece(y_true, y_conf, y_pred, n_bins=10)
        assert ece == pytest.approx(1.0, abs=1e-5)

    def test_brier_score_bounds(self):
        # Perfect prediction -> Brier score = 0
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        bs_perfect = compute_brier_score(y_true, y_proba, num_classes=3)
        assert bs_perfect == pytest.approx(0.0, abs=1e-5)

        # Worst prediction -> Brier score = 2.0
        y_proba_worst = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        bs_worst = compute_brier_score(y_true, y_proba_worst, num_classes=3)
        assert bs_worst == pytest.approx(2.0, abs=1e-5)

    def test_reliability_data_structure(self):
        y_true = np.random.randint(0, 4, size=50)
        y_pred = np.random.randint(0, 4, size=50)
        y_conf = np.random.uniform(0.25, 1.0, size=50)

        rel = compute_reliability_data(y_true, y_conf, y_pred, n_bins=5)
        assert len(rel["bin_centers"]) == 5
        assert len(rel["bin_accs"]) == 5
        assert len(rel["bin_confs"]) == 5
        assert len(rel["bin_counts"]) == 5
        assert sum(rel["bin_counts"]) == 50
        assert 0.0 <= rel["ece"] <= 1.0
