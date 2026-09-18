"""Unit tests for Quantitative Faithfulness Evaluation module."""

import os
import json
import pytest
import numpy as np
import torch
import torch.nn as nn

from src.config import load_config, AppConfig
from src.evaluation.faithfulness import (
    compute_grid_importance,
    rank_regions,
    create_perturbed_baseline,
    apply_region_mask,
    calculate_auc,
    evaluate_deletion_curve,
    evaluate_insertion_curve,
    FaithfulnessEvaluator,
)


# ──────────────────────────────────────────────────────────────────────────────
# Mock Model for Fast Deterministic Unit Testing
# ──────────────────────────────────────────────────────────────────────────────

class MockConvNet(nn.Module):
    """Simple CNN with a conv layer and linear classifier for unit tests."""

    def __init__(self, num_classes: int = 8, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        flat = torch.flatten(feat, 1)
        return self.classifier(flat)


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite
# ──────────────────────────────────────────────────────────────────────────────

class TestFaithfulnessCore:
    """Test individual numerical and spatial functions of faithfulness."""

    def test_grid_importance_creation_and_shape(self):
        """Test 2: Grid creation and output shape."""
        H, W = 224, 224
        grid_size = 14
        cam = torch.rand(1, 1, H, W)

        grid = compute_grid_importance(cam, grid_size=grid_size)
        assert isinstance(grid, np.ndarray)
        assert grid.shape == (14, 14)
        assert np.all(grid >= 0.0)
        assert np.all(grid <= 1.0)

    def test_correct_number_of_regions(self):
        """Test 4: Correct total number of ranked regions (grid_size^2)."""
        grid_size = 14
        grid = np.random.rand(grid_size, grid_size)
        coords = rank_regions(grid, order="descending")

        assert len(coords) == 14 * 14
        # Verify all coordinates are unique and within bounds
        coord_set = set(coords)
        assert len(coord_set) == 196
        for r, c in coords:
            assert 0 <= r < 14
            assert 0 <= c < 14

    def test_region_ranking_orders(self):
        """Test 3: Ranking descending vs ascending."""
        grid = np.array([
            [0.1, 0.9],
            [0.4, 0.7],
        ])
        desc_coords = rank_regions(grid, order="descending")
        asc_coords = rank_regions(grid, order="ascending")

        # Highest value is at (0, 1) with 0.9
        assert desc_coords[0] == (0, 1)
        # Lowest value is at (0, 0) with 0.1
        assert desc_coords[-1] == (0, 0)

        # In ascending order, lowest is first
        assert asc_coords[0] == (0, 0)
        assert asc_coords[-1] == (0, 1)

    def test_random_ranking_reproducibility_with_seed(self):
        """Test 8: Random ranking is reproducible with a fixed seed."""
        grid = np.random.rand(14, 14)
        order_1 = rank_regions(grid, order="random", seed=42)
        order_2 = rank_regions(grid, order="random", seed=42)
        order_diff_seed = rank_regions(grid, order="random", seed=999)

        assert order_1 == order_2
        assert order_1 != order_diff_seed

    def test_perturbation_preserves_dimensions(self):
        """Test 5: Perturbation preserves image shape and tensor type."""
        img = torch.randn(1, 3, 224, 224)
        for method in ("blur", "mean", "zero"):
            baseline = create_perturbed_baseline(img, perturbation=method, blur_kernel_size=21, blur_sigma=10.0)
            assert baseline.shape == img.shape
            assert baseline.dtype == img.dtype
            assert baseline.device == img.device

    def test_zero_percent_perturbation_leaves_image_unchanged(self):
        """Test 6: 0% perturbation returns exact original image."""
        orig = torch.randn(1, 3, 224, 224)
        base = create_perturbed_baseline(orig, perturbation="blur")
        # No regions active
        masked = apply_region_mask(orig, base, active_regions=[], grid_size=14, mode="deletion")
        assert torch.allclose(masked, orig, atol=1e-7)

    def test_hundred_percent_perturbation_masks_all_regions(self):
        """Test 7: 100% deletion masks all regions, matching the baseline."""
        orig = torch.randn(1, 3, 224, 224)
        base = create_perturbed_baseline(orig, perturbation="blur")
        all_regions = [(r, c) for r in range(14) for c in range(14)]

        # 100% deletion -> fully baseline
        del_100 = apply_region_mask(orig, base, active_regions=all_regions, grid_size=14, mode="deletion")
        assert torch.allclose(del_100, base, atol=1e-7)

        # 100% insertion -> fully original
        ins_100 = apply_region_mask(orig, base, active_regions=all_regions, grid_size=14, mode="insertion")
        assert torch.allclose(ins_100, orig, atol=1e-7)

    def test_auc_calculation_on_synthetic_curves(self):
        """Test 11: AUC trapezoidal integration correctness on known analytic curves."""
        fractions = [0.0, 0.25, 0.5, 0.75, 1.0]

        # Constant 1.0 -> AUC = 1.0
        conf_const = [1.0, 1.0, 1.0, 1.0, 1.0]
        assert pytest.approx(calculate_auc(fractions, conf_const), 1e-5) == 1.0

        # Linear decay from 1.0 to 0.0 -> Triangle area = 0.5 * 1.0 * 1.0 = 0.5
        conf_linear = [1.0, 0.75, 0.5, 0.25, 0.0]
        assert pytest.approx(calculate_auc(fractions, conf_linear), 1e-5) == 0.5

        # Zero -> AUC = 0.0
        conf_zero = [0.0, 0.0, 0.0, 0.0, 0.0]
        assert pytest.approx(calculate_auc(fractions, conf_zero), 1e-5) == 0.0


class TestFaithfulnessCurves:
    """Test deletion and insertion curves on a mock model."""

    @pytest.fixture
    def mock_model(self):
        model = MockConvNet(num_classes=8)
        model.eval()
        return model

    def test_deletion_and_insertion_curves_point_count(self, mock_model):
        """Test 9 & 10: Curve lengths match fractions length."""
        fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        img = torch.randn(1, 3, 224, 224)
        ranked = [(r, c) for r in range(14) for c in range(14)]

        del_res = evaluate_deletion_curve(
            model=mock_model,
            image_tensor=img,
            ranked_regions=ranked,
            target_class=0,
            fractions=fractions,
            grid_size=14,
        )

        ins_res = evaluate_insertion_curve(
            model=mock_model,
            image_tensor=img,
            ranked_regions=ranked,
            target_class=0,
            fractions=fractions,
            grid_size=14,
        )

        assert len(del_res["confidences"]) == len(fractions)
        assert len(ins_res["confidences"]) == len(fractions)
        assert 0.0 <= del_res["auc"] <= 1.0
        assert 0.0 <= ins_res["auc"] <= 1.0


class TestFaithfulnessEvaluatorIntegration:
    """Test full FaithfulnessEvaluator orchestrator."""

    @pytest.fixture
    def mock_pipeline(self):
        cfg = load_config()
        cfg.model.num_classes = 8
        cfg.faithfulness.grid_size = 4  # Fast 4x4 grid for test speed
        cfg.faithfulness.fractions = [0.0, 0.5, 1.0]
        cfg.faithfulness.random_seeds = [42]

        model = MockConvNet(num_classes=8)
        target_layer = model.features[-2]  # ReLU after second conv
        evaluator = FaithfulnessEvaluator(
            model=model,
            cfg=cfg,
            target_layer=target_layer,
            device="cpu",
        )
        return evaluator

    def test_gradcam_normalization_and_attribution(self, mock_pipeline):
        """Test 1: Grad-CAM heatmap normalization within [0, 1]."""
        img = torch.randn(1, 3, 224, 224)
        cam = mock_pipeline.explainer.attribute(img, target_category=0)

        assert cam.shape == (1, 1, 224, 224)
        assert cam.min().item() >= 0.0
        assert cam.max().item() <= 1.0 + 1e-6

    def test_single_image_evaluation_structure(self, mock_pipeline):
        """Test 12: Single-image result dictionary structure and validity."""
        img = torch.randn(1, 3, 224, 224)
        res = mock_pipeline.evaluate_sample(
            image_tensor=img,
            target_category=None,
            sample_id="test_img_01",
        )

        assert "predicted_class" in res
        assert "original_confidence" in res
        assert 0.0 <= res["original_confidence"] <= 1.0

        # Grad-CAM metrics
        assert "gradcam" in res
        assert "deletion_auc" in res["gradcam"]
        assert "insertion_auc" in res["gradcam"]
        assert len(res["gradcam"]["deletion_curve"]) == 3
        assert len(res["gradcam"]["insertion_curve"]) == 3

        # Random baseline metrics
        assert "random" in res
        assert "deletion_auc" in res["random"]
        assert "insertion_auc" in res["random"]

        # Control
        assert "least_important" in res
        assert "deletion_auc" in res["least_important"]

        # Verify JSON serializability
        json_str = json.dumps(res)
        assert len(json_str) > 0

    def test_plots_and_report_saving(self, mock_pipeline, tmp_path):
        """Test saving report JSON and PNG curve plots without errors."""
        img = torch.randn(1, 3, 224, 224)
        res = mock_pipeline.evaluate_sample(image_tensor=img, sample_id="test_plot")

        report_file = tmp_path / "test_report.json"
        mock_pipeline.save_report(res, str(report_file))
        assert report_file.exists()

        plots_dir = tmp_path / "plots"
        saved_plots = mock_pipeline.plot_curves(res, str(plots_dir), prefix="test")
        assert os.path.exists(saved_plots["deletion_plot"])
        assert os.path.exists(saved_plots["insertion_plot"])
        assert os.path.exists(saved_plots["comparison_plot"])
