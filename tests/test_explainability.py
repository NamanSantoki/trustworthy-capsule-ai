"""Unit tests for Phase 6 Explainability & Trustworthiness module."""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from src.config import load_config, AppConfig
from src.models.factory import build_model
from src.models.efficientnet import EfficientNetV2SClassifier
from src.models.hybrid import HybridCapsuleModel
from src.explainability.base import BaseExplainer
from src.explainability.gradcam import GradCAM
from src.explainability.utils import (
    get_target_layer,
    load_model_checkpoint,
    preprocess_image,
    overlay_heatmap,
    save_explainability_output,
)
from src.explainability.pipeline import ExplainabilityPipeline


@pytest.fixture
def sample_image_path(tmp_path):
    """Creates a temporary RGB image for testing."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_file = img_dir / "sample.jpg"
    img_arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_arr)
    img.save(img_file)
    return str(img_file)


@pytest.fixture
def mock_checkpoint_path(tmp_path):
    """Creates a dummy checkpoint file."""
    cfg = load_config()
    model = build_model(cfg)
    ckpt_path = tmp_path / "mock_model.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    return str(ckpt_path)


class TestTargetLayerSelection:
    """Test target layer resolution across model architectures."""

    def test_auto_target_layer_efficientnet_only(self):
        cfg = load_config()
        cfg.model.variant = "efficientnet_only"
        model = build_model(cfg)
        layer = get_target_layer(model)
        assert isinstance(layer, nn.Module)

    def test_auto_target_layer_hybrid_concat(self):
        cfg = load_config()
        cfg.model.variant = "hybrid_concat"
        model = build_model(cfg)
        layer = get_target_layer(model)
        assert isinstance(layer, nn.Module)

    def test_auto_target_layer_hybrid_adaptive(self):
        cfg = load_config()
        cfg.model.variant = "hybrid_adaptive"
        model = build_model(cfg)
        layer = get_target_layer(model)
        assert isinstance(layer, nn.Module)

    def test_explicit_target_layer_string(self):
        cfg = load_config()
        cfg.model.variant = "efficientnet_only"
        model = build_model(cfg)
        layer = get_target_layer(model, "backbone.features.0")
        assert layer == model.backbone.features[0]

    def test_invalid_target_layer_string_raises(self):
        cfg = load_config()
        model = build_model(cfg)
        with pytest.raises(ValueError):
            get_target_layer(model, "non_existent_layer_name")


class TestGradCAMForwardBackwardPass:
    """Test Grad-CAM activation extraction, gradients, shape, and value bounds."""

    @pytest.mark.parametrize("variant", ["efficientnet_only", "hybrid_concat", "hybrid_adaptive"])
    def test_gradcam_forward_pass_and_shape(self, variant):
        cfg = load_config()
        cfg.model.variant = variant
        model = build_model(cfg)
        target_layer = get_target_layer(model)
        gradcam = GradCAM(model=model, target_layer=target_layer)

        input_tensor = torch.randn(2, 3, 224, 224)
        cam = gradcam.attribute(input_tensor, target_category=0)

        assert isinstance(cam, torch.Tensor)
        assert cam.shape == (2, 1, 224, 224)
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0

        gradcam.remove_hooks()

    def test_gradcam_auto_target_category(self):
        cfg = load_config()
        cfg.model.variant = "efficientnet_only"
        model = build_model(cfg)
        target_layer = get_target_layer(model)
        gradcam = GradCAM(model=model, target_layer=target_layer)

        input_tensor = torch.randn(1, 3, 224, 224)
        cam = gradcam.attribute(input_tensor, target_category=None)
        assert cam.shape == (1, 1, 224, 224)
        gradcam.remove_hooks()

    def test_gradcam_eval_mode_determinism(self):
        cfg = load_config()
        cfg.model.variant = "efficientnet_only"
        model = build_model(cfg)
        model.eval()
        target_layer = get_target_layer(model)
        gradcam = GradCAM(model=model, target_layer=target_layer)

        input_tensor = torch.randn(1, 3, 224, 224)
        cam1 = gradcam.attribute(input_tensor, target_category=1)
        cam2 = gradcam.attribute(input_tensor, target_category=1)

        assert torch.allclose(cam1, cam2, atol=1e-5)
        gradcam.remove_hooks()

    def test_invalid_target_category_type(self):
        cfg = load_config()
        model = build_model(cfg)
        target_layer = get_target_layer(model)
        gradcam = GradCAM(model=model, target_layer=target_layer)
        input_tensor = torch.randn(1, 3, 224, 224)

        with pytest.raises(TypeError):
            gradcam.attribute(input_tensor, target_category="invalid_class")
        gradcam.remove_hooks()


class TestExplainabilityUtilities:
    """Test preprocessing, checkpoint loading, overlay generation, and file saving."""

    def test_preprocess_image(self, sample_image_path):
        tensor, orig_rgb = preprocess_image(sample_image_path, img_size=224)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)
        assert isinstance(orig_rgb, np.ndarray)
        assert orig_rgb.shape == (224, 224, 3)

    def test_overlay_heatmap(self):
        heatmap = torch.rand(224, 224)
        orig_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        overlay, heatmap_rgb = overlay_heatmap(heatmap, orig_rgb, alpha=0.5)

        assert overlay.shape == (224, 224, 3)
        assert heatmap_rgb.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8

    def test_save_explainability_output(self, tmp_path):
        orig_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        overlay_rgb = np.ones((224, 224, 3), dtype=np.uint8) * 128
        heatmap_rgb = np.ones((224, 224, 3), dtype=np.uint8) * 200

        save_path = str(tmp_path / "test_out")
        save_explainability_output(orig_rgb, overlay_rgb, heatmap_rgb, save_path=save_path)

        assert os.path.exists(f"{save_path}_overlay.png")
        assert os.path.exists(f"{save_path}_side_by_side.png")

    def test_checkpoint_loading(self, mock_checkpoint_path):
        cfg = load_config()
        model = build_model(cfg)
        loaded_model = load_model_checkpoint(model, mock_checkpoint_path, device="cpu")
        assert not loaded_model.training


class TestExplainabilityPipeline:
    """Test high-level pipeline orchestration."""

    def test_pipeline_explain_image(self, sample_image_path, mock_checkpoint_path, tmp_path):
        cfg = load_config()
        out_dir = str(tmp_path / "pipeline_outputs")

        pipeline = ExplainabilityPipeline(cfg=cfg, checkpoint_path=mock_checkpoint_path, device="cpu")
        res = pipeline.explain_image(image_path=sample_image_path, output_dir=out_dir)

        assert "predicted_class" in res
        assert "heatmap" in res
        assert isinstance(res["predicted_class"], int)
        assert os.path.exists(f"{res['save_path_base']}_overlay.png")

    def test_pipeline_explain_folder(self, sample_image_path, mock_checkpoint_path, tmp_path):
        cfg = load_config()
        folder_dir = str(os.path.dirname(sample_image_path))
        out_dir = str(tmp_path / "folder_outputs")

        pipeline = ExplainabilityPipeline(cfg=cfg, checkpoint_path=mock_checkpoint_path, device="cpu")
        results = pipeline.explain_folder(folder_path=folder_dir, output_dir=out_dir)

        assert len(results) >= 1
        assert "predicted_class" in results[0]
