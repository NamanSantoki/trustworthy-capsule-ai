"""Unit tests for Phase 3 (refactored): Model Architecture.

Covers:
  - EfficientNetV2SBackbone: output shape, freeze/unfreeze.
  - EfficientNetV2SClassifier: forward pass, trainer interface.
  - ViTSmallBackbone:           output shape, freeze/unfreeze.
  - ViTSmallClassifier:         forward pass, trainer interface.
  - HybridCapsuleModel:         composable constructor, forward, freeze API.
  - build_model factory:        all supported variants route correctly.
"""

import pytest
import torch
import torch.nn as nn

from src.config import load_config, AppConfig
from src.models.efficientnet import EfficientNetV2SBackbone, EfficientNetV2SClassifier
from src.models.vit import ViTSmallBackbone, ViTSmallClassifier
from src.models.fusion.concat import ConcatFusion
from src.models.hybrid import HybridCapsuleModel
from src.models.factory import build_model

DEVICE     = torch.device("cpu")
BATCH_SIZE = 2
IMG_SIZE   = 224


@pytest.fixture(scope="module")
def cfg() -> AppConfig:
    return load_config("config")


@pytest.fixture(scope="module")
def dummy_input() -> torch.Tensor:
    """Synthetic image batch (B, 3, H, W)."""
    return torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)


# ─── EfficientNet Backbone ────────────────────────────────────────────────────

class TestEfficientNetBackbone:

    def test_output_shape(self, dummy_input):
        model = EfficientNetV2SBackbone(pretrained=False)
        model.eval()
        with torch.no_grad():
            feats = model(dummy_input)
        assert feats.shape == (BATCH_SIZE, EfficientNetV2SBackbone.OUT_FEATURES)

    def test_freeze_disables_gradients(self):
        model = EfficientNetV2SBackbone(pretrained=False)
        model.freeze()
        assert sum(1 for p in model.parameters() if p.requires_grad) == 0

    def test_unfreeze_restores_gradients(self):
        model = EfficientNetV2SBackbone(pretrained=False, freeze=True)
        model.unfreeze()
        total     = sum(1 for _ in model.parameters())
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert trainable == total


class TestEfficientNetClassifier:

    def test_output_shape(self, cfg, dummy_input):
        model = EfficientNetV2SClassifier(
            pretrained=False, num_classes=cfg.model.num_classes
        )
        model.eval()
        with torch.no_grad():
            logits = model(dummy_input)
        assert logits.shape == (BATCH_SIZE, cfg.model.num_classes)

    def test_trainer_interface_present(self, cfg):
        model = EfficientNetV2SClassifier(pretrained=False)
        assert hasattr(model, "freeze_backbones")
        assert hasattr(model, "unfreeze_backbones")
        assert hasattr(model, "freeze_backbone_epochs")

    def test_freeze_backbone_epoch_config(self):
        model = EfficientNetV2SClassifier(pretrained=False, freeze_backbone_epochs=7)
        assert model.freeze_backbone_epochs == 7

    def test_count_parameters_keys(self):
        model = EfficientNetV2SClassifier(pretrained=False)
        keys = set(model.count_parameters().keys())
        assert {"backbone_trainable", "classifier_trainable", "total_trainable"} == keys


# ─── ViT Backbone ─────────────────────────────────────────────────────────────

class TestViTBackbone:

    def test_output_shape(self, dummy_input):
        model = ViTSmallBackbone(pretrained=False)
        model.eval()
        with torch.no_grad():
            feats = model(dummy_input)
        assert feats.shape == (BATCH_SIZE, ViTSmallBackbone.OUT_FEATURES)

    def test_freeze_disables_gradients(self):
        model = ViTSmallBackbone(pretrained=False)
        model.freeze()
        assert sum(1 for p in model.parameters() if p.requires_grad) == 0

    def test_unfreeze_restores_gradients(self):
        model = ViTSmallBackbone(pretrained=False, freeze=True)
        model.unfreeze()
        total     = sum(1 for _ in model.parameters())
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert trainable == total


class TestViTClassifier:

    def test_output_shape(self, cfg, dummy_input):
        model = ViTSmallClassifier(
            pretrained=False, num_classes=cfg.model.num_classes
        )
        model.eval()
        with torch.no_grad():
            logits = model(dummy_input)
        assert logits.shape == (BATCH_SIZE, cfg.model.num_classes)

    def test_trainer_interface_present(self):
        model = ViTSmallClassifier(pretrained=False)
        assert hasattr(model, "freeze_backbones")
        assert hasattr(model, "unfreeze_backbones")
        assert hasattr(model, "freeze_backbone_epochs")

    def test_count_parameters_keys(self):
        model = ViTSmallClassifier(pretrained=False)
        keys = set(model.count_parameters().keys())
        assert {"backbone_trainable", "classifier_trainable", "total_trainable"} == keys


# ─── ConcatFusion ─────────────────────────────────────────────────────────────

class TestConcatFusion:

    def test_output_shape(self):
        fusion = ConcatFusion(cnn_dim=1280, transformer_dim=384, hidden_dim=512, num_classes=8)
        cnn_feats   = torch.randn(BATCH_SIZE, 1280)
        trans_feats = torch.randn(BATCH_SIZE, 384)
        logits = fusion(cnn_feats, trans_feats)
        assert logits.shape == (BATCH_SIZE, 8)


# ─── HybridCapsuleModel ───────────────────────────────────────────────────────

class TestHybridCapsuleModel:

    def _build_hybrid(self, cfg):
        cnn    = EfficientNetV2SBackbone(pretrained=False)
        vit    = ViTSmallBackbone(pretrained=False)
        fusion = ConcatFusion(
            cnn_dim=EfficientNetV2SBackbone.OUT_FEATURES,
            transformer_dim=ViTSmallBackbone.OUT_FEATURES,
            hidden_dim=cfg.model.fusion_hidden_dim,
            num_classes=cfg.model.num_classes,
            dropout_rate=cfg.model.dropout_rate,
        )
        return HybridCapsuleModel(
            cnn_backbone=cnn,
            transformer_backbone=vit,
            fusion_module=fusion,
            freeze_backbone_epochs=cfg.model.freeze_backbone_epochs,
        )

    def test_output_shape(self, cfg, dummy_input):
        model = self._build_hybrid(cfg)
        model.eval()
        with torch.no_grad():
            logits = model(dummy_input)
        assert logits.shape == (BATCH_SIZE, cfg.model.num_classes)

    def test_backbones_frozen_at_init(self, cfg):
        model = self._build_hybrid(cfg)
        cnn_trainable = sum(1 for p in model.cnn_backbone.parameters() if p.requires_grad)
        vit_trainable = sum(1 for p in model.transformer_backbone.parameters() if p.requires_grad)
        assert cnn_trainable == 0, f"CNN backbone should be frozen, got {cnn_trainable}"
        assert vit_trainable == 0, f"ViT backbone should be frozen, got {vit_trainable}"

    def test_unfreeze_exposes_all_params(self, cfg):
        model = self._build_hybrid(cfg)
        model.unfreeze_backbones()
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        total     = sum(1 for _ in model.parameters())
        assert trainable == total

    def test_count_parameters_keys(self, cfg):
        model = self._build_hybrid(cfg)
        keys = set(model.count_parameters().keys())
        expected = {
            "cnn_backbone_trainable",
            "transformer_backbone_trainable",
            "fusion_module_trainable",
            "total_trainable",
        }
        assert keys == expected

    def test_freeze_backbone_epochs_property(self, cfg):
        model = self._build_hybrid(cfg)
        assert model.freeze_backbone_epochs == cfg.model.freeze_backbone_epochs


# ─── Factory ──────────────────────────────────────────────────────────────────

class TestBuildModelFactory:

    def _cfg_with_variant(self, cfg: AppConfig, variant: str) -> AppConfig:
        """Return a shallow copy of cfg with variant and pretrained overridden."""
        data = cfg.model_dump()
        data["model"]["variant"]   = variant
        data["model"]["pretrained"] = False
        return type(cfg).model_validate(data)

    def test_efficientnet_only(self, cfg, dummy_input):
        model = build_model(self._cfg_with_variant(cfg, "efficientnet_only"))
        model.eval()
        with torch.no_grad():
            logits = model(dummy_input)
        assert logits.shape == (BATCH_SIZE, cfg.model.num_classes)
        assert isinstance(model, EfficientNetV2SClassifier)

    def test_vit_only(self, cfg, dummy_input):
        model = build_model(self._cfg_with_variant(cfg, "vit_only"))
        model.eval()
        with torch.no_grad():
            logits = model(dummy_input)
        assert logits.shape == (BATCH_SIZE, cfg.model.num_classes)
        assert isinstance(model, ViTSmallClassifier)

    def test_hybrid_concat(self, cfg, dummy_input):
        model = build_model(self._cfg_with_variant(cfg, "hybrid_concat"))
        model.eval()
        with torch.no_grad():
            logits = model(dummy_input)
        assert logits.shape == (BATCH_SIZE, cfg.model.num_classes)
        assert isinstance(model, HybridCapsuleModel)

    def test_hybrid_adaptive_builds(self, cfg, dummy_input):
        """Phase 5: hybrid_adaptive must now build successfully and return HybridCapsuleModel."""
        model = build_model(self._cfg_with_variant(cfg, "hybrid_adaptive"))
        model.eval()
        with torch.no_grad():
            logits = model(dummy_input)
        assert logits.shape == (BATCH_SIZE, cfg.model.num_classes)
        assert isinstance(model, HybridCapsuleModel)

    def test_unknown_variant_raises(self, cfg):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self._cfg_with_variant(cfg, "nonexistent_model")

        # Test factory fallback check
        class DummyModelConfig:
            variant = "nonexistent_model"
            pretrained = False
            num_classes = 8
            dropout_rate = 0.2
            freeze_backbone_epochs = 5
            fusion_hidden_dim = 512

        class DummyConfig:
            model = DummyModelConfig()

        with pytest.raises(ValueError, match="Unknown model variant"):
            build_model(DummyConfig())
