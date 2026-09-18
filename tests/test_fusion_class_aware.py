"""Tests for ClassAwareAdaptiveFusion module."""

import pytest
import torch
import torch.nn as nn

from src.models.fusion.class_aware import ClassAwareAdaptiveFusion
from src.config import load_config
from src.models.factory import build_model


class TestClassAwareAdaptiveFusion:
    def test_instantiation(self):
        fusion = ClassAwareAdaptiveFusion(
            cnn_dim=1280,
            transformer_dim=384,
            num_classes=8,
            d_model=256,
            num_heads=8,
            hidden_dim=512,
            dropout_rate=0.2,
            prototype_temperature=0.1,
            normalize_prototypes=True,
        )
        assert fusion is not None
        assert fusion.class_prototypes.shape == (8, 256)

    def test_forward_shape(self):
        fusion = ClassAwareAdaptiveFusion(
            cnn_dim=1280,
            transformer_dim=384,
            num_classes=8,
            d_model=256,
            num_heads=8,
        )
        B = 4
        cnn_feat = torch.randn(B, 1280)
        vit_feat = torch.randn(B, 384)
        out = fusion(cnn_feat, vit_feat)
        assert out.shape == (B, 8)
        assert not torch.isnan(out).any()

    def test_forward_with_diagnostics(self):
        fusion = ClassAwareAdaptiveFusion(
            cnn_dim=1280,
            transformer_dim=384,
            num_classes=8,
            d_model=256,
            num_heads=8,
        )
        B = 2
        cnn_feat = torch.randn(B, 1280)
        vit_feat = torch.randn(B, 384)
        diag = fusion.forward_with_diagnostics(cnn_feat, vit_feat)

        assert "logits" in diag
        assert "prototype_similarities" in diag
        assert "prototype_weights" in diag
        assert "prototype_representation" in diag
        assert "gate_alpha" in diag

        assert diag["logits"].shape == (B, 8)
        assert diag["prototype_similarities"].shape == (B, 8)
        assert diag["prototype_weights"].shape == (B, 8)
        assert diag["prototype_representation"].shape == (B, 256)
        assert diag["gate_alpha"].shape == (B, 256)

        # Prototype weights should sum to 1 across classes
        weights_sum = diag["prototype_weights"].sum(dim=1)
        assert torch.allclose(weights_sum, torch.ones(B), atol=1e-5)

        # Gate values should be in (0, 1)
        assert (diag["gate_alpha"] >= 0.0).all() and (diag["gate_alpha"] <= 1.0).all()

    def test_factory_build(self):
        cfg = load_config("config_kvasir")
        assert cfg.model.variant == "hybrid_class_aware"
        model = build_model(cfg)
        assert model is not None

        # Test forward pass with dummy image batch
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, cfg.model.num_classes)
        assert not torch.isnan(out).any()

    def test_backprop_gradients(self):
        fusion = ClassAwareAdaptiveFusion(
            cnn_dim=1280,
            transformer_dim=384,
            num_classes=8,
            d_model=256,
            num_heads=8,
        )
        cnn_feat = torch.randn(2, 1280, requires_grad=True)
        vit_feat = torch.randn(2, 384, requires_grad=True)
        out = fusion(cnn_feat, vit_feat)
        loss = out.sum()
        loss.backward()

        assert cnn_feat.grad is not None
        assert vit_feat.grad is not None
        assert fusion.class_prototypes.grad is not None
        assert not torch.isnan(fusion.class_prototypes.grad).any()
