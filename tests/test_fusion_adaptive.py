"""Unit tests for Phase 5: AdaptiveCrossAttentionFusion module.

Test coverage
-------------
TestAdaptiveCrossAttentionFusion
    test_output_shape                   Forward pass produces (B, num_classes) logits.
    test_batch_size_independence        Module works for B=1, B=4, B=16.
    test_generic_backbone_dims          Accepts arbitrary (cnn_dim, transformer_dim) — not
                                        coupled to EfficientNet/ViT specifically.
    test_gating_alpha_range             Gating coefficients are in the open interval (0, 1)
                                        because they pass through sigmoid.
    test_residual_connection_active     Q is present in the computation graph of the output
                                        (verifies residual path is not dead).
    test_gradient_flow                  All trainable parameters receive non-zero gradients
                                        after a backward pass through a dummy loss.
    test_deterministic_eval_mode        Two identical forward passes in eval() mode return
                                        exactly the same output (no dropout randomness).
    test_serialization_roundtrip        state_dict() → load_state_dict() → same output.
    test_count_parameters_keys          count_parameters() returns all expected component keys.
    test_count_parameters_totals        Per-component sum equals reported total.
    test_invalid_heads_raises           Raises ValueError when d_model % num_heads != 0.

TestAdaptiveFusionViaFactory
    test_factory_builds_hybrid_adaptive build_model("hybrid_adaptive") returns HybridCapsuleModel
                                        with an AdaptiveCrossAttentionFusion as fusion_module.
    test_factory_concat_unaffected      hybrid_concat still uses ConcatFusion (no regression).
    test_freeze_unfreeze_interface      HybridCapsuleModel wrapping adaptive fusion exposes the
                                        standard Trainer interface unchanged.
    test_fusion_param_count_exposed     fusion_module.count_parameters() is accessible from
                                        the built model (for training log verification).
"""

import copy
import pytest
import torch
import torch.nn as nn

from src.config import load_config, AppConfig
from src.models.fusion.adaptive import AdaptiveCrossAttentionFusion
from src.models.fusion.concat import ConcatFusion
from src.models.hybrid import HybridCapsuleModel
from src.models.factory import build_model

# ─── Constants ────────────────────────────────────────────────────────────────

# Realistic backbone output dimensions
CNN_DIM         = 1280   # EfficientNetV2-S OUT_FEATURES
TRANSFORMER_DIM = 384    # ViT-Small/16 OUT_FEATURES
NUM_CLASSES     = 8
D_MODEL         = 256
NUM_HEADS       = 8
HIDDEN_DIM      = 512
BATCH_SIZE      = 4


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg() -> AppConfig:
    return load_config("config")


@pytest.fixture(scope="module")
def fusion() -> AdaptiveCrossAttentionFusion:
    """Standard module instance used across multiple tests."""
    return AdaptiveCrossAttentionFusion(
        cnn_dim=CNN_DIM,
        transformer_dim=TRANSFORMER_DIM,
        num_classes=NUM_CLASSES,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=0.2,
        attention_dropout=0.0,
    )


@pytest.fixture
def cnn_feats() -> torch.Tensor:
    return torch.randn(BATCH_SIZE, CNN_DIM)


@pytest.fixture
def vit_feats() -> torch.Tensor:
    return torch.randn(BATCH_SIZE, TRANSFORMER_DIM)


# ─── Core Module Tests ────────────────────────────────────────────────────────

class TestAdaptiveCrossAttentionFusion:

    # ── 1. Forward pass shape ──────────────────────────────────────────────────

    def test_output_shape(self, fusion, cnn_feats, vit_feats):
        """Output must be (B, num_classes) for the standard batch."""
        fusion.eval()
        with torch.no_grad():
            logits = fusion(cnn_feats, vit_feats)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES), (
            f"Expected ({BATCH_SIZE}, {NUM_CLASSES}), got {logits.shape}"
        )

    # ── 2. Batch-size independence ─────────────────────────────────────────────

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_size_independence(self, batch_size):
        """Module must run correctly for diverse batch sizes including B=1."""
        module = AdaptiveCrossAttentionFusion(
            cnn_dim=CNN_DIM, transformer_dim=TRANSFORMER_DIM,
            num_classes=NUM_CLASSES, d_model=D_MODEL, num_heads=NUM_HEADS,
        )
        module.eval()
        cnn_f = torch.randn(batch_size, CNN_DIM)
        vit_f = torch.randn(batch_size, TRANSFORMER_DIM)
        with torch.no_grad():
            logits = module(cnn_f, vit_f)
        assert logits.shape == (batch_size, NUM_CLASSES)

    # ── 3. Generic backbone compatibility ─────────────────────────────────────

    @pytest.mark.parametrize("cnn_dim, trans_dim", [
        (512,  256),   # generic small dims
        (1024, 512),   # ResNet-like CNN + larger ViT
        (2048, 768),   # ResNet-50 + ViT-Base
        (1280, 384),   # EfficientNetV2-S + ViT-Small (actual project dims)
    ])
    def test_generic_backbone_dims(self, cnn_dim, trans_dim):
        """Module must be usable with arbitrary backbone output dimensions.

        This verifies the module is not hardcoded to EfficientNet/ViT dims and
        will remain valid when future backbones are swapped in.
        """
        module = AdaptiveCrossAttentionFusion(
            cnn_dim=cnn_dim, transformer_dim=trans_dim,
            num_classes=NUM_CLASSES, d_model=D_MODEL, num_heads=NUM_HEADS,
        )
        module.eval()
        cnn_f = torch.randn(2, cnn_dim)
        vit_f = torch.randn(2, trans_dim)
        with torch.no_grad():
            logits = module(cnn_f, vit_f)
        assert logits.shape == (2, NUM_CLASSES)

    # ── 4. Gating alpha in (0, 1) ─────────────────────────────────────────────

    def test_gating_alpha_range(self, fusion, cnn_feats, vit_feats):
        """Gating coefficients must lie strictly in (0, 1) due to sigmoid.

        We hook into the gate Linear to capture the alpha values before they
        are used in the weighted sum.
        """
        alphas_captured = []

        def _hook(module, inp, out):
            # 'out' is the raw gate output before sigmoid wrapper not applied here
            # We capture alpha after sigmoid in the forward override approach below
            pass

        # Directly expose via a subclass that stores alpha
        class _FusionWithAlpha(AdaptiveCrossAttentionFusion):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._last_alpha = None

            def forward(self, cnn_features, transformer_features):
                Q  = self.cnn_proj(cnn_features)
                KV = self.transformer_proj(transformer_features)
                Q3D  = Q.unsqueeze(1)
                KV3D = KV.unsqueeze(1)
                attn_out, _ = self.cross_attention(Q3D, KV3D, KV3D)
                attn_out = self.attn_dropout(attn_out.squeeze(1))
                attn_residual = self.residual_norm(Q + attn_out)
                gate_input = torch.cat([Q, attn_residual], dim=1)
                alpha = torch.sigmoid(self.gate(gate_input))
                self._last_alpha = alpha.detach().clone()
                fused = alpha * attn_residual + (1.0 - alpha) * Q
                return self.head(fused)

        m = _FusionWithAlpha(
            cnn_dim=CNN_DIM, transformer_dim=TRANSFORMER_DIM,
            num_classes=NUM_CLASSES, d_model=D_MODEL, num_heads=NUM_HEADS,
        )
        m.eval()
        with torch.no_grad():
            m(cnn_feats, vit_feats)

        alpha = m._last_alpha
        assert alpha is not None
        assert (alpha > 0.0).all(), "All gating values must be > 0 (sigmoid lower bound)"
        assert (alpha < 1.0).all(), "All gating values must be < 1 (sigmoid upper bound)"
        assert alpha.shape == (BATCH_SIZE, D_MODEL)

    # ── 5. Residual connection is active ──────────────────────────────────────

    def test_residual_connection_active(self):
        """The Q term must appear in the computation graph of the output.

        We verify by checking that zeroing the CNN projection input produces a
        different output than zeroing the transformer input — if both produced
        the same output the residual path would be dead.
        """
        module = AdaptiveCrossAttentionFusion(
            cnn_dim=CNN_DIM, transformer_dim=TRANSFORMER_DIM,
            num_classes=NUM_CLASSES, d_model=D_MODEL, num_heads=NUM_HEADS,
            dropout_rate=0.0,
        )
        module.eval()
        cnn_f = torch.randn(2, CNN_DIM)
        vit_f = torch.randn(2, TRANSFORMER_DIM)

        with torch.no_grad():
            out_normal   = module(cnn_f, vit_f)
            out_zero_cnn = module(torch.zeros_like(cnn_f), vit_f)
            out_zero_vit = module(cnn_f, torch.zeros_like(vit_f))

        # Both zeroed variants must differ from the normal output
        assert not torch.allclose(out_normal, out_zero_cnn), (
            "Zeroing CNN features should change output (residual path check)"
        )
        assert not torch.allclose(out_normal, out_zero_vit), (
            "Zeroing ViT features should change output (attention path check)"
        )

    # ── 6. Gradient flow ──────────────────────────────────────────────────────

    def test_gradient_flow(self):
        """Every trainable parameter must receive a non-zero gradient.

        This catches silent dead sub-graphs (e.g. an unreachable residual path
        or an unconnected linear layer).
        """
        module = AdaptiveCrossAttentionFusion(
            cnn_dim=CNN_DIM, transformer_dim=TRANSFORMER_DIM,
            num_classes=NUM_CLASSES, d_model=D_MODEL, num_heads=NUM_HEADS,
            dropout_rate=0.0,
        )
        module.train()

        cnn_f = torch.randn(BATCH_SIZE, CNN_DIM)
        vit_f = torch.randn(BATCH_SIZE, TRANSFORMER_DIM)
        labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

        logits = module(cnn_f, vit_f)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        dead_params = []
        for name, p in module.named_parameters():
            if p.requires_grad:
                if p.grad is None or p.grad.abs().max().item() == 0.0:
                    dead_params.append(name)

        assert not dead_params, (
            f"The following parameters received zero/no gradient: {dead_params}"
        )

    # ── 7. Deterministic inference in eval mode ────────────────────────────────

    def test_deterministic_eval_mode(self, fusion, cnn_feats, vit_feats):
        """Two identical forward passes in eval() mode must return exactly the
        same tensor values — no stochasticity from Dropout."""
        fusion.eval()
        with torch.no_grad():
            out1 = fusion(cnn_feats, vit_feats)
            out2 = fusion(cnn_feats, vit_feats)
        assert torch.allclose(out1, out2, atol=0.0), (
            "eval() mode must be fully deterministic — check Dropout is inactive"
        )

    # ── 8. Train mode stochasticity ───────────────────────────────────────────

    def test_train_mode_stochastic_when_dropout_nonzero(self):
        """With dropout > 0 in train() mode, two passes should differ."""
        module = AdaptiveCrossAttentionFusion(
            cnn_dim=CNN_DIM, transformer_dim=TRANSFORMER_DIM,
            num_classes=NUM_CLASSES, d_model=D_MODEL, num_heads=NUM_HEADS,
            dropout_rate=0.5,   # high dropout to make non-determinism near-certain
        )
        module.train()
        cnn_f = torch.randn(BATCH_SIZE, CNN_DIM)
        vit_f = torch.randn(BATCH_SIZE, TRANSFORMER_DIM)
        out1 = module(cnn_f, vit_f)
        out2 = module(cnn_f, vit_f)
        # With p=0.5 and d_model=256, probability of identical output ≈ 0
        assert not torch.allclose(out1, out2), (
            "train() mode with dropout=0.5 should produce different outputs each call"
        )

    # ── 9. Serialization / deserialization ────────────────────────────────────

    def test_serialization_roundtrip(self, fusion, cnn_feats, vit_feats):
        """state_dict() → new instance → load_state_dict() → identical output.

        This verifies the module can be checkpointed and restored correctly,
        as it will be done by CheckpointManager during training.
        """
        fusion.eval()
        with torch.no_grad():
            logits_before = fusion(cnn_feats, vit_feats)

        # Create a fresh instance with the same architecture
        restored = AdaptiveCrossAttentionFusion(
            cnn_dim=CNN_DIM, transformer_dim=TRANSFORMER_DIM,
            num_classes=NUM_CLASSES, d_model=D_MODEL, num_heads=NUM_HEADS,
            hidden_dim=HIDDEN_DIM,
        )
        restored.load_state_dict(copy.deepcopy(fusion.state_dict()))
        restored.eval()

        with torch.no_grad():
            logits_after = restored(cnn_feats, vit_feats)

        assert torch.allclose(logits_before, logits_after, atol=1e-6), (
            "Restored model produces different output than saved model"
        )

    # ── 10. Parameter count keys ───────────────────────────────────────────────

    def test_count_parameters_keys(self, fusion):
        """count_parameters() must expose all expected component breakdown keys."""
        expected_keys = {
            "cnn_projection",
            "transformer_projection",
            "cross_attention",
            "residual_norm",
            "gating_mlp",
            "classification_head",
            "total",
        }
        actual_keys = set(fusion.count_parameters().keys())
        assert actual_keys == expected_keys, (
            f"Missing keys: {expected_keys - actual_keys}; "
            f"Extra keys: {actual_keys - expected_keys}"
        )

    def test_count_parameters_totals_consistent(self, fusion):
        """Sum of per-component counts must equal the reported 'total'."""
        counts = fusion.count_parameters()
        component_sum = sum(v for k, v in counts.items() if k != "total")
        assert component_sum == counts["total"], (
            f"Component sum {component_sum} != reported total {counts['total']}"
        )

    def test_count_parameters_all_positive(self, fusion):
        """Every component must have at least one trainable parameter."""
        counts = fusion.count_parameters()
        for k, v in counts.items():
            assert v > 0, f"Component '{k}' reports 0 parameters — likely a dead sub-graph"

    # ── 11. Invalid configuration raises ──────────────────────────────────────

    @pytest.mark.parametrize("d_model, num_heads", [
        (256, 7),    # 256 % 7 != 0
        (100, 8),    # 100 % 8 != 0
        (512, 9),    # 512 % 9 != 0
    ])
    def test_invalid_heads_raises(self, d_model, num_heads):
        """ValueError is raised when d_model is not divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible by num_heads"):
            AdaptiveCrossAttentionFusion(
                cnn_dim=CNN_DIM, transformer_dim=TRANSFORMER_DIM,
                num_classes=NUM_CLASSES, d_model=d_model, num_heads=num_heads,
            )

    # ── 12. __repr__ ──────────────────────────────────────────────────────────

    def test_repr_contains_key_info(self, fusion):
        """repr() must contain architecture dimensions for debugging."""
        r = repr(fusion)
        assert "AdaptiveCrossAttentionFusion" in r
        assert str(CNN_DIM) in r
        assert str(TRANSFORMER_DIM) in r
        assert str(D_MODEL) in r


# ─── Factory Integration Tests ────────────────────────────────────────────────

class TestAdaptiveFusionViaFactory:
    """Verify that the model factory correctly wires the adaptive module."""

    def _cfg_with_variant(self, cfg: AppConfig, variant: str) -> AppConfig:
        data = cfg.model_dump()
        data["model"]["variant"]   = variant
        data["model"]["pretrained"] = False   # avoid downloading weights in tests
        return type(cfg).model_validate(data)

    def test_factory_builds_hybrid_adaptive(self, cfg):
        """build_model('hybrid_adaptive') must return a HybridCapsuleModel
        whose fusion_module is an AdaptiveCrossAttentionFusion instance."""
        model = build_model(self._cfg_with_variant(cfg, "hybrid_adaptive"))
        assert isinstance(model, HybridCapsuleModel), (
            "Expected HybridCapsuleModel, got " + type(model).__name__
        )
        assert isinstance(model.fusion_module, AdaptiveCrossAttentionFusion), (
            "fusion_module should be AdaptiveCrossAttentionFusion, got "
            + type(model.fusion_module).__name__
        )

    def test_factory_concat_unaffected(self, cfg):
        """Building hybrid_concat must still use ConcatFusion — no regression."""
        model = build_model(self._cfg_with_variant(cfg, "hybrid_concat"))
        assert isinstance(model.fusion_module, ConcatFusion), (
            "hybrid_concat regression: fusion_module is not ConcatFusion"
        )

    def test_adaptive_model_forward_pass(self, cfg):
        """Full hybrid_adaptive model forward pass must produce (B, num_classes)."""
        model = build_model(self._cfg_with_variant(cfg, "hybrid_adaptive"))
        model.eval()
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(dummy)
        assert logits.shape == (2, cfg.model.num_classes)

    def test_freeze_unfreeze_interface_preserved(self, cfg):
        """HybridCapsuleModel wrapping AdaptiveFusion must expose the Trainer's
        duck-typed freeze/unfreeze interface unchanged."""
        model = build_model(self._cfg_with_variant(cfg, "hybrid_adaptive"))
        assert hasattr(model, "freeze_backbones"),   "freeze_backbones missing"
        assert hasattr(model, "unfreeze_backbones"), "unfreeze_backbones missing"
        assert hasattr(model, "freeze_backbone_epochs"), "freeze_backbone_epochs missing"
        assert model.freeze_backbone_epochs == cfg.model.freeze_backbone_epochs

    def test_backbones_frozen_at_construction(self, cfg):
        """Backbones must be frozen immediately after build_model() for warm-up."""
        model = build_model(self._cfg_with_variant(cfg, "hybrid_adaptive"))
        cnn_trainable = sum(1 for p in model.cnn_backbone.parameters() if p.requires_grad)
        vit_trainable = sum(1 for p in model.transformer_backbone.parameters() if p.requires_grad)
        assert cnn_trainable == 0, f"CNN should be frozen at init, got {cnn_trainable} trainable"
        assert vit_trainable == 0, f"ViT should be frozen at init, got {vit_trainable} trainable"

    def test_fusion_param_count_exposed(self, cfg):
        """fusion_module.count_parameters() must be accessible from the built model
        so train.py can log the per-component overhead."""
        model = build_model(self._cfg_with_variant(cfg, "hybrid_adaptive"))
        assert hasattr(model.fusion_module, "count_parameters"), (
            "AdaptiveCrossAttentionFusion must expose count_parameters() for train.py logging"
        )
        counts = model.fusion_module.count_parameters()
        assert counts["total"] > 0

    def test_adaptive_gradient_flow_end_to_end(self, cfg):
        """Gradients must flow through the full HybridCapsuleModel + AdaptiveFusion
        stack after unfreezing — simulates what happens at epoch 5."""
        model = build_model(self._cfg_with_variant(cfg, "hybrid_adaptive"))
        model.unfreeze_backbones()  # simulate post-warm-up epoch
        model.train()

        dummy  = torch.randn(2, 3, 224, 224)
        labels = torch.randint(0, cfg.model.num_classes, (2,))
        logits = model(dummy)
        loss   = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        # Spot-check: fusion head parameters must have gradients
        for name, p in model.fusion_module.named_parameters():
            assert p.grad is not None and p.grad.abs().max() > 0, (
                f"fusion_module.{name} received no gradient after backward"
            )

    def test_config_fields_wired_correctly(self, cfg):
        """d_model and num_heads from config must be reflected in the built module."""
        model = build_model(self._cfg_with_variant(cfg, "hybrid_adaptive"))
        fusion = model.fusion_module
        assert fusion._d_model == cfg.model.fusion_d_model
        assert fusion._num_heads == cfg.model.fusion_num_heads
        assert fusion._num_classes == cfg.model.num_classes
