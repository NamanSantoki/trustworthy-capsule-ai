"""Tests for Monte Carlo Dropout uncertainty estimation."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.inference.uncertainty import MCDropoutPredictor, mc_dropout_mode, UncertaintyResult


class SimpleToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.bn = nn.BatchNorm1d(20)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(20, 4)

    def forward(self, x):
        return self.fc2(self.dropout(self.bn(self.fc1(x))))


class TestUncertainty:
    def test_mc_dropout_context_manager(self):
        model = SimpleToyModel()
        model.eval()

        assert not model.dropout.training
        assert not model.bn.training

        with mc_dropout_mode(model):
            # Dropout should be active (training=True)
            assert model.dropout.training
            # BatchNorm should remain inactive (training=False)
            assert not model.bn.training

        # Outside context, should be restored to eval
        assert not model.dropout.training
        assert not model.bn.training

    def test_mc_dropout_prediction_stochasticity(self):
        model = SimpleToyModel()
        predictor = MCDropoutPredictor(model)

        x = torch.randn(1, 10)
        result = predictor.predict(x, n_samples=30)

        assert isinstance(result, UncertaintyResult)
        assert result.n_samples == 30
        assert result.mean_proba.shape == (4,)
        assert 0.0 <= result.confidence <= 1.0
        assert result.predictive_entropy >= 0.0
        assert 0.0 <= result.variation_ratio <= 1.0
        assert result.all_probas.shape == (30, 4)

    def test_deterministic_mode(self):
        model = SimpleToyModel()
        predictor = MCDropoutPredictor(model)

        x = torch.randn(1, 10)
        result = predictor.predict(x, n_samples=0)

        assert result.n_samples == 1
        assert result.variation_ratio == 0.0
        assert 0.0 <= result.confidence <= 1.0
