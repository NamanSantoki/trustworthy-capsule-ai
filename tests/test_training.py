"""Unit tests for Phase 4: Training Engine.

Tests cover:
  - MetricsCalculator: stateless metric evaluation of precision, recall, F1, accuracy, ROC-AUC.
  - ExperimentTracker & EpochRecord: tracking history, converting to DataFrame, and saving/loading JSON.
  - build_loss: verifying CrossEntropyLoss construction and label smoothing parameter.
  - build_scheduler: verifying Cosine and Plateau scheduler creation.
  - EarlyStopping: validation mode monitoring and patience counter updates.
  - CheckpointManager: model/optimizer checkpoint persistence and resume loaders.
  - Trainer: A model-agnostic smoke training run with a dummy CNN model and simple Tensor Dict data loader.
"""

import os
import tempfile
import shutil
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD

from src.config import load_config, AppConfig
from src.training.metrics import MetricsCalculator, MetricsResult
from src.training.history import ExperimentTracker, EpochRecord
from src.training.losses import build_loss
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler, is_plateau_scheduler
from src.training.callbacks import EarlyStopping
from src.training.checkpoint import CheckpointManager
from src.training.trainer import Trainer


# ─── Setup Temporary Workspace for File-based Tests ─────────────────────────────

@pytest.fixture
def temp_dir():
    """Create a temporary directory for configuration / checkpoints / histories."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def app_config(temp_dir) -> AppConfig:
    """Fixture containing a standard AppConfig instance pointing to temp directories."""
    cfg = load_config("config")
    cfg.paths.checkpoints_dir = os.path.join(temp_dir, "checkpoints")
    cfg.paths.outputs_dir = os.path.join(temp_dir, "outputs")
    cfg.paths.logs_dir = os.path.join(temp_dir, "logs")
    cfg.training.epochs = 2
    cfg.training.batch_size = 4
    cfg.training.num_workers = 0
    cfg.training.mixed_precision = False
    cfg.training.tensorboard = False
    return cfg


# ─── MetricsCalculator Tests ──────────────────────────────────────────────────

class TestMetricsCalculator:

    def test_metrics_evaluation(self):
        # 3 samples, 2 classes (0 and 1)
        y_true = np.array([0, 1, 1], dtype=np.int64)
        y_pred = np.array([0, 1, 0], dtype=np.int64)
        # Class probabilities: shape (3, 2)
        y_proba = np.array([
            [0.8, 0.2],
            [0.1, 0.9],
            [0.7, 0.3]
        ], dtype=np.float32)
        class_names = ["normal", "abnormal"]

        result = MetricsCalculator.compute_all(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            num_classes=2,
            class_names=class_names
        )

        assert isinstance(result, MetricsResult)
        assert result.accuracy == pytest.approx(2.0 / 3.0)
        assert result.num_samples == 3
        # Class 1: True labels are [1, 1], predictions [1, 0] -> recall=0.5, precision=1.0 -> F1 = 2/3
        # Class 0: True label is [0], prediction [0, 0] (actually sample 3 predicted 0, sample 1 predicted 0). 
        # For class 0: True is [1, 0, 0] (as negative, positive, negative). True is [0], preds are [0, 0], one false positive.
        assert result.confusion_matrix.shape == (2, 2)
        assert "normal" in result.f1_per_class
        assert "abnormal" in result.f1_per_class
        
        # Test serialization
        d = result.to_dict()
        assert isinstance(d["confusion_matrix"], list)
        restored = MetricsResult.from_dict(d)
        assert np.array_equal(result.confusion_matrix, restored.confusion_matrix)


# ─── ExperimentTracker & EpochRecord Tests ────────────────────────────────────

class TestExperimentTracker:

    def test_tracker_logging_and_serialization(self, temp_dir):
        tracker = ExperimentTracker("test_experiment")
        metrics_dict = {
            "accuracy": 0.8,
            "balanced_accuracy": 0.8,
            "precision_macro": 0.8,
            "recall_macro": 0.8,
            "f1_macro": 0.8,
            "f1_per_class": {"class0": 0.8, "class1": 0.8},
            "roc_auc_macro": 0.9,
            "roc_auc_per_class": {"class0": 0.9, "class1": 0.9},
            "confusion_matrix": [[2, 0], [0, 2]],
            "num_samples": 4
        }
        
        # Add epoch records
        record0 = EpochRecord(
            epoch=0,
            train_loss=0.5,
            val_loss=0.4,
            metrics=metrics_dict,
            learning_rate=0.001,
            elapsed_seconds=1.5
        )
        record1 = EpochRecord(
            epoch=1,
            train_loss=0.3,
            val_loss=0.2,
            metrics=metrics_dict,
            learning_rate=0.0005,
            elapsed_seconds=1.6
        )

        tracker.log_epoch(record0)
        tracker.log_epoch(record1)

        assert len(tracker) == 2
        assert tracker[0].epoch == 0
        assert tracker[1].epoch == 1

        best_loss = tracker.get_best_epoch("val_loss")
        assert best_loss.epoch == 1
        assert best_loss.val_loss == 0.2

        best_f1 = tracker.get_best_epoch("f1_macro")
        assert best_f1.epoch == 0 or best_f1.epoch == 1 # they both have 0.8

        summary = tracker.summary()
        assert "Last epoch" in summary

        df = tracker.to_dataframe()
        assert "train_loss" in df.columns
        assert "f1_macro" in df.columns

        # Save and load round-trip
        json_path = os.path.join(temp_dir, "history.json")
        tracker.save(json_path)

        tracker2 = ExperimentTracker()
        tracker2.load(json_path)
        assert len(tracker2) == 2
        assert tracker2.experiment_name == "test_experiment"
        assert tracker2[0].train_loss == 0.5


# ─── build_loss Tests ─────────────────────────────────────────────────────────

class TestLossFactory:

    def test_cross_entropy_loss_with_label_smoothing(self, app_config):
        app_config.training.label_smoothing = 0.15
        criterion = build_loss(app_config)

        assert isinstance(criterion, nn.CrossEntropyLoss)
        assert criterion.label_smoothing == pytest.approx(0.15)


# ─── build_optimizer Tests ──────────────────────────────────────────────────

class TestOptimizerFactory:

    def test_optimizer_differential_learning_rates_standalone(self, app_config):
        class MockBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 10)

        class MockStandaloneModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = MockBackbone()
                self.classifier = nn.Linear(10, 8)

        model = MockStandaloneModel()
        app_config.training.learning_rate = 0.0005
        app_config.training.backbone_lr = 0.00005
        app_config.training.weight_decay = 0.0001

        optimizer = build_optimizer(app_config, model)
        assert len(optimizer.param_groups) == 2
        # Head group
        assert optimizer.param_groups[0]["name"] == "head"
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0005)
        assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.0001)
        # Backbone group
        assert optimizer.param_groups[1]["name"] == "backbone"
        assert optimizer.param_groups[1]["lr"] == pytest.approx(0.00005)
        assert optimizer.param_groups[1]["weight_decay"] == pytest.approx(0.0001)

    def test_optimizer_differential_learning_rates_hybrid(self, app_config):
        class MockBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 10)

        class MockHybridModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cnn_backbone = MockBackbone()
                self.transformer_backbone = MockBackbone()
                self.fusion_module = nn.Linear(20, 8)

        model = MockHybridModel()
        app_config.training.learning_rate = 0.0005
        app_config.training.backbone_lr = 0.00005

        optimizer = build_optimizer(app_config, model)
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["name"] == "head"
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0005)
        assert optimizer.param_groups[1]["name"] == "backbone"
        assert optimizer.param_groups[1]["lr"] == pytest.approx(0.00005)


# ─── build_scheduler Tests ────────────────────────────────────────────────────

class TestSchedulerFactory:

    def test_cosine_scheduler_creation(self, app_config):
        optimizer = SGD([nn.Parameter(torch.randn(2, 2))], lr=0.1)
        app_config.training.scheduler = "cosine"
        scheduler = build_scheduler(app_config, optimizer)
        
        assert not is_plateau_scheduler(scheduler)
        assert scheduler.T_max == app_config.training.epochs

    def test_plateau_scheduler_creation(self, app_config):
        optimizer = SGD([nn.Parameter(torch.randn(2, 2))], lr=0.1)
        app_config.training.scheduler = "plateau"
        scheduler = build_scheduler(app_config, optimizer)
        
        assert is_plateau_scheduler(scheduler)


# ─── EarlyStopping Tests ──────────────────────────────────────────────────────

class TestEarlyStopping:

    def test_early_stopping_minimization_mode(self):
        es = EarlyStopping(patience=3, mode="min", min_delta=0.01)
        
        # Improvements
        assert not es.step(1.0)
        assert not es.step(0.9) # improvement
        assert es.counter == 0

        # No improvement
        assert not es.step(0.895) # delta < 0.01
        assert es.counter == 1

        assert not es.step(0.95)  # increase
        assert es.counter == 2

        # Triggers
        assert es.step(0.92)      # counter = 3 -> triggers should_stop
        assert es.triggered
        assert es.should_stop

    def test_early_stopping_state_dict_roundtrip(self):
        es = EarlyStopping(patience=5, mode="max")
        es.step(0.5)
        es.step(0.4) # counter = 1
        
        state = es.state_dict()
        
        es2 = EarlyStopping(patience=10, mode="min")
        es2.load_state_dict(state)
        
        assert es2.patience == 5
        assert es2.mode == "max"
        assert es2.counter == 1
        assert es2.best_value == 0.5


# ─── CheckpointManager Tests ──────────────────────────────────────────────────

class TestCheckpointManager:

    def test_checkpoint_saving_and_loading(self, app_config):
        manager = CheckpointManager(app_config)
        
        # Create dummy states
        model_state = {"weights": torch.tensor([1.0, 2.0])}
        optimizer_state = {"lr": 0.01}
        early_stopping_state = {"counter": 0}
        metrics_dict = {"f1_macro": 0.75}

        # 1. First Save
        updated1 = manager.save(
            epoch=0,
            model_state=model_state,
            optimizer_state=optimizer_state,
            scaler_state=None,
            early_stopping_state=early_stopping_state,
            metrics=metrics_dict,
            val_loss=0.5,
            f1_macro=0.75
        )
        assert updated1["last"]
        assert updated1["best_val_loss"]
        assert updated1["best_f1"]

        # Verify files exist
        assert os.path.exists(os.path.join(app_config.paths.checkpoints_dir, CheckpointManager.LAST_EPOCH))
        assert os.path.exists(os.path.join(app_config.paths.checkpoints_dir, CheckpointManager.BEST_VAL_LOSS))
        assert os.path.exists(os.path.join(app_config.paths.checkpoints_dir, CheckpointManager.BEST_F1))

        # 2. Second Save (better F1, worse loss)
        updated2 = manager.save(
            epoch=1,
            model_state=model_state,
            optimizer_state=optimizer_state,
            scaler_state=None,
            early_stopping_state=early_stopping_state,
            metrics={"f1_macro": 0.85},
            val_loss=0.6,
            f1_macro=0.85
        )
        assert updated2["last"]
        assert not updated2["best_val_loss"]
        assert updated2["best_f1"]

        # 3. Load for Resume
        start_epoch, states = manager.load_for_resume()
        assert start_epoch == 2
        assert torch.equal(states["model"]["weights"], model_state["weights"])
        assert states["optimizer"]["lr"] == 0.01


# ─── Dummy Model, Dataset & Trainer Smoke Tests ────────────────────────────────

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)
        # Setup attributes for the Trainer's unfreeze scheduler duck-typing
        self.freeze_backbone_epochs = 1
        self.backbone_frozen = True

    def forward(self, x):
        return self.fc(x)

    def freeze_backbones(self):
        self.backbone_frozen = True

    def unfreeze_backbones(self):
        self.backbone_frozen = False


class DummyDataset(Dataset):
    def __init__(self):
        self.classes = ["0_normal", "1_tumor"]

    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3), # dummy 3-dim vector
            "label": torch.tensor(idx % 2, dtype=torch.long),
            "path": f"/dummy/path/{idx}.png",
            "class_name": self.classes[idx % 2]
        }


class TestTrainerSmokeRun:

    def test_trainer_executes_epochs(self, app_config):
        app_config.model.num_classes = 2
        device = torch.device("cpu")
        model = DummyModel()
        
        dataset = DummyDataset()
        train_loader = DataLoader(dataset, batch_size=4, shuffle=False)
        val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

        optimizer = SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        scheduler = build_scheduler(app_config, optimizer)
        tracker = ExperimentTracker("trainer_smoke")
        manager = CheckpointManager(app_config)

        # Run trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            cfg=app_config,
            tracker=tracker,
            checkpoint_manager=manager,
            device=device,
            class_names=dataset.classes
        )

        assert model.backbone_frozen # Verify frozen at init

        tracker_out = trainer.train(start_epoch=0)

        # Verify warm-up unfreezing triggers
        assert not model.backbone_frozen
        
        # Verify tracker output matches the number of epochs
        assert len(tracker_out) == 2
        assert tracker_out[0].epoch == 0
        assert tracker_out[1].epoch == 1

        # Check that checkpoints have been saved
        assert os.path.exists(os.path.join(app_config.paths.checkpoints_dir, CheckpointManager.LAST_EPOCH))
