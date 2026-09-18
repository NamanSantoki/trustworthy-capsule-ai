"""Experiment history tracker for Trustworthy-Capsule-AI.

Records per-epoch training and validation metrics, serialises/deserialises to
JSON, and supports resuming interrupted experiments.

Classes
-------
EpochRecord
    DataClass capturing all per-epoch measurements.

ExperimentTracker
    Append-only log of ``EpochRecord`` instances.  Provides helpers for
    best-epoch retrieval, tabular summaries, and JSON persistence.
"""

from __future__ import annotations

import json
import time
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional

from src.training.metrics import MetricsResult


# ──────────────────────────────────────────────────────────────────────────────
# Per-Epoch Record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EpochRecord:
    """All measurements captured at the end of a single training epoch.

    Attributes:
        epoch:           0-indexed epoch number.
        train_loss:      Average training loss for the epoch.
        val_loss:        Average validation loss for the epoch.
        metrics:         Full ``MetricsResult`` (serialised as dict).
        learning_rate:   Effective LR at the end of the epoch.
        elapsed_seconds: Wall-clock time for the epoch (train + val).
    """

    epoch:           int
    train_loss:      float
    val_loss:        float
    metrics:         dict           # MetricsResult.to_dict()
    learning_rate:   float
    elapsed_seconds: float

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def f1_macro(self) -> float:
        return self.metrics.get("f1_macro", float("nan"))

    @property
    def accuracy(self) -> float:
        return self.metrics.get("accuracy", float("nan"))

    @property
    def roc_auc_macro(self) -> float:
        return self.metrics.get("roc_auc_macro", float("nan"))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EpochRecord":
        return cls(**d)


# ──────────────────────────────────────────────────────────────────────────────
# Experiment Tracker
# ──────────────────────────────────────────────────────────────────────────────

class ExperimentTracker:
    """Append-only log of training epochs with JSON persistence.

    Designed to be decoupled from PyTorch — all data is stored as plain
    Python types so the log can be loaded and analysed without a GPU.

    Args:
        experiment_name: Human-readable label embedded in saved JSON.

    Example::

        tracker = ExperimentTracker("hybrid_concat_run1")
        tracker.log_epoch(EpochRecord(epoch=0, ...))
        tracker.save("outputs/history.json")

        # Later, for analysis or resuming:
        tracker2 = ExperimentTracker()
        tracker2.load("outputs/history.json")
    """

    def __init__(self, experiment_name: str = "") -> None:
        self.experiment_name = experiment_name
        self._records: list[EpochRecord] = []
        self._start_time: float = time.time()

    # ── Record management ─────────────────────────────────────────────────────

    def log_epoch(self, record: EpochRecord) -> None:
        """Append an epoch record to the history."""
        self._records.append(record)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> EpochRecord:
        return self._records[idx]

    # ── Best-epoch queries ────────────────────────────────────────────────────

    def get_best_epoch(self, metric: str = "val_loss") -> Optional[EpochRecord]:
        """Return the epoch record with the best value for ``metric``.

        For ``val_loss`` the minimum is best; for all other metrics the
        maximum is best.

        Args:
            metric: One of ``"val_loss"``, ``"f1_macro"``, ``"accuracy"``,
                    ``"roc_auc_macro"``, ``"balanced_accuracy"``.

        Returns:
            The best :class:`EpochRecord`, or ``None`` if history is empty.
        """
        if not self._records:
            return None

        minimize = metric == "val_loss"

        def _val(r: EpochRecord) -> float:
            if metric == "val_loss":
                return r.val_loss
            if metric == "train_loss":
                return r.train_loss
            return r.metrics.get(metric, float("-inf") if not minimize else float("inf"))

        return min(self._records, key=_val) if minimize else max(self._records, key=_val)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialise the full history to a JSON file.

        Args:
            path: Target file path (parent directories must exist).
        """
        payload = {
            "experiment_name": self.experiment_name,
            "num_epochs":      len(self._records),
            "records":         [r.to_dict() for r in self._records],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load(self, path: str) -> None:
        """Deserialise history from a JSON file and append to current records.

        Calling ``load()`` on an existing tracker extends its history, enabling
        seamless resume from a previous checkpoint.

        Args:
            path: Source file path produced by :meth:`save`.
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        loaded = [EpochRecord.from_dict(r) for r in payload.get("records", [])]
        self._records.extend(loaded)
        if not self.experiment_name and payload.get("experiment_name"):
            self.experiment_name = payload["experiment_name"]

    # ── Analysis helpers ──────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Return history as a flat ``pd.DataFrame`` suitable for plotting.

        Nested metric keys are unpacked into top-level columns
        (e.g., ``metrics.f1_macro`` → ``f1_macro``).
        """
        rows = []
        for r in self._records:
            row = {
                "epoch":           r.epoch,
                "train_loss":      r.train_loss,
                "val_loss":        r.val_loss,
                "learning_rate":   r.learning_rate,
                "elapsed_seconds": r.elapsed_seconds,
            }
            row.update({k: v for k, v in r.metrics.items() if not isinstance(v, (list, dict))})
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Return a human-readable summary string of the training run."""
        if not self._records:
            return "ExperimentTracker: no epochs recorded."

        last = self._records[-1]
        best_loss = self.get_best_epoch("val_loss")
        best_f1   = self.get_best_epoch("f1_macro")

        lines = [
            f"Experiment : {self.experiment_name or '(unnamed)'}",
            f"Epochs     : {len(self._records)}",
            f"Last epoch : #{last.epoch}  |  train_loss={last.train_loss:.4f}"
            f"  val_loss={last.val_loss:.4f}  f1={last.f1_macro:.4f}",
        ]
        if best_loss:
            lines.append(
                f"Best loss  : epoch #{best_loss.epoch}  val_loss={best_loss.val_loss:.4f}"
            )
        if best_f1:
            lines.append(
                f"Best F1    : epoch #{best_f1.epoch}  f1_macro={best_f1.f1_macro:.4f}"
            )
        return "\n".join(lines)
