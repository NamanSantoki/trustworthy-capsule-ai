"""train.py — Training entry point for Trustworthy-Capsule-AI.

Wires together the full training pipeline:
    config → dataloaders → model → optimizer → trainer → checkpoints → history

Usage
-----
Fresh run (hybrid baseline):
    python train.py

Custom config directory:
    python train.py --config-dir config

Resume interrupted training:
    # Set resume_training: true in config/training.yaml, then:
    python train.py

Switch ablation variant (no code changes needed):
    # Edit config/model.yaml:  variant: vit_only
    python train.py

Outputs
-------
    outputs/checkpoints/best_val_loss.pth
    outputs/checkpoints/best_f1.pth
    outputs/checkpoints/last_epoch.pth
    outputs/history.json
    outputs/logs/            ← TensorBoard event files
"""

import argparse
import os
import torch

from src.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.data.dataloader import create_dataloaders
from src.models.factory import build_model
from src.training.optimizer import build_optimizer
from src.training.losses import build_loss
from src.training.scheduler import build_scheduler
from src.training.checkpoint import CheckpointManager
from src.training.history import ExperimentTracker
from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trustworthy-Capsule-AI — Training Script"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Path to the directory containing YAML config files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Configuration ──────────────────────────────────────────────────────────
    cfg = load_config(args.config_dir)

    # Scope output paths by model variant to avoid overwriting baseline results
    variant = cfg.model.variant
    cfg.paths.outputs_dir = os.path.join(cfg.paths.outputs_dir, variant)
    cfg.paths.checkpoints_dir = os.path.join(cfg.paths.outputs_dir, "checkpoints")
    cfg.paths.logs_dir = os.path.join(cfg.paths.outputs_dir, "logs")
    cfg.paths.plots_dir = os.path.join(cfg.paths.outputs_dir, "plots")
    cfg.paths.reports_dir = os.path.join(cfg.paths.outputs_dir, "reports")

    # Correct default resume checkpoint path if it points to the generic location
    if cfg.training.resume_checkpoint == "outputs/checkpoints/last_epoch.pth":
        cfg.training.resume_checkpoint = os.path.join(cfg.paths.checkpoints_dir, "last_epoch.pth")

    # ── Reproducibility ────────────────────────────────────────────────────────
    set_seed(cfg.seed)

    # ── Logging ───────────────────────────────────────────────────────────────
    os.makedirs(cfg.paths.logs_dir, exist_ok=True)
    logger = setup_logger(
        name="Trainer",
        log_dir=cfg.paths.logs_dir,
        level=20,  # INFO
    )
    logger.info(f"Project : {cfg.project_name}")
    logger.info(f"Variant : {cfg.model.variant}")
    logger.info(f"Seed    : {cfg.seed}")

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(
        cfg.device
        if (cfg.device == "cuda" and torch.cuda.is_available())
        else "cpu"
    )
    logger.info(f"Device  : {device}")
    if device.type == "cuda":
        logger.info(f"GPU     : {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Building dataloaders …")
    train_loader, val_loader, _ = create_dataloaders(cfg)
    class_names = train_loader.dataset.classes
    logger.info(f"Classes : {class_names}")
    logger.info(
        f"Samples — train: {len(train_loader.dataset)}, "
        f"val: {len(val_loader.dataset)}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info(f"Building model: {cfg.model.variant} …")
    model = build_model(cfg).to(device)

    if hasattr(model, "count_parameters"):
        counts = model.count_parameters()
        for k, v in counts.items():
            logger.info(f"  {k}: {v:,}")

    # Log adaptive fusion module parameter breakdown when applicable
    if (
        hasattr(model, "fusion_module")
        and hasattr(model.fusion_module, "count_parameters")
    ):
        logger.info("  -- Fusion module breakdown --")
        fusion_counts = model.fusion_module.count_parameters()
        for k, v in fusion_counts.items():
            logger.info(f"    fusion.{k}: {v:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = build_optimizer(cfg, model)

    # ── Loss & Scheduler ──────────────────────────────────────────────────────
    criterion = build_loss(cfg)
    scheduler = build_scheduler(cfg, optimizer)

    # ── Infrastructure ────────────────────────────────────────────────────────
    tracker     = ExperimentTracker(
        experiment_name=f"{cfg.project_name}_{cfg.model.variant}"
    )
    ckpt_mgr    = CheckpointManager(cfg)
    start_epoch = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    if cfg.training.resume_training:
        resume_path = cfg.training.resume_checkpoint
        logger.info(f"Resuming from checkpoint: {resume_path}")
        start_epoch, states = ckpt_mgr.load_for_resume(resume_path)
        model.load_state_dict(states["model"])
        optimizer.load_state_dict(states["optimizer"])
        logger.info(f"Resumed from epoch {start_epoch - 1} -> starting epoch {start_epoch}")

        # Restore history
        history_path = os.path.join(cfg.paths.outputs_dir, "history.json")
        if os.path.exists(history_path):
            tracker.load(history_path)
            logger.info(f"History restored: {len(tracker)} epochs")

    # -- Train --
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        cfg=cfg,
        tracker=tracker,
        checkpoint_manager=ckpt_mgr,
        device=device,
        class_names=class_names,
    )

    tracker = trainer.train(start_epoch=start_epoch)

    # ── Save history ──────────────────────────────────────────────────────────
    os.makedirs(cfg.paths.outputs_dir, exist_ok=True)
    history_path = os.path.join(cfg.paths.outputs_dir, "history.json")
    tracker.save(history_path)
    logger.info(f"Training history saved → {history_path}")
    logger.info("\n" + tracker.summary())


if __name__ == "__main__":
    main()
