"""evaluate.py — Standalone test-set evaluation script for Trustworthy-Capsule-AI.

Evaluates a trained model checkpoint on the test set, computing classification
metrics (Accuracy, Precision, Recall, Macro-F1, Confusion Matrix, Per-class metrics)
and calibration metrics (Expected Calibration Error, Brier Score, Reliability Diagram).

Usage
-----
Evaluate hybrid_class_aware on Kvasir test set:
    python evaluate.py --config-dir config_kvasir --checkpoint outputs_kvasir/checkpoints/best_f1.pth

Evaluate on default config:
    python evaluate.py --checkpoint outputs/checkpoints/best_f1.pth

Outputs
-------
    <reports_dir>/evaluation_report.json
    <reports_dir>/reliability_diagram.png
"""

import argparse
import logging
import os
import sys
import torch

from src.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.data.dataloader import create_dataloaders
from src.models.factory import build_model
from src.evaluation.evaluator import ModelEvaluator

CLASS_NAMES_8 = [
    "Angiectasia",
    "Erosion",
    "Ileocecal valve",
    "Lymphangiectasia",
    "Normal clean mucosa",
    "Pylorus",
    "Reduced mucosal view",
    "Ulcer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trustworthy-Capsule-AI — Model Evaluation and Calibration Script"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Directory containing YAML configuration files (default: config).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Override model variant (e.g. hybrid_class_aware).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation reports and plots.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on ('cuda' or 'cpu').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config_dir)
    if args.variant:
        cfg.model.variant = args.variant  # type: ignore

    setup_logger(
        log_dir=cfg.paths.logs_dir,
        log_file="evaluation.log",
        console_level=logging.INFO,
    )
    logger = logging.getLogger("evaluate")

    # Seed
    set_seed(cfg.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")

    # Data loaders
    logger.info(f"Loading dataset from: {cfg.paths.dataset_dir}")
    _, _, test_loader = create_dataloaders(cfg)

    class_names = getattr(test_loader.dataset, "classes", CLASS_NAMES_8[: cfg.model.num_classes])

    # Build model
    logger.info(f"Building model variant: {cfg.model.variant}")
    model = build_model(cfg)

    # Load checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        candidate_1 = os.path.join(cfg.paths.outputs_dir, cfg.model.variant, "checkpoints", "best_f1.pth")
        candidate_2 = os.path.join(cfg.paths.checkpoints_dir, "best_f1.pth")
        candidate_3 = f"outputs_kvasir/hybrid_class_aware/checkpoints/best_f1.pth"
        for cand in [candidate_1, candidate_2, candidate_3]:
            if os.path.exists(cand):
                ckpt_path = cand
                break

    if ckpt_path and os.path.exists(ckpt_path):
        logger.info(f"Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
    else:
        logger.warning(f"Checkpoint not found at {ckpt_path}. Evaluating uninitialised model weights.")

    model.to(device)

    # Run evaluator
    reports_dir = args.output_dir or cfg.paths.reports_dir
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        cfg=cfg,
        device=device,
        class_names=class_names,
    )

    logger.info("Starting test evaluation...")
    report = evaluator.evaluate()
    saved_path = evaluator.save_report(report, reports_dir=reports_dir, plot_diagram=True)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model Variant : {report['variant']}")
    print(f"Test Samples  : {report['num_samples']}")
    print(f"Accuracy      : {report['classification']['accuracy'] * 100:.2f}%")
    print(f"Macro F1      : {report['classification']['f1_macro'] * 100:.2f}%")
    print(f"Weighted F1   : {report['classification']['f1_weighted'] * 100:.2f}%")
    print(f"ECE           : {report['calibration']['ece']:.4f}")
    print(f"Brier Score   : {report['calibration']['brier_score']:.4f}")
    print(f"Report Saved  : {saved_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
