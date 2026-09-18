"""faithfulness.py — Quantitative faithfulness evaluation script for Trustworthy-Capsule-AI.

Evaluates whether Grad-CAM heatmaps highlight truly essential decision-making regions
using coarse grid-based spatial perturbation (Deletion AUC & Insertion AUC),
benchmarked against a Random Region Baseline and Least-Important Control.

Usage Examples:
---------------
1. Single Image Faithfulness Test:
    python faithfulness.py --config-dir config_kvasir --checkpoint outputs_kvasir/hybrid_class_aware/checkpoints/best_f1.pth --image Dataset/test/Ulcer/test_00000.jpg

2. Small Test Set Evaluation (10 images):
    python faithfulness.py --config-dir config_kvasir --checkpoint outputs_kvasir/hybrid_class_aware/checkpoints/best_f1.pth --split test --max-images 10

3. Moderate Test Set Evaluation (100 images):
    python faithfulness.py --config-dir config_kvasir --checkpoint outputs_kvasir/hybrid_class_aware/checkpoints/best_f1.pth --split test --max-images 100

4. Full Test Set Evaluation:
    python faithfulness.py --config-dir config_kvasir --checkpoint outputs_kvasir/hybrid_class_aware/checkpoints/best_f1.pth --split test

Outputs Saved:
--------------
- JSON Report: <output_dir>/reports/faithfulness_report.json
- Deletion Curve Plot: <output_dir>/plots/faithfulness_deletion.png
- Insertion Curve Plot: <output_dir>/plots/faithfulness_insertion.png
- Combined Comparison Plot: <output_dir>/plots/faithfulness_comparison.png
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import glob
from typing import Optional, List
import torch
from PIL import Image
import numpy as np

from src.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.models.factory import build_model
from src.explainability.utils import load_model_checkpoint, preprocess_image
from src.data.dataset import GenericCapsuleDataset
from src.data.transforms import build_val_test_transforms
from torch.utils.data import DataLoader
from src.evaluation.faithfulness import FaithfulnessEvaluator

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
        description="Trustworthy-Capsule-AI — Quantitative Explainability & Faithfulness Evaluation"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Directory containing YAML configuration files (e.g. config_kvasir or config).",
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
        help="Override model variant (e.g. hybrid_class_aware, hybrid_adaptive, efficientnet_only, vit_only).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image file for individual faithfulness evaluation.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Path to folder of images to evaluate.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate if --image is not provided (e.g. 'test' or 'val', default: test).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to evaluate (e.g. 10, 50, 100). If omitted, processes the entire split.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory to save faithfulness reports and plots.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Spatial grid dimension (default: 14 for 14x14 = 196 regions).",
    )
    parser.add_argument(
        "--perturbation",
        type=str,
        default=None,
        choices=["blur", "mean", "zero"],
        help="Perturbation strategy: 'blur' (default), 'mean', or 'zero'.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Target class index. Defaults to model's predicted class.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save intermediate Grad-CAM and masked progression images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load configuration
    cfg = load_config(args.config_dir)

    # CLI Overrides
    if args.variant:
        cfg.model.variant = args.variant  # type: ignore
    if args.seed is not None:
        cfg.seed = args.seed
    if args.grid_size is not None:
        cfg.faithfulness.grid_size = args.grid_size
    if args.perturbation is not None:
        cfg.faithfulness.perturbation = args.perturbation

    # Seed
    set_seed(cfg.seed)

    # Output directory resolution
    if args.output_dir:
        base_out = args.output_dir
    else:
        variant_tag = cfg.model.variant
        base_out = os.path.join(cfg.paths.outputs_dir, variant_tag, "faithfulness")

    reports_dir = os.path.join(base_out, "reports")
    plots_dir = os.path.join(base_out, "plots")
    vis_dir = os.path.join(base_out, "visualizations")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    if args.save_visualizations:
        os.makedirs(vis_dir, exist_ok=True)

    # Logger
    setup_logger(
        log_dir=cfg.paths.logs_dir,
        log_file="faithfulness.log",
        console_level=logging.INFO,
    )
    logger = logging.getLogger("faithfulness")
    logger.info(f"Loaded config from '{args.config_dir}'. Model variant: {cfg.model.variant}")

    # Device
    if args.device:
        device = torch.device(args.device)
    elif cfg.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Build model
    model = build_model(cfg)
    ckpt_path = args.checkpoint or getattr(cfg.deployment, "model_checkpoint", None)
    if ckpt_path and os.path.exists(ckpt_path):
        logger.info(f"Loading checkpoint weights from: {ckpt_path}")
        model = load_model_checkpoint(model, ckpt_path, device=device)
    else:
        if ckpt_path:
            logger.warning(f"Checkpoint file '{ckpt_path}' not found! Running with initial weights.")
        model.to(device)

    model.eval()

    # Evaluator instance
    evaluator = FaithfulnessEvaluator(
        model=model,
        cfg=cfg,
        device=device,
        class_names=CLASS_NAMES_8 if cfg.model.num_classes == 8 else None,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # MODE A: Single Image Evaluation
    # ──────────────────────────────────────────────────────────────────────────
    if args.image:
        logger.info(f"Running Single-Image Faithfulness Evaluation on: {args.image}")
        if not os.path.exists(args.image):
            logger.error(f"Image not found at path: {args.image}")
            sys.exit(1)

        mean_tuple = (float(cfg.data.mean[0]), float(cfg.data.mean[1]), float(cfg.data.mean[2]))
        std_tuple = (float(cfg.data.std[0]), float(cfg.data.std[1]), float(cfg.data.std[2]))

        img_tensor, orig_rgb = preprocess_image(
            image_input=args.image,
            img_size=cfg.model.img_size,
            mean=mean_tuple,
            std=std_tuple,
        )

        res = evaluator.evaluate_sample(
            image_tensor=img_tensor,
            original_rgb=orig_rgb,
            target_category=args.target_class,
            sample_id=os.path.basename(args.image),
        )

        # Print clean results table to stdout
        print("\n" + "=" * 65)
        print("  SINGLE-IMAGE FAITHFULNESS EVALUATION RESULTS")
        print("=" * 65)
        print(f"  Image Path:           {args.image}")
        print(f"  Predicted Class:      {res['predicted_class']} ({res['predicted_class_name']})")
        print(f"  Original Confidence:  {res['original_confidence']:.4f}")
        print("-" * 65)
        print(f"  Grad-CAM Deletion AUC:     {res['gradcam']['deletion_auc']:.4f}  (Lower = More Faithful)")
        print(f"  Grad-CAM Insertion AUC:    {res['gradcam']['insertion_auc']:.4f}  (Higher = More Faithful)")
        print("-" * 65)
        print(f"  Random Baseline Del AUC:   {res['random']['deletion_auc']:.4f}")
        print(f"  Random Baseline Ins AUC:   {res['random']['insertion_auc']:.4f}")
        if "least_important" in res:
            print("-" * 65)
            print(f"  Least-Imp Control Del AUC: {res['least_important']['deletion_auc']:.4f}")
            print(f"  Least-Imp Control Ins AUC: {res['least_important']['insertion_auc']:.4f}")
        print("=" * 65 + "\n")

        # Save single image report & plots
        report_file = os.path.join(reports_dir, f"faithfulness_{os.path.splitext(os.path.basename(args.image))[0]}.json")
        evaluator.save_report(res, report_file)
        saved_plots = evaluator.plot_curves(res, plots_dir, prefix=f"faithfulness_{os.path.splitext(os.path.basename(args.image))[0]}")

        if args.save_visualizations:
            vis_saved = evaluator.generate_step_visualizations(
                image_tensor=img_tensor,
                orig_rgb=orig_rgb,
                output_dir=vis_dir,
                prefix=f"vis_{os.path.splitext(os.path.basename(args.image))[0]}",
                target_category=args.target_class,
            )
            print(f"  Visualizations saved: {vis_saved}")

        print(f"  Report saved to: {report_file}")
        print(f"  Plots saved to:  {list(saved_plots.values())}\n")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # MODE B: Dataset / Folder Evaluation
    # ──────────────────────────────────────────────────────────────────────────
    if args.image_dir:
        logger.info(f"Evaluating images from directory: {args.image_dir}")
        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            img_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
            img_paths.extend(glob.glob(os.path.join(args.image_dir, "**", ext), recursive=True))

        if not img_paths:
            logger.error(f"No images found in {args.image_dir}")
            sys.exit(1)

        if args.max_images:
            img_paths = img_paths[: args.max_images]

        mean_tuple = (float(cfg.data.mean[0]), float(cfg.data.mean[1]), float(cfg.data.mean[2]))
        std_tuple = (float(cfg.data.std[0]), float(cfg.data.std[1]), float(cfg.data.std[2]))

        per_image_results = []
        pbar = tqdm(img_paths, desc="Processing Image Folder")
        for p in pbar:
            img_tensor, orig_rgb = preprocess_image(p, img_size=cfg.model.img_size, mean=mean_tuple, std=std_tuple)
            res = evaluator.evaluate_sample(image_tensor=img_tensor, sample_id=os.path.basename(p))
            per_image_results.append(res)

        # Aggregate
        del_gc = [r["gradcam"]["deletion_auc"] for r in per_image_results]
        ins_gc = [r["gradcam"]["insertion_auc"] for r in per_image_results]
        del_rnd = [r["random"]["deletion_auc"] for r in per_image_results]
        ins_rnd = [r["random"]["insertion_auc"] for r in per_image_results]

        summary = {
            "num_images_evaluated": len(per_image_results),
            "grid_size": evaluator.grid_size,
            "perturbation": evaluator.perturbation,
            "fractions": list(evaluator.fractions),
            "mean_original_confidence": float(np.mean([r["original_confidence"] for r in per_image_results])),
            "gradcam": {
                "mean_deletion_auc": float(np.mean(del_gc)),
                "std_deletion_auc": float(np.std(del_gc)),
                "mean_insertion_auc": float(np.mean(ins_gc)),
                "std_insertion_auc": float(np.std(ins_gc)),
                "mean_deletion_curve": np.mean([r["gradcam"]["deletion_curve"] for r in per_image_results], axis=0).tolist(),
                "mean_insertion_curve": np.mean([r["gradcam"]["insertion_curve"] for r in per_image_results], axis=0).tolist(),
            },
            "random_baseline": {
                "mean_deletion_auc": float(np.mean(del_rnd)),
                "std_deletion_auc": float(np.std(del_rnd)),
                "mean_insertion_auc": float(np.mean(ins_rnd)),
                "std_insertion_auc": float(np.std(ins_rnd)),
                "mean_deletion_curve": np.mean([r["random"]["deletion_curve"] for r in per_image_results], axis=0).tolist(),
                "mean_insertion_curve": np.mean([r["random"]["insertion_curve"] for r in per_image_results], axis=0).tolist(),
            },
        }
        dataset_results = {"summary": summary, "per_image": per_image_results}

    else:
        split_dir = os.path.join(cfg.paths.dataset_dir, args.split)
        if not os.path.exists(split_dir):
            logger.error(f"Dataset split directory '{split_dir}' does not exist.")
            sys.exit(1)

        val_transforms = build_val_test_transforms(cfg)
        dataset = GenericCapsuleDataset(split_dir, transform=val_transforms)
        loader = DataLoader(
            dataset,
            batch_size=min(cfg.training.batch_size, 32),
            shuffle=False,
            num_workers=cfg.training.num_workers,
        )

        logger.info(f"Evaluating {args.split} split dataset ({len(dataset)} total samples, max={args.max_images})")
        dataset_results = evaluator.evaluate_dataset(dataloader=loader, max_images=args.max_images)

    # Print summary table
    summ = dataset_results["summary"]
    print("\n" + "=" * 70)
    print("  FAITHFULNESS EVALUATION SUMMARY REPORT")
    print("=" * 70)
    print(f"  Evaluated Samples:          {summ['num_images_evaluated']}")
    print(f"  Spatial Grid Size:          {summ['grid_size']} x {summ['grid_size']} ({summ['grid_size']**2} regions)")
    print(f"  Perturbation Strategy:      {summ['perturbation']}")
    print(f"  Mean Original Confidence:   {summ['mean_original_confidence']:.4f}")
    print("-" * 70)
    print(f"  Grad-CAM Mean Deletion AUC:       {summ['gradcam']['mean_deletion_auc']:.4f} +/- {summ['gradcam']['std_deletion_auc']:.4f}")
    print(f"  Grad-CAM Mean Insertion AUC:      {summ['gradcam']['mean_insertion_auc']:.4f} +/- {summ['gradcam']['std_insertion_auc']:.4f}")
    print("-" * 70)
    print(f"  Random Baseline Mean Del AUC:     {summ['random_baseline']['mean_deletion_auc']:.4f} +/- {summ['random_baseline']['std_deletion_auc']:.4f}")
    print(f"  Random Baseline Mean Ins AUC:     {summ['random_baseline']['mean_insertion_auc']:.4f} +/- {summ['random_baseline']['std_insertion_auc']:.4f}")
    if "least_important_control" in summ:
        print("-" * 70)
        print(f"  Least-Imp Control Mean Del AUC:   {summ['least_important_control']['mean_deletion_auc']:.4f} +/- {summ['least_important_control']['std_deletion_auc']:.4f}")
        print(f"  Least-Imp Control Mean Ins AUC:   {summ['least_important_control']['mean_insertion_auc']:.4f} +/- {summ['least_important_control']['std_insertion_auc']:.4f}")
    print("=" * 70 + "\n")

    # Save JSON report and plots
    report_file = os.path.join(reports_dir, "faithfulness_report.json")
    evaluator.save_report(dataset_results, report_file)
    saved_plots = evaluator.plot_curves(dataset_results, plots_dir, prefix="faithfulness")

    print(f"Report saved to: {report_file}")
    print(f"Plots saved to:")
    for k, v in saved_plots.items():
        print(f"  - {k}: {v}")
    print()


if __name__ == "__main__":
    main()
