"""Faithfulness evaluation module for Trustworthy-Capsule-AI.

This module provides quantitative perturbation-based evaluation of explainability
methods (e.g. Grad-CAM) via:
1. Deletion Test: Progressively masking high-importance regions with blurred replacements
   and measuring the drop in prediction confidence (Lower Deletion AUC = more faithful).
2. Insertion Test: Progressively restoring high-importance regions onto a blurred baseline
   and measuring the rise in prediction confidence (Higher Insertion AUC = more faithful).
3. Random Region Baseline: Comparing against randomly shuffled spatial region rankings.
4. Least-Important Control: Comparing against inverted Grad-CAM rankings.

All metrics are computed on a coarse spatial grid (default 14x14 = 196 regions for 224x224 input)
with blurred-region replacement to preserve overall medical image statistics while eliminating
local diagnostic cues.
"""

from __future__ import annotations

import os
import json
import logging
import random
from typing import Optional, Sequence, Union, Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.config import AppConfig
from src.explainability.gradcam import GradCAM
from src.explainability.utils import get_target_layer, preprocess_image, overlay_heatmap

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Core Pure-Function Utilities
# ──────────────────────────────────────────────────────────────────────────────

def compute_grid_importance(
    cam: Union[torch.Tensor, np.ndarray],
    grid_size: int = 14,
) -> np.ndarray:
    """Compute coarse spatial grid importance by averaging CAM values in each grid cell.

    Args:
        cam: 2D or 4D heatmap tensor/array, normalized in [0, 1].
             Shapes accepted: (H, W), (1, 1, H, W), or (1, H, W).
        grid_size: Number of subdivisions along each spatial dimension (e.g. 14 for 14x14).

    Returns:
        2D numpy array of shape ``(grid_size, grid_size)`` containing mean importance per cell.
    """
    if isinstance(cam, torch.Tensor):
        cam_np = cam.detach().cpu().squeeze().numpy()
    else:
        cam_np = np.squeeze(cam)

    if cam_np.ndim != 2:
        raise ValueError(f"Expected 2D CAM array after squeezing, got shape {cam_np.shape}")

    H, W = cam_np.shape
    h_step = H / grid_size
    w_step = W / grid_size

    importance_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    for r in range(grid_size):
        r_start = int(round(r * h_step))
        r_end = int(round((r + 1) * h_step)) if r < grid_size - 1 else H

        for c in range(grid_size):
            c_start = int(round(c * w_step))
            c_end = int(round((c + 1) * w_step)) if c < grid_size - 1 else W

            cell_vals = cam_np[r_start:r_end, c_start:c_end]
            if cell_vals.size > 0:
                importance_grid[r, c] = float(np.mean(cell_vals))
            else:
                importance_grid[r, c] = 0.0

    return importance_grid


def rank_regions(
    importance_grid: Union[torch.Tensor, np.ndarray],
    order: str = "descending",
    seed: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Rank grid regions according to importance or random permutation.

    Args:
        importance_grid: Array/tensor of shape ``(grid_size, grid_size)``.
        order: Ranking strategy:
               - ``'descending'``: Most important first (Grad-CAM default).
               - ``'ascending'``: Least important first (control).
               - ``'random'``: Random permutation.
        seed: Random seed for reproducible random shuffling.

    Returns:
        List of ``(r, c)`` coordinate tuples representing the ordered sequence of regions.
    """
    if isinstance(importance_grid, torch.Tensor):
        grid_np = importance_grid.detach().cpu().numpy()
    else:
        grid_np = np.asarray(importance_grid)

    grid_size_r, grid_size_c = grid_np.shape
    coords: List[Tuple[int, int]] = [
        (r, c) for r in range(grid_size_r) for c in range(grid_size_c)
    ]

    if order == "descending":
        coords.sort(key=lambda coord: float(grid_np[coord[0], coord[1]]), reverse=True)
    elif order == "ascending":
        coords.sort(key=lambda coord: float(grid_np[coord[0], coord[1]]), reverse=False)
    elif order == "random":
        rng = random.Random(seed)
        rng.shuffle(coords)
    else:
        raise ValueError(f"Unknown ordering strategy: {order}. Choose 'descending', 'ascending', or 'random'.")

    return coords


def create_perturbed_baseline(
    image_tensor: torch.Tensor,
    perturbation: str = "blur",
    blur_kernel_size: int = 21,
    blur_sigma: float = 10.0,
) -> torch.Tensor:
    """Generate the reference perturbed image used for spatial masking.

    Args:
        image_tensor: Normalized image tensor ``(B, C, H, W)`` or ``(C, H, W)``.
        perturbation: Method: ``'blur'`` (Gaussian blur), ``'mean'`` (channel mean), or ``'zero'`` (black).
        blur_kernel_size: Kernel size for Gaussian blur (must be odd, default: 21).
        blur_sigma: Gaussian standard deviation (default: 10.0).

    Returns:
        Perturbed baseline tensor with identical shape, dtype, and device.
    """
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Ensure kernel size is odd
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    if perturbation == "blur":
        # PyTorch native Gaussian blur operates directly on tensor
        return TF.gaussian_blur(
            image_tensor,
            kernel_size=[blur_kernel_size, blur_kernel_size],
            sigma=[blur_sigma, blur_sigma],
        )
    elif perturbation == "mean":
        mean_val = image_tensor.mean(dim=(2, 3), keepdim=True)
        return mean_val.expand_as(image_tensor)
    elif perturbation == "zero":
        return torch.zeros_like(image_tensor)
    else:
        raise ValueError(f"Unsupported perturbation method: {perturbation}. Choose 'blur', 'mean', or 'zero'.")


def apply_region_mask(
    original_tensor: torch.Tensor,
    baseline_tensor: torch.Tensor,
    active_regions: List[Tuple[int, int]],
    grid_size: int = 14,
    mode: str = "deletion",
) -> torch.Tensor:
    """Compose an image by perturbing or restoring selected spatial regions.

    Args:
        original_tensor: Original unperturbed image tensor ``(1, C, H, W)``.
        baseline_tensor: Fully perturbed baseline image tensor ``(1, C, H, W)``.
        active_regions: List of ``(r, c)`` coordinates currently selected.
        grid_size: Grid spatial resolution (default: 14).
        mode:
            - ``'deletion'``: active regions are replaced with baseline (perturbed).
            - ``'insertion'``: active regions are replaced with original (restored).

    Returns:
        Composite image tensor of shape ``(1, C, H, W)``.
    """
    if original_tensor.ndim == 3:
        original_tensor = original_tensor.unsqueeze(0)
    if baseline_tensor.ndim == 3:
        baseline_tensor = baseline_tensor.unsqueeze(0)

    _, _, H, W = original_tensor.shape
    h_step = H / grid_size
    w_step = W / grid_size

    if mode == "deletion":
        # Start from original, overwrite active_regions with baseline
        composite = original_tensor.clone()
        for r, c in active_regions:
            r_start = int(round(r * h_step))
            r_end = int(round((r + 1) * h_step)) if r < grid_size - 1 else H
            c_start = int(round(c * w_step))
            c_end = int(round((c + 1) * w_step)) if c < grid_size - 1 else W
            composite[:, :, r_start:r_end, c_start:c_end] = baseline_tensor[:, :, r_start:r_end, c_start:c_end]
    elif mode == "insertion":
        # Start from baseline, overwrite active_regions with original
        composite = baseline_tensor.clone()
        for r, c in active_regions:
            r_start = int(round(r * h_step))
            r_end = int(round((r + 1) * h_step)) if r < grid_size - 1 else H
            c_start = int(round(c * w_step))
            c_end = int(round((c + 1) * w_step)) if c < grid_size - 1 else W
            composite[:, :, r_start:r_end, c_start:c_end] = original_tensor[:, :, r_start:r_end, c_start:c_end]
    else:
        raise ValueError(f"Unknown masking mode: {mode}. Choose 'deletion' or 'insertion'.")

    return composite


def calculate_auc(fractions: Sequence[float], confidences: Sequence[float]) -> float:
    """Compute Area Under the Curve (AUC) using the composite trapezoidal rule.

    Formula:
        AUC = sum_{i=0}^{N-2} ( (c_i + c_{i+1}) / 2 ) * (f_{i+1} - f_i)

    Args:
        fractions: Sequence of fraction values strictly increasing in [0, 1].
        confidences: Sequence of model predicted-class probabilities at each fraction.

    Returns:
        AUC value as float in [0, 1].
    """
    if len(fractions) != len(confidences):
        raise ValueError("fractions and confidences must have identical lengths.")
    if len(fractions) < 2:
        return float(confidences[0]) if confidences else 0.0

    x = np.asarray(fractions, dtype=np.float64)
    y = np.asarray(confidences, dtype=np.float64)

    # Trapezoidal rule integration
    auc = float(np.sum((y[:-1] + y[1:]) * 0.5 * (x[1:] - x[:-1])))
    return float(np.clip(auc, 0.0, 1.0))


def evaluate_deletion_curve(
    model: nn.Module,
    image_tensor: torch.Tensor,
    ranked_regions: List[Tuple[int, int]],
    target_class: int,
    fractions: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    perturbation: str = "blur",
    grid_size: int = 14,
    device: Optional[Union[str, torch.device]] = None,
    blur_kernel_size: int = 21,
    blur_sigma: float = 10.0,
) -> Dict[str, Any]:
    """Execute Deletion Test on an image with ordered region ranking.

    Progressively masks the highest-ranked regions and records target class confidence.
    Lower Deletion AUC indicates higher faithfulness.

    Args:
        model: PyTorch model in eval mode.
        image_tensor: Normalized input tensor ``(1, 3, H, W)``.
        ranked_regions: Ordered list of ``(r, c)`` regions.
        target_class: Target class index to evaluate confidence for.
        fractions: Perturbation fraction steps from 0.0 to 1.0.
        perturbation: Perturbation strategy ('blur', 'mean', 'zero').
        grid_size: Grid spatial resolution.
        device: Torch device.
        blur_kernel_size: Gaussian kernel size.
        blur_sigma: Gaussian sigma.

    Returns:
        Dict with keys: ``'fractions'``, ``'confidences'``, ``'auc'``, ``'strategy'``.
    """
    if device is None:
        device = next(model.parameters()).device
    dev = torch.device(device)

    image_tensor = image_tensor.to(dev)
    baseline_tensor = create_perturbed_baseline(
        image_tensor,
        perturbation=perturbation,
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
    ).to(dev)

    total_regions = len(ranked_regions)
    confidences: List[float] = []

    model.eval()
    with torch.no_grad():
        for frac in fractions:
            num_perturbed = int(round(frac * total_regions))
            active = ranked_regions[:num_perturbed]
            masked_img = apply_region_mask(
                original_tensor=image_tensor,
                baseline_tensor=baseline_tensor,
                active_regions=active,
                grid_size=grid_size,
                mode="deletion",
            )

            logits = model(masked_img)
            probs = F.softmax(logits, dim=1)
            conf = float(probs[0, target_class].item())
            confidences.append(conf)

    auc = calculate_auc(fractions, confidences)
    return {
        "fractions": list(fractions),
        "confidences": confidences,
        "auc": auc,
        "strategy": "deletion",
    }


def evaluate_insertion_curve(
    model: nn.Module,
    image_tensor: torch.Tensor,
    ranked_regions: List[Tuple[int, int]],
    target_class: int,
    fractions: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    perturbation: str = "blur",
    grid_size: int = 14,
    device: Optional[Union[str, torch.device]] = None,
    blur_kernel_size: int = 21,
    blur_sigma: float = 10.0,
) -> Dict[str, Any]:
    """Execute Insertion Test on an image with ordered region ranking.

    Progressively restores the highest-ranked regions onto a blurred baseline.
    Higher Insertion AUC indicates higher faithfulness.

    Args:
        model: PyTorch model in eval mode.
        image_tensor: Normalized input tensor ``(1, 3, H, W)``.
        ranked_regions: Ordered list of ``(r, c)`` regions.
        target_class: Target class index to evaluate confidence for.
        fractions: Insertion fraction steps from 0.0 to 1.0.
        perturbation: Perturbation strategy ('blur', 'mean', 'zero').
        grid_size: Grid spatial resolution.
        device: Torch device.
        blur_kernel_size: Gaussian kernel size.
        blur_sigma: Gaussian sigma.

    Returns:
        Dict with keys: ``'fractions'``, ``'confidences'``, ``'auc'``, ``'strategy'``.
    """
    if device is None:
        device = next(model.parameters()).device
    dev = torch.device(device)

    image_tensor = image_tensor.to(dev)
    baseline_tensor = create_perturbed_baseline(
        image_tensor,
        perturbation=perturbation,
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
    ).to(dev)

    total_regions = len(ranked_regions)
    confidences: List[float] = []

    model.eval()
    with torch.no_grad():
        for frac in fractions:
            num_restored = int(round(frac * total_regions))
            active = ranked_regions[:num_restored]
            masked_img = apply_region_mask(
                original_tensor=image_tensor,
                baseline_tensor=baseline_tensor,
                active_regions=active,
                grid_size=grid_size,
                mode="insertion",
            )

            logits = model(masked_img)
            probs = F.softmax(logits, dim=1)
            conf = float(probs[0, target_class].item())
            confidences.append(conf)

    auc = calculate_auc(fractions, confidences)
    return {
        "fractions": list(fractions),
        "confidences": confidences,
        "auc": auc,
        "strategy": "insertion",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Faithfulness Evaluator Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class FaithfulnessEvaluator:
    """Comprehensive faithfulness evaluation pipeline.

    Computes Grad-CAM, Random baseline, and Least-Important control deletion &
    insertion curves and AUC metrics on single images or entire dataset splits.

    Args:
        model: Trained PyTorch model instance.
        cfg: Validated :class:`~src.config.AppConfig`.
        target_layer: Optional target layer override for Grad-CAM.
        device: Device to execute model on.
        class_names: Optional list of class names.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: AppConfig,
        target_layer: Optional[nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.cfg = cfg
        self.model = model

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

        if class_names:
            self.class_names = class_names
        else:
            self.class_names = [f"Class_{i}" for i in range(cfg.model.num_classes)]

        # Resolve target layer for GradCAM
        target_layer_path = getattr(cfg.explainability, "target_layer", None)
        resolved_layer = target_layer or get_target_layer(self.model, target_layer_path)
        self.explainer = GradCAM(model=self.model, target_layer=resolved_layer)

        # Settings from cfg.faithfulness
        self.grid_size = getattr(cfg.faithfulness, "grid_size", 14)
        self.perturbation = getattr(cfg.faithfulness, "perturbation", "blur")
        self.blur_kernel_size = getattr(cfg.faithfulness, "blur_kernel_size", 21)
        self.blur_sigma = getattr(cfg.faithfulness, "blur_sigma", 10.0)
        self.fractions = getattr(
            cfg.faithfulness,
            "fractions",
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        self.random_seeds = getattr(cfg.faithfulness, "random_seeds", [42])
        self.eval_least_important = getattr(cfg.faithfulness, "eval_least_important", True)

    def evaluate_sample(
        self,
        image_tensor: torch.Tensor,
        original_rgb: Optional[np.ndarray] = None,
        target_category: Optional[int] = None,
        ground_truth: Optional[int] = None,
        sample_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate faithfulness metrics on a single input image.

        Args:
            image_tensor: Preprocessed tensor ``(1, 3, H, W)`` or ``(3, H, W)``.
            original_rgb: Optional RGB numpy array ``(H, W, 3)`` in [0, 255] for plotting.
            target_category: Class index to evaluate. If None, uses model's predicted class.
            ground_truth: Optional ground truth class index for logging.
            sample_id: Identifier or file name for the sample.

        Returns:
            Structured dictionary with all metrics, curves, and metadata.
        """
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        # 1. Forward pass for base prediction
        self.model.eval()
        with torch.no_grad():
            orig_logits = self.model(image_tensor)
            orig_probs = F.softmax(orig_logits, dim=1)
            pred_class = int(torch.argmax(orig_probs, dim=1).item())
            orig_pred_conf = float(orig_probs[0, pred_class].item())

        target_class = target_category if target_category is not None else pred_class
        target_conf = float(orig_probs[0, target_class].item())

        # 2. Compute Grad-CAM heatmap
        cam = self.explainer.attribute(image_tensor, target_category=target_class)  # (1, 1, H, W)

        # 3. Calculate spatial grid importance
        importance_grid = compute_grid_importance(cam, grid_size=self.grid_size)

        # 4. Grad-CAM (Most Important First) evaluation
        ranked_gradcam = rank_regions(importance_grid, order="descending")
        del_gradcam = evaluate_deletion_curve(
            model=self.model,
            image_tensor=image_tensor,
            ranked_regions=ranked_gradcam,
            target_class=target_class,
            fractions=self.fractions,
            perturbation=self.perturbation,
            grid_size=self.grid_size,
            device=self.device,
            blur_kernel_size=self.blur_kernel_size,
            blur_sigma=self.blur_sigma,
        )
        ins_gradcam = evaluate_insertion_curve(
            model=self.model,
            image_tensor=image_tensor,
            ranked_regions=ranked_gradcam,
            target_class=target_class,
            fractions=self.fractions,
            perturbation=self.perturbation,
            grid_size=self.grid_size,
            device=self.device,
            blur_kernel_size=self.blur_kernel_size,
            blur_sigma=self.blur_sigma,
        )

        # 5. Random Baseline evaluation (averaged over seeds)
        del_rand_aucs: List[float] = []
        ins_rand_aucs: List[float] = []
        del_rand_curves: List[List[float]] = []
        ins_rand_curves: List[List[float]] = []

        for s in self.random_seeds:
            ranked_rand = rank_regions(importance_grid, order="random", seed=s)
            del_r = evaluate_deletion_curve(
                model=self.model,
                image_tensor=image_tensor,
                ranked_regions=ranked_rand,
                target_class=target_class,
                fractions=self.fractions,
                perturbation=self.perturbation,
                grid_size=self.grid_size,
                device=self.device,
                blur_kernel_size=self.blur_kernel_size,
                blur_sigma=self.blur_sigma,
            )
            ins_r = evaluate_insertion_curve(
                model=self.model,
                image_tensor=image_tensor,
                ranked_regions=ranked_rand,
                target_class=target_class,
                fractions=self.fractions,
                perturbation=self.perturbation,
                grid_size=self.grid_size,
                device=self.device,
                blur_kernel_size=self.blur_kernel_size,
                blur_sigma=self.blur_sigma,
            )
            del_rand_aucs.append(del_r["auc"])
            ins_rand_aucs.append(ins_r["auc"])
            del_rand_curves.append(del_r["confidences"])
            ins_rand_curves.append(ins_r["confidences"])

        mean_del_rand_conf = np.mean(del_rand_curves, axis=0).tolist()
        mean_ins_rand_conf = np.mean(ins_rand_curves, axis=0).tolist()
        mean_del_rand_auc = float(np.mean(del_rand_aucs))
        mean_ins_rand_auc = float(np.mean(ins_rand_aucs))

        # 6. Least Important First Control (optional)
        del_least_res: Optional[Dict[str, Any]] = None
        ins_least_res: Optional[Dict[str, Any]] = None
        if self.eval_least_important:
            ranked_least = rank_regions(importance_grid, order="ascending")
            del_least_res = evaluate_deletion_curve(
                model=self.model,
                image_tensor=image_tensor,
                ranked_regions=ranked_least,
                target_class=target_class,
                fractions=self.fractions,
                perturbation=self.perturbation,
                grid_size=self.grid_size,
                device=self.device,
                blur_kernel_size=self.blur_kernel_size,
                blur_sigma=self.blur_sigma,
            )
            ins_least_res = evaluate_insertion_curve(
                model=self.model,
                image_tensor=image_tensor,
                ranked_regions=ranked_least,
                target_class=target_class,
                fractions=self.fractions,
                perturbation=self.perturbation,
                grid_size=self.grid_size,
                device=self.device,
                blur_kernel_size=self.blur_kernel_size,
                blur_sigma=self.blur_sigma,
            )

        result: Dict[str, Any] = {
            "sample_id": sample_id or "sample",
            "ground_truth": ground_truth,
            "ground_truth_name": self.class_names[ground_truth] if ground_truth is not None and ground_truth < len(self.class_names) else None,
            "predicted_class": pred_class,
            "predicted_class_name": self.class_names[pred_class] if pred_class < len(self.class_names) else f"Class_{pred_class}",
            "original_confidence": orig_pred_conf,
            "target_class": target_class,
            "target_class_name": self.class_names[target_class] if target_class < len(self.class_names) else f"Class_{target_class}",
            "target_confidence": target_conf,
            "gradcam": {
                "deletion_auc": del_gradcam["auc"],
                "insertion_auc": ins_gradcam["auc"],
                "deletion_curve": del_gradcam["confidences"],
                "insertion_curve": ins_gradcam["confidences"],
            },
            "random": {
                "deletion_auc": mean_del_rand_auc,
                "insertion_auc": mean_ins_rand_auc,
                "deletion_curve": mean_del_rand_conf,
                "insertion_curve": mean_ins_rand_conf,
            },
            "fractions": list(self.fractions),
            "grid_size": self.grid_size,
            "perturbation": self.perturbation,
        }

        if del_least_res and ins_least_res:
            result["least_important"] = {
                "deletion_auc": del_least_res["auc"],
                "insertion_auc": ins_least_res["auc"],
                "deletion_curve": del_least_res["confidences"],
                "insertion_curve": ins_least_res["confidences"],
            }

        return result

    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        max_images: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate faithfulness across a dataset split with progress bar.

        Args:
            dataloader: DataLoader yielding ``(images, labels)`` batches.
            max_images: Maximum number of images to evaluate (e.g. 10, 50, 100).

        Returns:
            Aggregated dictionary with summary metrics and per-image results.
        """
        per_image_results: List[Dict[str, Any]] = []
        count = 0
        total_to_process = max_images or len(dataloader.dataset)  # type: ignore

        pbar = tqdm(total=total_to_process, desc="Evaluating Faithfulness")

        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                images = batch["image"]
                labels = batch["label"]
            else:
                images, labels = batch
            batch_size = images.shape[0]
            for i in range(batch_size):
                if max_images is not None and count >= max_images:
                    break

                img_tensor = images[i : i + 1]
                gt_label = int(labels[i].item()) if isinstance(labels[i], torch.Tensor) else int(labels[i])
                sample_id = f"sample_{count:05d}"

                res = self.evaluate_sample(
                    image_tensor=img_tensor,
                    target_category=None,  # defaults to model's predicted class
                    ground_truth=gt_label,
                    sample_id=sample_id,
                )
                per_image_results.append(res)
                count += 1
                pbar.update(1)

            if max_images is not None and count >= max_images:
                break

        pbar.close()

        # Compute summary aggregations
        del_gradcam_aucs = [r["gradcam"]["deletion_auc"] for r in per_image_results]
        ins_gradcam_aucs = [r["gradcam"]["insertion_auc"] for r in per_image_results]
        del_rand_aucs = [r["random"]["deletion_auc"] for r in per_image_results]
        ins_rand_aucs = [r["random"]["insertion_auc"] for r in per_image_results]

        mean_del_gc_curve = np.mean([r["gradcam"]["deletion_curve"] for r in per_image_results], axis=0).tolist()
        mean_ins_gc_curve = np.mean([r["gradcam"]["insertion_curve"] for r in per_image_results], axis=0).tolist()
        mean_del_rand_curve = np.mean([r["random"]["deletion_curve"] for r in per_image_results], axis=0).tolist()
        mean_ins_rand_curve = np.mean([r["random"]["insertion_curve"] for r in per_image_results], axis=0).tolist()

        summary: Dict[str, Any] = {
            "num_images_evaluated": len(per_image_results),
            "grid_size": self.grid_size,
            "perturbation": self.perturbation,
            "fractions": list(self.fractions),
            "mean_original_confidence": float(np.mean([r["original_confidence"] for r in per_image_results])),
            "gradcam": {
                "mean_deletion_auc": float(np.mean(del_gradcam_aucs)),
                "std_deletion_auc": float(np.std(del_gradcam_aucs)),
                "mean_insertion_auc": float(np.mean(ins_gradcam_aucs)),
                "std_insertion_auc": float(np.std(ins_gradcam_aucs)),
                "mean_deletion_curve": mean_del_gc_curve,
                "mean_insertion_curve": mean_ins_gc_curve,
            },
            "random_baseline": {
                "mean_deletion_auc": float(np.mean(del_rand_aucs)),
                "std_deletion_auc": float(np.std(del_rand_aucs)),
                "mean_insertion_auc": float(np.mean(ins_rand_aucs)),
                "std_insertion_auc": float(np.std(ins_rand_aucs)),
                "mean_deletion_curve": mean_del_rand_curve,
                "mean_insertion_curve": mean_ins_rand_curve,
            },
        }

        if self.eval_least_important and all("least_important" in r for r in per_image_results):
            del_least_aucs = [r["least_important"]["deletion_auc"] for r in per_image_results]
            ins_least_aucs = [r["least_important"]["insertion_auc"] for r in per_image_results]
            mean_del_least_curve = np.mean([r["least_important"]["deletion_curve"] for r in per_image_results], axis=0).tolist()
            mean_ins_least_curve = np.mean([r["least_important"]["insertion_curve"] for r in per_image_results], axis=0).tolist()

            summary["least_important_control"] = {
                "mean_deletion_auc": float(np.mean(del_least_aucs)),
                "std_deletion_auc": float(np.std(del_least_aucs)),
                "mean_insertion_auc": float(np.mean(ins_least_aucs)),
                "std_insertion_auc": float(np.std(ins_least_aucs)),
                "mean_deletion_curve": mean_del_least_curve,
                "mean_insertion_curve": mean_ins_least_curve,
            }

        # Class distribution of predictions
        pred_dist: Dict[str, int] = {}
        for r in per_image_results:
            name = r["predicted_class_name"]
            pred_dist[name] = pred_dist.get(name, 0) + 1
        summary["predicted_class_distribution"] = pred_dist

        return {
            "summary": summary,
            "per_image": per_image_results,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Plotting & Reporting
    # ──────────────────────────────────────────────────────────────────────────

    def plot_curves(
        self,
        results: Dict[str, Any],
        output_dir: str,
        prefix: str = "faithfulness",
    ) -> Dict[str, str]:
        """Generate and save publication-quality Deletion, Insertion, and Combined plots.

        Args:
            results: Results dictionary returned by evaluate_dataset or evaluate_sample.
            output_dir: Directory path to save PNG plots.
            prefix: Filename prefix.

        Returns:
            Dict mapping plot name to saved file path.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        is_dataset = "summary" in results
        fractions = results["summary"]["fractions"] if is_dataset else results["fractions"]

        gc_del = results["summary"]["gradcam"]["mean_deletion_curve"] if is_dataset else results["gradcam"]["deletion_curve"]
        gc_del_auc = results["summary"]["gradcam"]["mean_deletion_auc"] if is_dataset else results["gradcam"]["deletion_auc"]
        rnd_del = results["summary"]["random_baseline"]["mean_deletion_curve"] if is_dataset else results["random"]["deletion_curve"]
        rnd_del_auc = results["summary"]["random_baseline"]["mean_deletion_auc"] if is_dataset else results["random"]["deletion_auc"]

        gc_ins = results["summary"]["gradcam"]["mean_insertion_curve"] if is_dataset else results["gradcam"]["insertion_curve"]
        gc_ins_auc = results["summary"]["gradcam"]["mean_insertion_auc"] if is_dataset else results["gradcam"]["insertion_auc"]
        rnd_ins = results["summary"]["random_baseline"]["mean_insertion_curve"] if is_dataset else results["random"]["insertion_curve"]
        rnd_ins_auc = results["summary"]["random_baseline"]["mean_insertion_auc"] if is_dataset else results["random"]["insertion_auc"]

        has_least = (
            ("least_important_control" in results["summary"]) if is_dataset else ("least_important" in results)
        )
        if has_least:
            least_del = results["summary"]["least_important_control"]["mean_deletion_curve"] if is_dataset else results["least_important"]["deletion_curve"]
            least_del_auc = results["summary"]["least_important_control"]["mean_deletion_auc"] if is_dataset else results["least_important"]["deletion_auc"]
            least_ins = results["summary"]["least_important_control"]["mean_insertion_curve"] if is_dataset else results["least_important"]["insertion_curve"]
            least_ins_auc = results["summary"]["least_important_control"]["mean_insertion_auc"] if is_dataset else results["least_important"]["insertion_auc"]

        # ── 1. Deletion Plot ──────────────────────────────────────────────────
        plt.figure(figsize=(7, 5), dpi=300)
        plt.plot(fractions, gc_del, "o-", color="#d9534f", linewidth=2.2, label=f"Grad-CAM (AUC = {gc_del_auc:.4f})")
        plt.plot(fractions, rnd_del, "s--", color="#6c757d", linewidth=1.8, label=f"Random Baseline (AUC = {rnd_del_auc:.4f})")
        if has_least:
            plt.plot(fractions, least_del, "^-.", color="#0275d8", linewidth=1.8, label=f"Least-Important (AUC = {least_del_auc:.4f})")

        plt.xlabel("Fraction of Spatial Regions Removed", fontsize=12, fontweight="bold")
        plt.ylabel("Prediction Confidence", fontsize=12, fontweight="bold")
        plt.title("Deletion Test (Lower AUC = Better Faithfulness)", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.ylim(-0.02, 1.05)
        plt.xlim(0.0, 1.0)
        plt.legend(frameon=True, fontsize=10, loc="upper right")
        plt.tight_layout()

        del_path = os.path.join(output_dir, f"{prefix}_deletion.png")
        plt.savefig(del_path)
        plt.close()
        paths["deletion_plot"] = del_path

        # ── 2. Insertion Plot ─────────────────────────────────────────────────
        plt.figure(figsize=(7, 5), dpi=300)
        plt.plot(fractions, gc_ins, "o-", color="#28a745", linewidth=2.2, label=f"Grad-CAM (AUC = {gc_ins_auc:.4f})")
        plt.plot(fractions, rnd_ins, "s--", color="#6c757d", linewidth=1.8, label=f"Random Baseline (AUC = {rnd_ins_auc:.4f})")
        if has_least:
            plt.plot(fractions, least_ins, "^-.", color="#0275d8", linewidth=1.8, label=f"Least-Important (AUC = {least_ins_auc:.4f})")

        plt.xlabel("Fraction of Spatial Regions Inserted", fontsize=12, fontweight="bold")
        plt.ylabel("Prediction Confidence", fontsize=12, fontweight="bold")
        plt.title("Insertion Test (Higher AUC = Better Faithfulness)", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.ylim(-0.02, 1.05)
        plt.xlim(0.0, 1.0)
        plt.legend(frameon=True, fontsize=10, loc="lower right")
        plt.tight_layout()

        ins_path = os.path.join(output_dir, f"{prefix}_insertion.png")
        plt.savefig(ins_path)
        plt.close()
        paths["insertion_plot"] = ins_path

        # ── 3. Combined Comparison Plot ───────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        # Left: Deletion
        ax1.plot(fractions, gc_del, "o-", color="#d9534f", linewidth=2.2, label=f"Grad-CAM (AUC: {gc_del_auc:.4f})")
        ax1.plot(fractions, rnd_del, "s--", color="#6c757d", linewidth=1.8, label=f"Random (AUC: {rnd_del_auc:.4f})")
        if has_least:
            ax1.plot(fractions, least_del, "^-.", color="#0275d8", linewidth=1.8, label=f"Least-Imp (AUC: {least_del_auc:.4f})")
        ax1.set_xlabel("Fraction Removed", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Confidence", fontsize=11, fontweight="bold")
        ax1.set_title("Deletion Test (Lower = Better)", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.set_ylim(-0.02, 1.05)
        ax1.legend(frameon=True, fontsize=9.5)

        # Right: Insertion
        ax2.plot(fractions, gc_ins, "o-", color="#28a745", linewidth=2.2, label=f"Grad-CAM (AUC: {gc_ins_auc:.4f})")
        ax2.plot(fractions, rnd_ins, "s--", color="#6c757d", linewidth=1.8, label=f"Random (AUC: {rnd_ins_auc:.4f})")
        if has_least:
            ax2.plot(fractions, least_ins, "^-.", color="#0275d8", linewidth=1.8, label=f"Least-Imp (AUC: {least_ins_auc:.4f})")
        ax2.set_xlabel("Fraction Inserted", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Confidence", fontsize=11, fontweight="bold")
        ax2.set_title("Insertion Test (Higher = Better)", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.set_ylim(-0.02, 1.05)
        ax2.legend(frameon=True, fontsize=9.5)

        fig.suptitle("Quantitative Faithfulness Evaluation — Trustworthy-Capsule-AI", fontsize=14, fontweight="bold")
        plt.tight_layout()

        comp_path = os.path.join(output_dir, f"{prefix}_comparison.png")
        plt.savefig(comp_path)
        plt.close()
        paths["comparison_plot"] = comp_path

        return paths

    def save_report(self, results: Dict[str, Any], output_path: str) -> None:
        """Save results to JSON format.

        Args:
            results: Results dictionary.
            output_path: File path to save JSON.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved faithfulness report to: {output_path}")

    def generate_step_visualizations(
        self,
        image_tensor: torch.Tensor,
        orig_rgb: np.ndarray,
        output_dir: str,
        prefix: str = "sample",
        target_category: Optional[int] = None,
    ) -> List[str]:
        """Save visual progression montage for deletion and insertion.

        Args:
            image_tensor: Normalized image tensor ``(1, 3, H, W)``.
            orig_rgb: Original RGB image array ``(H, W, 3)`` uint8.
            output_dir: Output directory.
            prefix: Filename prefix.
            target_category: Class index.

        Returns:
            List of saved image paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved: List[str] = []

        image_tensor = image_tensor.to(self.device)
        cam = self.explainer.attribute(image_tensor, target_category=target_category)
        importance_grid = compute_grid_importance(cam, grid_size=self.grid_size)
        ranked = rank_regions(importance_grid, order="descending")

        baseline = create_perturbed_baseline(
            image_tensor,
            perturbation=self.perturbation,
            blur_kernel_size=self.blur_kernel_size,
            blur_sigma=self.blur_sigma,
        ).to(self.device)

        # 1. Save Original + Grad-CAM overlay
        overlay, _ = overlay_heatmap(cam, orig_rgb, alpha=self.cfg.explainability.alpha)
        cam_vis_path = os.path.join(output_dir, f"{prefix}_gradcam_overlay.png")
        Image.fromarray(overlay).save(cam_vis_path)
        saved.append(cam_vis_path)

        # 2. Progression montage (0%, 25%, 50%, 75%, 100%)
        demo_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
        fig, axes = plt.subplots(2, len(demo_fractions), figsize=(15, 6), dpi=200)

        # Un-normalize function for visualization
        mean = np.array(self.cfg.data.mean, dtype=np.float32)
        std = np.array(self.cfg.data.std, dtype=np.float32)

        def to_rgb_img(t: torch.Tensor) -> np.ndarray:
            arr = t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            arr = (arr * std + mean) * 255.0
            return np.clip(arr, 0, 255).astype(np.uint8)

        total_r = len(ranked)
        for idx, frac in enumerate(demo_fractions):
            k = int(round(frac * total_r))
            active = ranked[:k]

            # Deletion step
            del_img = apply_region_mask(image_tensor, baseline, active, grid_size=self.grid_size, mode="deletion")
            axes[0, idx].imshow(to_rgb_img(del_img))
            axes[0, idx].set_title(f"Deletion {int(frac*100)}%")
            axes[0, idx].axis("off")

            # Insertion step
            ins_img = apply_region_mask(image_tensor, baseline, active, grid_size=self.grid_size, mode="insertion")
            axes[1, idx].imshow(to_rgb_img(ins_img))
            axes[1, idx].set_title(f"Insertion {int(frac*100)}%")
            axes[1, idx].axis("off")

        plt.suptitle(f"Spatial Perturbation Progression ({self.perturbation})", fontsize=14, fontweight="bold")
        plt.tight_layout()

        montage_path = os.path.join(output_dir, f"{prefix}_progression_montage.png")
        plt.savefig(montage_path)
        plt.close()
        saved.append(montage_path)

        return saved
