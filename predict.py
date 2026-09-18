"""predict.py — Uncertainty-aware prediction entry point for Trustworthy-Capsule-AI.

Performs deterministic or Monte Carlo (MC) Dropout inference on a target image,
providing:
    - Predicted class and confidence
    - Total uncertainty (Predictive Entropy)
    - Classification spread (Variation Ratio)
    - Per-class probabilities
    - Optional Class-Aware Prototype diagnostics (when using hybrid_class_aware)

Usage
-----
Run MC Dropout prediction (20 passes) on an image:
    python predict.py --image path/to/image.jpg --checkpoint outputs_kvasir/checkpoints/best_f1.pth --mc-samples 20

Deterministic prediction (1 pass):
    python predict.py --image path/to/image.jpg --checkpoint outputs_kvasir/checkpoints/best_f1.pth --mc-samples 0
"""

import argparse
import json
import os
import time
import torch
from PIL import Image

from src.config import load_config
from src.models.factory import build_model
from src.data.transforms import build_val_test_transforms
from src.inference.uncertainty import MCDropoutPredictor

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
        description="Trustworthy-Capsule-AI — Uncertainty-Aware Inference"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image file.")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Path to directory containing YAML config files (default: config).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pth).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Override model variant (e.g. hybrid_class_aware).",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=20,
        help="Number of MC Dropout passes (0 for deterministic).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device ('cuda' or 'cpu').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config_dir)
    if args.variant:
        cfg.model.variant = args.variant  # type: ignore

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    # Build model & load checkpoint
    model = build_model(cfg)
    
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
        print(f"Loading weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Notice: Checkpoint '{ckpt_path}' not found. Using initialized model weights.")

    model.to(device)

    # Prepare input image
    transforms = build_val_test_transforms(cfg)
    pil_img = Image.open(args.image).convert("RGB")
    tensor_img = transforms(pil_img).unsqueeze(0).to(device)

    # Uncertainty predictor
    predictor = MCDropoutPredictor(model)
    start_t = time.perf_counter()
    result = predictor.predict(tensor_img, n_samples=args.mc_samples, device=device)
    latency_ms = (time.perf_counter() - start_t) * 1000.0

    class_names = CLASS_NAMES_8[: cfg.model.num_classes]
    pred_label = class_names[result.predicted_class]

    print("\n" + "=" * 60)
    print("UNCERTAINTY-AWARE PREDICTION")
    print("=" * 60)
    print(f"Input Image        : {args.image}")
    print(f"Model Variant      : {cfg.model.variant}")
    print(f"MC Dropout Passes  : {result.n_samples}")
    print(f"Predicted Class    : {pred_label} (Index: {result.predicted_class})")
    print(f"Confidence         : {result.confidence * 100:.2f}%")
    print(f"Predictive Entropy : {result.predictive_entropy:.4f} (Total uncertainty)")
    print(f"Variation Ratio    : {result.variation_ratio:.4f} (Prediction spread)")
    print(f"Inference Latency  : {latency_ms:.2f} ms")
    print("\nClass Probabilities:")
    for name, p in zip(class_names, result.mean_proba):
        pct = float(p * 100.0)
        bar = "#" * int(pct / 4)
        print(f"  {name:22s}: {pct:6.2f}% | {bar}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
