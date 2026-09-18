"""classify_spot.py — On-the-spot image classification script for Trustworthy-Capsule-AI.

Loads a trained model checkpoint and classifies target image(s), printing
probabilities, predicted class, ground-truth class, and inference latency.
"""

import time
import argparse
from typing import Literal, cast
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import load_config
from src.models.factory import build_model
from src.data.transforms import build_val_test_transforms

CLASS_NAMES = [
    "Angiectasia",
    "Erosion",
    "Ileocecal valve",
    "Lymphangiectasia",
    "Normal clean mucosa",
    "Pylorus",
    "Reduced mucosal view",
    "Ulcer",
]

ModelVariant = Literal[
    "efficientnet_only",
    "vit_only",
    "hybrid_concat",
    "hybrid_adaptive",
    "hybrid_class_aware",
]


def load_model_from_checkpoint(variant: str, checkpoint_path: str, device: torch.device, config_dir: str = "config"):
    cfg = load_config(config_dir)
    cfg.model.variant = cast(ModelVariant, variant)
    
    model = build_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model, cfg


def classify_image(image_path: str, model, cfg, device: torch.device):
    transforms = build_val_test_transforms(cfg)
    pil_img = Image.open(image_path).convert("RGB")
    tensor_img = transforms(pil_img).unsqueeze(0).to(device)
    
    start_t = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor_img)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    latency_ms = (time.perf_counter() - start_t) * 1000.0
    
    pred_idx = probs.argmax()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100.0
    
    return {
        "image_path": image_path,
        "pred_class": pred_class,
        "pred_idx": int(pred_idx),
        "confidence": confidence,
        "probabilities": {name: float(p * 100.0) for name, p in zip(CLASS_NAMES, probs)},
        "latency_ms": latency_ms
    }


def main():
    parser = argparse.ArgumentParser(description="On-the-spot classification")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--variant", type=str, default="hybrid_class_aware", help="Model variant")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pth")
    parser.add_argument("--config-dir", type=str, default="config", help="Directory containing config YAML files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_path = args.checkpoint
    if not ckpt_path:
        ckpt_path = f"outputs/{args.variant}/checkpoints/best_f1.pth"
        
    print(f"Loading {args.variant} model from: {ckpt_path} on {device}...")
    model, cfg = load_model_from_checkpoint(args.variant, ckpt_path, device, config_dir=args.config_dir)
    
    print(f"\nClassifying image: {args.image}\n" + "="*50)
    res = classify_image(args.image, model, cfg, device)
    
    print(f"Predicted Class : {res['pred_class']} ({res['confidence']:.2f}% confidence)")
    print(f"Inference Time  : {res['latency_ms']:.2f} ms")
    print("\nClass Probabilities:")
    for name, p in res["probabilities"].items():
        bar = "#" * int(p / 5)
        print(f"  {name:22s}: {p:6.2f}% | {bar}")


if __name__ == "__main__":
    main()
