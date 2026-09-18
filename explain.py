"""Command line script to generate model explanations using Grad-CAM.

Examples:
    # Single image
    python explain.py --image path/to/image.jpg --checkpoint outputs/checkpoints/best_val_loss.pth

    # Folder of images
    python explain.py --image_dir path/to/folder --checkpoint outputs/checkpoints/best_val_loss.pth

    # Specify custom output directory and target layer
    python explain.py --image path/to/image.jpg --checkpoint outputs/checkpoints/best_val_loss.pth --output_dir outputs/explainability_custom
"""

import argparse
import sys
from src.config import load_config
from src.explainability.pipeline import ExplainabilityPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanations for Trustworthy Capsule AI models.")
    parser.add_argument("--config", type=str, default="config", help="Directory containing YAML configuration files.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pth).")
    parser.add_argument("--image", type=str, default=None, help="Path to single input image.")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to directory containing input images.")
    parser.add_argument("--target_class", type=int, default=None, help="Target class index (optional). Defaults to model prediction.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory to save visualizations.")
    parser.add_argument("--target_layer", type=str, default=None, help="Optional layer attribute path string override.")
    parser.add_argument("--device", type=str, default=None, help="Device to execute model on ('cpu', 'cuda').")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.image and not args.image_dir:
        print("Error: You must provide either --image or --image_dir argument.")
        sys.exit(1)

    cfg = load_config(args.config)
    if args.checkpoint:
        cfg.deployment.model_checkpoint = args.checkpoint
    if args.target_layer:
        cfg.explainability.target_layer = args.target_layer
    if args.output_dir:
        cfg.explainability.outputs_dir = args.output_dir

    pipeline = ExplainabilityPipeline(cfg=cfg, checkpoint_path=args.checkpoint, device=args.device)

    if args.image:
        print(f"Generating explanation for single image: {args.image}")
        res = pipeline.explain_image(
            image_path=args.image,
            target_category=args.target_class,
            output_dir=args.output_dir,
        )
        print(f"Done! Predicted class: {res['predicted_class']}. Outputs saved under prefix: {res['save_path_base']}")

    if args.image_dir:
        print(f"Generating explanations for images in directory: {args.image_dir}")
        results = pipeline.explain_folder(
            folder_path=args.image_dir,
            output_dir=args.output_dir,
        )
        print(f"Done! Processed {len(results)} images. Visualizations saved to: {args.output_dir or cfg.explainability.outputs_dir}")


if __name__ == "__main__":
    main()
