"""High-level orchestration pipeline for model explainability."""

import os
import glob
from typing import Optional, List, Dict, Any, Union, Tuple
import torch
import torch.nn as nn
from src.config import AppConfig, load_config
from src.models.factory import build_model
from src.explainability.gradcam import GradCAM
from src.explainability.utils import (
    get_target_layer,
    load_model_checkpoint,
    preprocess_image,
    overlay_heatmap,
    save_explainability_output,
)


class ExplainabilityPipeline:
    """End-to-end Grad-CAM inference and visualization pipeline."""

    def __init__(
        self,
        cfg: Optional[AppConfig] = None,
        model: Optional[nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize pipeline with model and config.

        Args:
            cfg: Application configuration. Defaults to loading from config/.
            model: PyTorch model instance. If None, initialized via factory.
            checkpoint_path: Optional path to checkpoint weights.
            device: Execution device (cpu or cuda).
        """
        self.cfg = cfg or load_config("config")

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model is not None:
            self.model = model
        else:
            self.model = build_model(self.cfg)

        ckpt = checkpoint_path or getattr(self.cfg.deployment, "model_checkpoint", None)
        if ckpt and os.path.exists(ckpt):
            self.model = load_model_checkpoint(self.model, ckpt, self.device)
        else:
            self.model.to(self.device)

        self.model.eval()

        # Initialize GradCAM explainer
        target_layer_path = getattr(self.cfg.explainability, "target_layer", None)
        target_layer = get_target_layer(self.model, target_layer_path)
        self.explainer = GradCAM(self.model, target_layer)

    def explain_image(
        self,
        image_path: str,
        target_category: Optional[int] = None,
        output_dir: Optional[str] = None,
        output_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate Grad-CAM explanation for a single image file.

        Args:
            image_path: Path to target input image file.
            target_category: Optional target class index. If None, uses predicted class.
            output_dir: Optional directory to save output artifacts. Defaults to cfg.explainability.outputs_dir.
            output_prefix: Prefix for output filenames. Defaults to image filename.

        Returns:
            Dictionary containing heatmap tensor, overlay image, predicted class, and saved paths.
        """
        out_dir = output_dir or self.cfg.explainability.outputs_dir
        os.makedirs(out_dir, exist_ok=True)

        mean_tuple: Tuple[float, float, float] = (
            float(self.cfg.data.mean[0]),
            float(self.cfg.data.mean[1]),
            float(self.cfg.data.mean[2]),
        )
        std_tuple: Tuple[float, float, float] = (
            float(self.cfg.data.std[0]),
            float(self.cfg.data.std[1]),
            float(self.cfg.data.std[2]),
        )

        tensor_img, orig_rgb = preprocess_image(
            image_input=image_path,
            img_size=self.cfg.model.img_size,
            mean=mean_tuple,
            std=std_tuple,
        )
        tensor_img = tensor_img.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_img)
            pred_class = int(torch.argmax(logits, dim=1).item())

        target_cls = target_category if target_category is not None else pred_class

        cam = self.explainer.attribute(tensor_img, target_category=target_cls) # (1, 1, H, W)

        overlay_rgb, heatmap_rgb = overlay_heatmap(
            heatmap=cam,
            original_rgb=orig_rgb,
            alpha=self.cfg.explainability.alpha,
            colormap=self.cfg.explainability.colormap,
        )

        base_name = output_prefix or os.path.splitext(os.path.basename(image_path))[0]
        save_base_path = os.path.join(out_dir, f"{base_name}_class{target_cls}")

        meta = {
            "image_path": image_path,
            "predicted_class": pred_class,
            "target_class": target_cls,
            "model_variant": self.cfg.model.variant,
            "explainability_method": self.cfg.explainability.method,
        }

        save_explainability_output(
            original_rgb=orig_rgb,
            overlay_rgb=overlay_rgb,
            heatmap_rgb=heatmap_rgb,
            save_path=save_base_path,
            save_overlay=self.cfg.explainability.save_overlay,
            save_side_by_side=self.cfg.explainability.save_side_by_side,
            metadata=meta,
        )

        return {
            "predicted_class": pred_class,
            "target_class": target_cls,
            "heatmap": cam,
            "save_path_base": save_base_path,
        }

    def explain_folder(
        self,
        folder_path: str,
        output_dir: Optional[str] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
    ) -> List[Dict[str, Any]]:
        """Run explainability pipeline on all images within a folder.

        Args:
            folder_path: Path to directory containing images.
            output_dir: Directory to save visual outputs.
            extensions: Tuple of file extensions to process.

        Returns:
            List of result dicts for each processed image.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))

        image_files = sorted(list(set(image_files)))
        results = []
        for img_path in image_files:
            res = self.explain_image(image_path=img_path, output_dir=output_dir)
            results.append(res)

        return results
