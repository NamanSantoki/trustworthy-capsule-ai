import os
import yaml
from typing import Optional, List, Any, Literal
from pydantic import BaseModel, Field

class PathsConfig(BaseModel):
    dataset_dir: str = "Dataset"
    outputs_dir: str = "outputs"
    logs_dir: str = "outputs/logs"
    plots_dir: str = "outputs/plots"
    reports_dir: str = "outputs/reports"
    checkpoints_dir: str = "outputs/checkpoints"

class AnalysisSection(BaseModel):
    dataset_dir: str = "Dataset"
    outputs_dir: str = "outputs"
    sample_grid_size: int = 4
    fig_format: str = "png"
    bin_count: int = 50

class ModelSection(BaseModel):
    variant: Literal["efficientnet_only", "vit_only", "hybrid_concat", "hybrid_adaptive", "hybrid_class_aware"] = "hybrid_class_aware"
    name: str = "efficientnet_v2_s"
    pretrained: bool = True
    num_classes: int = 8
    img_size: int = 224
    dropout_rate: float = 0.2
    fusion_hidden_dim: int = 512
    freeze_backbone_epochs: int = 5
    fusion_d_model: int = 256
    fusion_num_heads: int = 8
    fusion_dropout: float = 0.0
    class_aware_prototype_temperature: float = 0.1
    class_aware_prototype_normalize: bool = True
    mc_samples: int = 20
    calibration_n_bins: int = 10

class TrainingSection(BaseModel):
    epochs: int = 20
    batch_size: int = 32
    num_workers: int = 4
    img_size: int = 224
    learning_rate: float = 0.0005
    backbone_lr: float = 0.00005
    weight_decay: float = 0.0001
    mixed_precision: bool = True
    early_stopping_patience: int = 5
    checkpoint_metric: str = "val_f1"
    tensorboard: bool = True
    label_smoothing: float = 0.1
    scheduler: str = "cosine"
    resume_training: bool = False
    resume_checkpoint: str = "outputs_kvasir/checkpoints/last_epoch.pth"

class DeploymentSection(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    model_checkpoint: str = "outputs/checkpoints/best_model.pth"

class DataSection(BaseModel):
    mean: list[float] = Field(default_factory=lambda: [0.61487, 0.40708, 0.36792])
    std: list[float] = Field(default_factory=lambda: [0.27496, 0.22390, 0.22929])

class AugmentationsConfig(BaseModel):
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotation_degrees: float = 15.0
    brightness: float = 0.1
    contrast: float = 0.1
    saturation: float = 0.1
    hue: float = 0.05

class DataPipelineSection(BaseModel):
    seed: int = 42
    augmentations: AugmentationsConfig = Field(default_factory=AugmentationsConfig)

class ExplainabilitySection(BaseModel):
    method: str = "grad_cam"
    target_layer: Optional[str] = None
    colormap: str = "cv2.COLORMAP_JET"
    alpha: float = 0.5
    outputs_dir: str = "outputs/explainability"
    save_overlay: bool = True
    save_side_by_side: bool = True

class FaithfulnessSection(BaseModel):
    grid_size: int = 14
    fractions: list[float] = Field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    random_seeds: list[int] = Field(default_factory=lambda: [42, 123, 456])
    outputs_dir: str = "outputs/faithfulness"
    blur_kernel_size: int = 15
    blur_sigma: float = 10.0

class AppConfig(BaseModel):
    project_name: str = "Trustworthy-Capsule-AI"
    seed: int = 42
    device: str = "cuda"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    analysis: AnalysisSection = Field(default_factory=AnalysisSection)
    model: ModelSection = Field(default_factory=ModelSection)
    training: TrainingSection = Field(default_factory=TrainingSection)
    deployment: DeploymentSection = Field(default_factory=DeploymentSection)
    data: DataSection = Field(default_factory=DataSection)
    data_pipeline: DataPipelineSection = Field(default_factory=DataPipelineSection)
    explainability: ExplainabilitySection = Field(default_factory=ExplainabilitySection)
    faithfulness: FaithfulnessSection = Field(default_factory=FaithfulnessSection)

def load_config(config_dir: str = "config") -> AppConfig:
    """Loads split configuration YAML files and merges them into a single validated AppConfig."""
    merged_dict = {}
    config_files = ["base.yaml", "analysis.yaml", "model.yaml", "training.yaml", "deployment.yaml", "explainability.yaml", "faithfulness.yaml"]
    
    for f in config_files:
        path = os.path.join(config_dir, f)
        if os.path.exists(path):
            with open(path, "r") as stream:
                content = yaml.safe_load(stream)
                if content and isinstance(content, dict):
                    # Handle top-level keys vs nested sections
                    if f == "explainability.yaml" and "explainability" not in content:
                        merged_dict["explainability"] = content
                    elif f == "faithfulness.yaml" and "faithfulness" not in content:
                        merged_dict["faithfulness"] = content
                    else:
                        merged_dict.update(content)
        elif f not in ("explainability.yaml", "faithfulness.yaml"):
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
    return AppConfig.model_validate(merged_dict)
