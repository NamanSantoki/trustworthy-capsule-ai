import os
import yaml
from typing import Optional
from pydantic import BaseModel, Field

class PathsConfig(BaseModel):
    dataset_dir: str = "Dataset"
    outputs_dir: str = "outputs"
    logs_dir: str = "outputs/logs"
    plots_dir: str = "outputs/plots"
    reports_dir: str = "outputs/reports"

class AnalysisSection(BaseModel):
    dataset_dir: str = "Dataset"
    outputs_dir: str = "outputs"
    sample_grid_size: int = 4
    fig_format: str = "png"
    bin_count: int = 50

class ModelSection(BaseModel):
    name: str = "efficientnet_v2_s"
    pretrained: bool = True
    num_classes: int = 4
    img_size: int = 224
    dropout_rate: float = 0.2

class TrainingSection(BaseModel):
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    mixed_precision: bool = True
    early_stopping_patience: int = 5
    checkpoint_metric: str = "val_loss"
    tensorboard: bool = True

class DeploymentSection(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    model_checkpoint: str = "outputs/checkpoints/best_model.pth"

class AppConfig(BaseModel):
    project_name: str = "Trustworthy-Capsule-AI"
    seed: int = 42
    device: str = "cuda"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    analysis: AnalysisSection = Field(default_factory=AnalysisSection)
    model: ModelSection = Field(default_factory=ModelSection)
    training: TrainingSection = Field(default_factory=TrainingSection)
    deployment: DeploymentSection = Field(default_factory=DeploymentSection)

def load_config(config_dir: str = "config") -> AppConfig:
    """Loads split configuration YAML files and merges them into a single validated AppConfig."""
    merged_dict = {}
    config_files = ["base.yaml", "analysis.yaml", "model.yaml", "training.yaml", "deployment.yaml"]
    
    for f in config_files:
        path = os.path.join(config_dir, f)
        if os.path.exists(path):
            with open(path, "r") as stream:
                content = yaml.safe_load(stream)
                if content and isinstance(content, dict):
                    merged_dict.update(content)
        else:
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
    return AppConfig.model_validate(merged_dict)
