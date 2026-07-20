import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Any, Dict, Type

from src.config import AppConfig
from src.data.dataset import GenericCapsuleDataset
from src.data.transforms import build_train_transforms, build_val_test_transforms

def seed_worker(worker_id: int) -> None:
    """Worker initialization hook to ensure multi-worker random state determinism.
    
    This prevents PyTorch dataloader workers from replicating random augmentation states.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloaders(
    cfg: AppConfig,
    train_sampler_class: Optional[Type[torch.utils.data.Sampler]] = None,
    train_sampler_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Generates deterministic training, validation, and testing dataloaders from configurations.
    
    Args:
        cfg: Validated Pydantic AppConfig object.
        train_sampler_class: Optional custom Sampler type for the training split (e.g., WeightedRandomSampler).
        train_sampler_kwargs: Optional initialization arguments dictionary for the sampler.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    dataset_dir = cfg.paths.dataset_dir
    
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    test_dir = os.path.join(dataset_dir, "test")
    
    # 1. Build transformation pipelines
    train_transforms = build_train_transforms(cfg)
    val_test_transforms = build_val_test_transforms(cfg)
    
    # 2. Instantiate datasets
    train_dataset = GenericCapsuleDataset(train_dir, transform=train_transforms)
    val_dataset = GenericCapsuleDataset(val_dir, transform=val_test_transforms)
    test_dataset = GenericCapsuleDataset(test_dir, transform=val_test_transforms)
    
    # 3. Configure determinism
    # Generator enforces deterministic index sampling
    generator = torch.Generator()
    generator.manual_seed(cfg.data_pipeline.seed)
    
    # 4. Handle custom sampler logic
    # Shuffle is disabled if a custom sampler is passed, as sampler dictates order.
    sampler = None
    shuffle = True
    
    if train_sampler_class is not None:
        kwargs = train_sampler_kwargs or {}
        sampler = train_sampler_class(train_dataset, **kwargs)
        shuffle = False
        
    # Check GPU availability to set memory pinning
    pin_memory = torch.cuda.is_available() and cfg.device == "cuda"
    
    # 5. Build loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator
    )
    
    # Validation and testing utilize simple sequential loading (no shuffle)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator
    )
    
    return train_loader, val_loader, test_loader
