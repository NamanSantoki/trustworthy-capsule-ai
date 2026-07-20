import os
import torch
from torch.utils.data import DataLoader
from src.config import load_config
from src.data.dataset import GenericCapsuleDataset
from src.data.transforms import build_train_transforms, build_val_test_transforms
from src.data.dataloader import create_dataloaders

def test_dataset_discovery():
    """Verify GenericCapsuleDataset dynamically discovers classes and labels."""
    cfg = load_config("config")
    test_dir = os.path.join(cfg.paths.dataset_dir, "test")
    
    dataset = GenericCapsuleDataset(test_dir)
    assert len(dataset.classes) == 4
    assert "0_normal" in dataset.classes
    assert dataset.class_to_idx["0_normal"] == 0
    assert len(dataset) == 800  # 200 images * 4 classes
    
    # Check item structure
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "image" in sample
    assert "label" in sample
    assert "path" in sample
    assert "class_name" in sample
    assert sample["class_name"] == "0_normal"
    assert sample["label"] == 0

def test_transforms_generation():
    """Test that dynamic transform factories return valid Compose pipelines."""
    cfg = load_config("config")
    
    train_tr = build_train_transforms(cfg)
    val_tr = build_val_test_transforms(cfg)
    
    assert train_tr is not None
    assert val_tr is not None

def test_dataloader_creation():
    """Test DataLoader factory configures correct batch shapes and types."""
    cfg = load_config("config")
    
    # Overwrite batch size to speed up testing
    cfg.training.batch_size = 4
    cfg.training.num_workers = 0  # 0 workers avoids Windows spawn latency in unit tests
    
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    
    # Draw one batch from val_loader
    batch = next(iter(val_loader))
    assert isinstance(batch, dict)
    assert batch["image"].shape == (4, 3, 224, 224)
    assert batch["label"].shape == (4,)
    
    # Inspect metadata types
    assert isinstance(batch["path"], list)
    assert len(batch["path"]) == 4
    assert isinstance(batch["class_name"], list)
    assert len(batch["class_name"]) == 4
