import os
import torch
from src.config import load_config
from src.data.dataloader import create_dataloaders

def run_pipeline_check():
    print("=== Pipeline Check: Initiating Data Pipeline Diagnostics ===")
    
    # 1. Load Configurations
    cfg = load_config("config")
    print(f"Loaded config successfully. Batch Size: {cfg.training.batch_size} | Image Size: {cfg.model.img_size}")
    print(f"Target Normalization Means: {cfg.data.mean}")
    print(f"Target Normalization Stds:  {cfg.data.std}\n")
    
    # 2. Build Loaders
    print("Initializing datasets and PyTorch Dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    
    # Validate details from training dataset
    train_dataset = train_loader.dataset
    print(f"Train Dataset size:      {len(train_dataset)} images")
    print(f"Validation Dataset size:  {len(val_loader.dataset)} images")
    print(f"Test Dataset size:        {len(test_loader.dataset)} images")
    print(f"Discovered Classes:       {train_dataset.classes}")
    print(f"Class-to-Index Maps:      {train_dataset.class_to_idx}\n")
    
    # 3. Pull a Batch
    print("Drawing a single batch from validation loader...")
    try:
        batch = next(iter(val_loader))
    except Exception as e:
        print(f"ERROR: Failed to draw batch. Reason: {e}")
        return
        
    # 4. Verify batch format
    print("\nBatch Verification:")
    if not isinstance(batch, dict):
        print(f"ERROR: Batch type is {type(batch)}, expected dict!")
        return
        
    print(f"Batch Type: {type(batch)}")
    print(f"Batch Keys: {list(batch.keys())}")
    
    img_batch = batch["image"]
    lbl_batch = batch["label"]
    path_batch = batch["path"]
    class_batch = batch["class_name"]
    
    print(f"Images Tensor Shape:  {img_batch.shape} (Expected: [B, 3, H, W])")
    print(f"Labels Tensor Shape:  {lbl_batch.shape} (Expected: [B])")
    print(f"Image tensor Range:  [{img_batch.min().item():.3f}, {img_batch.max().item():.3f}]")
    print(f"Labels Vector:        {lbl_batch.tolist()}")
    
    print("\nMetadata Check (First 3 samples in batch):")
    for i in range(min(3, len(path_batch))):
        print(f"  Sample {i}:")
        print(f"    Path:       {path_batch[i]}")
        print(f"    Class Name: {class_batch[i]}")
        print(f"    Label Index:{lbl_batch[i].item()}")
        
    # Check normalization alignment
    print(f"\nBatch Stats:")
    print(f"  Mean channel values: {[round(img_batch[:, c].mean().item(), 3) for c in range(3)]}")
    print(f"  Std channel values:  {[round(img_batch[:, c].std().item(), 3) for c in range(3)]}")
    
    # 5. Determinism test
    print("\nDeterministic Loading test:")
    train_iterator_1 = iter(train_loader)
    train_iterator_2 = iter(train_loader)
    
    batch_1a = next(train_iterator_1)
    # Re-instantiate loaders to test seed resetting
    train_loader_a, _, _ = create_dataloaders(cfg)
    train_loader_b, _, _ = create_dataloaders(cfg)
    
    batch_a = next(iter(train_loader_a))
    batch_b = next(iter(train_loader_b))
    
    # Assert labels and paths match between separate loader instances initialized with the same config seed
    paths_match = batch_a["path"] == batch_b["path"]
    labels_match = torch.equal(batch_a["label"], batch_b["label"])
    
    print(f"  Repeated Instantiation Paths Match:  {paths_match}")
    print(f"  Repeated Instantiation Labels Match: {labels_match}")
    
    if paths_match and labels_match:
        print("\n>>> PIPELINE CONFIGURATION VALIDATED successfully. Output is conforming and reproducible. <<<")
    else:
        print("\nWARNING: Loader outputs differ across identical seeds. Review PyTorch Generator manual settings.")

if __name__ == "__main__":
    # Add project root to PYTHONPATH dynamically if run as a script
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    run_pipeline_check()
