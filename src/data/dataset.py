import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Tuple, Any, Callable, Optional

class GenericCapsuleDataset(Dataset):
    """Generic dataset loader that dynamically maps class folders and outputs dictionary-based batches with metadata.
    
    Exposures like `class_to_idx` and `idx_to_class` are resolved dynamically via split subdirectory layouts.
    """
    
    def __init__(self, split_dir: str, transform: Optional[Callable] = None):
        """
        Args:
            split_dir: Path to directory containing class subfolders (e.g. Dataset/train).
            transform: Callable PyTorch transforms to apply to images.
        """
        self.split_dir = split_dir
        self.transform = transform
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
            
        # Discover directories dynamically to avoid hardcoding class labels
        self.classes = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])
        
        if not self.classes:
            raise RuntimeError(f"No class subfolders found inside split directory: {split_dir}")
            
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.idx_to_class = {idx: class_name for idx, class_name in enumerate(self.classes)}
        
        # Crawl directories for valid image files
        self.samples = []
        for class_name in self.classes:
            class_path = os.path.join(split_dir, class_name)
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    filepath = os.path.normpath(os.path.join(class_path, fname))
                    self.samples.append((filepath, self.class_to_idx[class_name], class_name))
                    
        if not self.samples:
            raise RuntimeError(f"No valid image files found within split subdirectories: {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Loads and returns a dictionary sample with image metadata.
        
        Returns:
            Dictionary containing:
              - "image": PIL image processed by transforms (typically torch.Tensor if torchvision transform is set)
              - "label": Target integer index map
              - "path": String containing exact raw image file path
              - "class_name": String target label folder name
        """
        img_path, label, class_name = self.samples[idx]
        
        try:
            # We open with PIL converting to RGB. 
            # This is native to torchvision transforms and avoids manual BGR-to-RGB conversion required by OpenCV.
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to load image file: '{img_path}'. Reason: {e}")
            
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = image
            
        return {
            "image": image_tensor,
            "label": label,
            "path": img_path,
            "class_name": class_name
        }
