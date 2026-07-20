import torchvision.transforms as T
from src.config import AppConfig

def build_train_transforms(cfg: AppConfig) -> T.Compose:
    """Builds PyTorch training transformations dynamically from configuration values.
    
    Args:
        cfg: Validated Pydantic configuration object.
        
    Returns:
        torchvision.transforms.Compose pipeline.
    """
    img_size = cfg.model.img_size
    mean = cfg.data.mean
    std = cfg.data.std
    aug = cfg.data_pipeline.augmentations
    
    transforms_list = []
    
    # Standard medical resizing keeping features isotropic
    transforms_list.append(T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC))
    
    # Conditional augmentations based on YAML configs
    if aug.horizontal_flip:
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))
        
    if aug.vertical_flip:
         transforms_list.append(T.RandomVerticalFlip(p=0.5))
         
    if aug.rotation_degrees > 0:
         transforms_list.append(T.RandomRotation(degrees=aug.rotation_degrees))
         
    if aug.brightness > 0 or aug.contrast > 0 or aug.saturation > 0 or aug.hue > 0:
         transforms_list.append(T.ColorJitter(
             brightness=aug.brightness,
             contrast=aug.contrast,
             saturation=aug.saturation,
             hue=aug.hue
         ))
         
    # Convert PIL to Tensor and Normalize using train split statistics
    transforms_list.append(T.ToTensor())
    transforms_list.append(T.Normalize(mean=mean, std=std))
    
    return T.Compose(transforms_list)

def build_val_test_transforms(cfg: AppConfig) -> T.Compose:
    """Builds PyTorch validation / testing transforms (resizing and normalization only).
    
    Args:
        cfg: Validated Pydantic configuration object.
        
    Returns:
        torchvision.transforms.Compose pipeline.
    """
    img_size = cfg.model.img_size
    mean = cfg.data.mean
    std = cfg.data.std
    
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
