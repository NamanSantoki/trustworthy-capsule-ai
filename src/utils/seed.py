import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Configures deterministic seeds across Python, NumPy, and PyTorch configurations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enforce deterministic convolutions to guarantee exact reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
