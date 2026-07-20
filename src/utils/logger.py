import os
import logging
import sys

def setup_logger(
    name: str = "TrustworthyCapsuleAI", 
    log_dir: str = "outputs/logs", 
    level: int = logging.INFO
) -> logging.Logger:
    """Configuration interface to register synchronized stdout and filesystem logging handlers."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid registering duplicate handlers
    if logger.hasHandlers():
         logger.handlers.clear()
         
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Target
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Target
    log_file = os.path.join(log_dir, "run.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
