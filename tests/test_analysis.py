import os
import tempfile
import numpy as np
import pytest
import cv2
from src.config import load_config, AppConfig
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.analysis.metrics import analyze_single_image

def test_config_loading():
    """Verify config loading compiles split YAML configurations correctly."""
    cfg = load_config("config")
    assert isinstance(cfg, AppConfig)
    assert cfg.project_name == "Trustworthy-Capsule-AI"
    assert cfg.seed == 42
    assert cfg.paths.dataset_dir == "Dataset"

def test_logger_setup():
    """Verify logging setup formats and writes records correctly to the disk."""
    import logging
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = setup_logger(name="TestLogger", log_dir=tmp_dir)
        logger.info("Verifying temporary log system check message.")
        log_file = os.path.join(tmp_dir, "run.log")
        assert os.path.exists(log_file)
        with open(log_file, "r", encoding="utf-8") as f:
            contents = f.read()
        assert "Verifying temporary log system check message." in contents
        
        # Close logging handlers to release file lock on Windows
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

def test_image_analysis():
    """Validate metric calculations and corrupted file handling."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a valid test image (height=200 pixels, width=100 pixels, color red)
        img_path = os.path.join(tmp_dir, "test_img.png")
        dummy_img = np.zeros((200, 100, 3), dtype=np.uint8)
        # BGR representation: Red is Index 2
        dummy_img[:, :, 2] = 255
        
        cv2.imwrite(img_path, dummy_img)
        
        # Analyze valid image
        stats = analyze_single_image(img_path, "train", "0_normal")
        assert stats["corrupted"] is False
        assert stats["width"] == 100
        assert stats["height"] == 200
        assert stats["aspect_ratio"] == 0.5
        assert stats["pixel_count"] == 20000
        assert stats["r_mean"] == pytest.approx(1.0, abs=1e-2)
        assert stats["g_mean"] == pytest.approx(0.0, abs=1e-2)
        assert stats["b_mean"] == pytest.approx(0.0, abs=1e-2)
        assert stats["brightness"] == pytest.approx(0.299, abs=1e-2)  # 0.299 * R
        
        # Test corrupt image validation
        corrupt_path = os.path.join(tmp_dir, "corrupt.png")
        with open(corrupt_path, "w") as f:
            f.write("This is raw corrupted text data, not a valid image format.")
            
        stats_corrupt = analyze_single_image(corrupt_path, "train", "0_normal")
        assert stats_corrupt["corrupted"] is True
        assert len(stats_corrupt["error_message"]) > 0
