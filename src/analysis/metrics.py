import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

def analyze_single_image(filepath: str, split: str, class_name: str) -> Dict[str, Any]:
    """Analyzes a single image file for resolution, RGB, brightness, blur, and corruption.
    
    Args:
        filepath: Absolute or relative path to the image.
        split: The dataset split (train, val, test).
        class_name: The disease label folders.
        
    Returns:
        A dictionary containing stats for this image file.
    """
    result = {
        "path": filepath,
        "split": split,
        "class": class_name,
        "width": 0,
        "height": 0,
        "aspect_ratio": 0.0,
        "r_mean": 0.0,
        "g_mean": 0.0,
        "b_mean": 0.0,
        "brightness": 0.0,
        "blur_score": 0.0,
        "corrupted": False,
        "error_message": "",
        "pixel_count": 0,
        "r_sum": 0.0,
        "g_sum": 0.0,
        "b_sum": 0.0,
        "r_sq_sum": 0.0,
        "g_sq_sum": 0.0,
        "b_sq_sum": 0.0
    }
    
    # 1. Check for file size / existence
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        result["corrupted"] = True
        result["error_message"] = "File is missing or empty."
        return result
        
    try:
        # Verify decoding with PIL
        with Image.open(filepath) as img:
            img.verify()
            
        # Re-open for actual reading with OpenCV to obtain pixel metrics
        # (Reads in BGR sequence)
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            result["corrupted"] = True
            result["error_message"] = "OpenCV failed to decode image."
            return result
            
        h, w, c = img_bgr.shape
        result["width"] = w
        result["height"] = h
        result["aspect_ratio"] = float(w) / float(h) if h > 0 else 0.0
        result["pixel_count"] = h * w
        
        # Convert BGR to RGB, scale to [0, 1]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Calculate sums and squared sums for global running mean / std dev computation
        r_chan = img_rgb[:, :, 0]
        g_chan = img_rgb[:, :, 1]
        b_chan = img_rgb[:, :, 2]
        
        # Store individual statistics
        result["r_mean"] = float(np.mean(r_chan))
        result["g_mean"] = float(np.mean(g_chan))
        result["b_mean"] = float(np.mean(b_chan))
        
        # Global running variables
        result["r_sum"] = float(np.sum(r_chan))
        result["g_sum"] = float(np.sum(g_chan))
        result["b_sum"] = float(np.sum(b_chan))
        
        result["r_sq_sum"] = float(np.sum(r_chan ** 2))
        result["g_sq_sum"] = float(np.sum(g_chan ** 2))
        result["b_sq_sum"] = float(np.sum(b_chan ** 2))
        
        # Luma Brightness: Y = 0.299R + 0.587G + 0.114B
        luma = 0.299 * r_chan + 0.587 * g_chan + 0.114 * b_chan
        result["brightness"] = float(np.mean(luma))
        
        # Blur Score using Variance of Laplacian
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        result["blur_score"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
    except Exception as e:
        result["corrupted"] = True
        result["error_message"] = str(e)
        
    return result

def perform_dataset_analysis(dataset_dir: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Crawls dataset directories and compiles metrics sequentially.
    
    Args:
        dataset_dir: Root directory of the dataset.
        
    Returns:
        A tuple of:
          - A pandas DataFrame containing metrics for all images.
          - A dictionary containing global dataset stats.
    """
    records = []
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
    splits = ["train", "val", "test"]
    
    # Verify directory structure
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Expected split directory is missing: {split_dir}")
            
    # Gather image file registry
    file_list = []
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        for class_folder in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_folder)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                        file_list.append((os.path.join(class_path, filename), split, class_folder))
                        
    # Run sequentially (avoid complex multiprocessing initial implementations)
    print(f"Starting analysis of {len(file_list)} images...")
    for filepath, split, class_name in tqdm(file_list, desc="Processing images", unit="img"):
        stats = analyze_single_image(filepath, split, class_name)
        records.append(stats)
        
    df = pd.DataFrame(records)
    
    # Compute global indicators, filtering out corrupted entries
    valid_df = df[~df["corrupted"]]
    
    global_stats = {}
    if not valid_df.empty:
        # Sum pixels and color fields across the entire dataset to get mathematically exact mean & std dev
        total_pixels = valid_df["pixel_count"].sum()
        
        global_r_mean = valid_df["r_sum"].sum() / total_pixels
        global_g_mean = valid_df["g_sum"].sum() / total_pixels
        global_b_mean = valid_df["b_sum"].sum() / total_pixels
        
        global_r_var = (valid_df["r_sq_sum"].sum() / total_pixels) - (global_r_mean ** 2)
        global_g_var = (valid_df["g_sq_sum"].sum() / total_pixels) - (global_g_mean ** 2)
        global_b_var = (valid_df["b_sq_sum"].sum() / total_pixels) - (global_b_mean ** 2)
        
        global_stats = {
            "r_mean": float(global_r_mean),
            "g_mean": float(global_g_mean),
            "b_mean": float(global_b_mean),
            "r_std": float(np.sqrt(max(0.0, global_r_var))),
            "g_std": float(np.sqrt(max(0.0, global_g_var))),
            "b_std": float(np.sqrt(max(0.0, global_b_var))),
            "corrupted_count": int(df["corrupted"].sum()),
            "total_images": len(df)
        }
        
        # Collect train split statistics separately for normalization recommendation
        train_df = valid_df[valid_df["split"] == "train"]
        if not train_df.empty:
            train_pixels = train_df["pixel_count"].sum()
            tr_mean = [
                float(train_df["r_sum"].sum() / train_pixels),
                float(train_df["g_sum"].sum() / train_pixels),
                float(train_df["b_sum"].sum() / train_pixels)
            ]
            tr_var = [
                float((train_df["r_sq_sum"].sum() / train_pixels) - (tr_mean[0] ** 2)),
                float((train_df["g_sq_sum"].sum() / train_pixels) - (tr_mean[1] ** 2)),
                float((train_df["b_sq_sum"].sum() / train_pixels) - (tr_mean[2] ** 2))
            ]
            global_stats["train_mean"] = tr_mean
            global_stats["train_std"] = [float(np.sqrt(max(0.0, v))) for v in tr_var]
    else:
        global_stats = {
            "r_mean": 0.0, "g_mean": 0.0, "b_mean": 0.0,
            "r_std": 0.0, "g_std": 0.0, "b_std": 0.0,
            "corrupted_count": len(df),
            "total_images": len(df)
        }
        
    return df, global_stats
