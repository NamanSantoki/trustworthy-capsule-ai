import os
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import List, Dict, Any

# Configure Matplotlib styles for academic, publication-friendly plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 16,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def plot_class_distribution(df: pd.DataFrame, output_path: str) -> None:
    """Plots a stacked or grouped bar chart displaying the image counts per class and split."""
    plt.figure(figsize=(10, 6))
    
    # Calculate counts grouped by class and split
    counts = df.groupby(["class", "split"]).size().reset_index(name="count")
    
    # Pivot for plotting
    pivot_df = counts.pivot(index="class", columns="split", values="count").fillna(0)
    # Ensure splits order
    splits_order = [s for s in ["train", "val", "test"] if s in pivot_df.columns]
    pivot_df = pivot_df[splits_order]
    
    # Standard stacked or grouped bar
    pivot_df.plot(kind="bar", stacked=True, color=["#1f77b4", "#ff7f0e", "#2ca02c"], edgecolor="black", width=0.6, ax=plt.gca())
    
    plt.title("Capsule Endoscopy Class Distribution Across Splits")
    plt.xlabel("Diagnosis Class")
    plt.ylabel("Image Count")
    plt.xticks(rotation=15)
    plt.legend(title="Dataset Split")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_resolution_distribution(df: pd.DataFrame, output_path: str) -> None:
    """Plots histograms of image resolutions (widths, heights, and aspect ratios)."""
    valid_df = df[~df["corrupted"]]
    if valid_df.empty:
         return
         
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Widths histogram
    axes[0].hist(valid_df["width"], bins=20, color="#8c564b", edgecolor="black", alpha=0.9)
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Frequency")
    
    # Heights histogram
    axes[1].hist(valid_df["height"], bins=20, color="#9467bd", edgecolor="black", alpha=0.9)
    axes[1].set_title("Image Height Distribution")
    axes[1].set_xlabel("Height (pixels)")
    axes[1].set_ylabel("Frequency")
    
    # Aspect Ratio histogram
    axes[2].hist(valid_df["aspect_ratio"], bins=20, color="#e377c2", edgecolor="black", alpha=0.9)
    axes[2].set_title("Aspect Ratio Distribution (W/H)")
    axes[2].set_xlabel("Aspect Ratio")
    axes[2].set_ylabel("Frequency")
    
    plt.suptitle("Spatial Dimensions and Layout Metrics", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_rgb_distribution(df: pd.DataFrame, output_path: str) -> None:
    """Plots histogram distributions of the R, G, and B channel means."""
    valid_df = df[~df["corrupted"]]
    if valid_df.empty:
         return
         
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    channels = ["r_mean", "g_mean", "b_mean"]
    titles = ["Red Channel Means", "Green Channel Means", "Blue Channel Means"]
    
    for i, (chan, col, title) in enumerate(zip(channels, colors, titles)):
        axes[i].hist(valid_df[chan], bins=50, color=col, edgecolor="black", alpha=0.85)
        axes[i].set_title(title)
        axes[i].set_xlabel("Pixel Value (normalized [0, 1])")
        axes[i].set_ylabel("Frequency")
        
    plt.suptitle("RGB Channel Average Intensities Distribution", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_brightness_blur_distributions(df: pd.DataFrame, output_path: str) -> None:
    """Plots publication-friendly histograms for the Luma Brightness and Laplacian-variance Blur scores."""
    valid_df = df[~df["corrupted"]]
    if valid_df.empty:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Brightness index (luma)
    axes[0].hist(valid_df["brightness"], bins=50, color="#bcbd22", edgecolor="black", alpha=0.85)
    axes[0].set_title("Global Luma Brightness Profile")
    axes[0].set_xlabel("Lightness (normalized [0, 1])")
    axes[0].set_ylabel("Frequency")
    
    # Blur index (variance of Laplacian)
    # Using log scale for blur score since Laplacian variance ranges can span several orders of magnitude
    scores = valid_df["blur_score"].replace(0.0, np.nan).dropna()
    log_scores = np.log10(scores)
    
    axes[1].hist(log_scores, bins=50, color="#17becf", edgecolor="black", alpha=0.85)
    axes[1].set_title("Laplacian Variance Blur Distribution")
    axes[1].set_xlabel("Log10(Variance of Laplacian)")
    axes[1].set_ylabel("Frequency")
    
    plt.suptitle("Illumination and Defocus Analysis", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_random_sample_grid(df: pd.DataFrame, dataset_dir: str, output_path: str, grid_size: int = 4) -> None:
    """Saves a grid showing random sample images (grid_size x grid_size) to visualize diagnostic features.
    
    The grid columns will represent the different classes (0_normal, 1_ulcerative_colitis, 2_polyps, 3_esophagitis)
    and the grid rows will represent sample images.
    """
    valid_df = df[~df["corrupted"]]
    if valid_df.empty:
         return
         
    classes = sorted(valid_df["class"].unique())
    if len(classes) == 0:
        return
        
    fig, axes = plt.subplots(grid_size, len(classes), figsize=(12, 12))
    
    # Ensure axes is always a 2D array
    if grid_size == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(classes) == 1:
        axes = np.expand_dims(axes, axis=1)
        
    for col_idx, class_name in enumerate(classes):
        class_subset = valid_df[valid_df["class"] == class_name]
        
        # Sample images
        if len(class_subset) >= grid_size:
            sampled_records = class_subset.sample(n=grid_size, random_state=42).to_dict("records")
        else:
            sampled_records = class_subset.to_dict("records")
            # Pad if needed
            while len(sampled_records) < grid_size and len(sampled_records) > 0:
                sampled_records.append(random.choice(sampled_records))
                
        for row_idx in range(grid_size):
            ax = axes[row_idx, col_idx]
            if row_idx < len(sampled_records):
                record = sampled_records[row_idx]
                img_path = record["path"]
                try:
                    # Load with OpenCV, convert to RGB for plotting
                    cv_img = cv2.imread(img_path)
                    if cv_img is not None:
                        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        ax.imshow(rgb_img)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        # Set column titles only for the top row
                        if row_idx == 0:
                            ax.set_title(class_name, fontsize=12, fontweight="bold")
                    else:
                        ax.text(0.5, 0.5, "Read Error", ha="center", va="center")
                        ax.axis('off')
                except Exception as e:
                    ax.text(0.5, 0.5, "Err Load", ha="center", va="center")
                    ax.axis('off')
            else:
                ax.axis('off')
                
    plt.suptitle("Clinical Classification Representative Visual Samples", y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
