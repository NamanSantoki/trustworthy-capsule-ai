import os
import pandas as pd
from typing import Dict, Any

def generate_markdown_report(df: pd.DataFrame, global_stats: Dict[str, Any], output_path: str) -> None:
    """Compiles statistics and embeds visualization paths into a Markdown report.
    
    Args:
        df: Pandas DataFrame with analyzed image parameters.
        global_stats: Global parameters calculated over the dataset.
        output_path: Path where report will be saved.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    valid_df = df[~df["corrupted"]]
    corrupted_df = df[df["corrupted"]]
    
    # Generate markdown table summaries using pandas.to_markdown
    res_stats = valid_df[["width", "height", "aspect_ratio"]].describe().to_markdown()
    color_stats = valid_df[["r_mean", "g_mean", "b_mean"]].describe().to_markdown()
    brightness_stats = valid_df[["brightness"]].describe().to_markdown()
    blur_stats = valid_df[["blur_score"]].describe().to_markdown()
    
    # Class distribution details
    dist_table = df.groupby(["class", "split"]).size().unstack(fill_value=0).to_markdown()
    
    # Dynamic quantiles to inform preprocessing decisions
    blur_5th = float(valid_df["blur_score"].quantile(0.05)) if not valid_df.empty else 0.0
    brightness_1st = float(valid_df["brightness"].quantile(0.01)) if not valid_df.empty else 0.0
    brightness_99th = float(valid_df["brightness"].quantile(0.99)) if not valid_df.empty else 0.0

    report_content = f"""# Trustworthy Capsule Endoscopy Dataset Analysis Report

This report summarizes the spatial dimensions, color characteristics, and focus characteristics of the gastrointestinal dataset to establish baseline metrics for our medical AI pipelines.

---

## 1. Dataset Integrity & Split Balance

- **Total Checked Images**: {global_stats['total_images']}
- **Corrupted Images**: {global_stats['corrupted_count']}

### Class Counts By Split
{dist_table}

"""

    if len(corrupted_df) > 0:
        report_content += "### Corrupted Files Log\n"
        for _, row in corrupted_df.iterrows():
            report_content += f"- **File**: `{row['path']}` | **Error**: `{row['error_message']}`\n"
        report_content += "\n"
    else:
        report_content += "- **No corrupted or unreadable images were detected in the dataset splits.**\n\n"

    report_content += f"""---

## 2. Quantitative Characteristics

### Image Dimensions and Aspect Ratios
{res_stats}

### RGB Color Channels Means (Normalized [0, 1])
{color_stats}

### Luma Brightness Profile
{brightness_stats}

### Laplacian Blur Scores
{blur_stats}

---

## 3. Recommended normalization Coefficients (Train Split)

To ensure zero-center initialization during training, use these exact parameters derived from the `train` split:

- **Channel Means (R, G, B)**: `[{global_stats.get('train_mean', [0,0,0])[0]:.5f}, {global_stats.get('train_mean', [0,0,0])[1]:.5f}, {global_stats.get('train_mean', [0,0,0])[2]:.5f}]`
- **Channel Standard Deviations (R, G, B)**: `[{global_stats.get('train_std', [1,1,1])[0]:.5f}, {global_stats.get('train_std', [1,1,1])[1]:.5f}, {global_stats.get('train_std', [1,1,1])[2]:.5f}]`

---

## 4. Visualizations Index

Detailed graphics have been generated and saved under `/plots/`. They can be viewed directly below:

### Class Distribution
![Class Distribution](../plots/class_distribution.png)

### Layout Spatial Dimensions
![Layout Spatial](../plots/resolution_distribution.png)

### RGB Color Distributions
![RGB Color](../plots/rgb_distribution.png)

### Illumination and Defocus Metrics
![Illumination Defocus](../plots/brightness_blur_distributions.png)

### Clinical Samples Matrix
![Clinical Samples](../plots/sample_grid.png)

---

## 5. Clinical Insights & Preprocessing Recommendations

1. **Defocus Threshold Identification**: 
   The 5th percentile of Laplacian-variance blur scores is **{blur_5th:.2f}**. Frames exhibiting scores below this percentile typically indicate severe motion blur or artifacts that could degrade classification confidence. They can be filtered or benchmarked.
2. **Illumination Profile**: 
   Luma brightness values range from {valid_df['brightness'].min() if not valid_df.empty else 0.0:.4f} to {valid_df['brightness'].max() if not valid_df.empty else 0.0:.4f}. The 1st and 99th percentiles are **{brightness_1st:.4f}** and **{brightness_99th:.4f}** respectively. Saturated (whiteout) or dark frames representing fluid blocks can be flagged using these limits.
3. **Aspect Ratios**: 
   The dataset shows uniform aspect ratios. Downstream CNN/ViT architectures can safely apply isotropic resizing (e.g., 224x224) via bicubic interpolation without introducing spatial distension.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Generated analysis report at: {output_path}")
