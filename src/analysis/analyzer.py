import os
import argparse
from src.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.analysis.metrics import perform_dataset_analysis
from src.analysis.visualizer import (
    plot_class_distribution,
    plot_resolution_distribution,
    plot_rgb_distribution,
    plot_brightness_blur_distributions,
    plot_random_sample_grid
)
from src.analysis.reporter import generate_markdown_report

def main():
    parser = argparse.ArgumentParser(description="Trustworthy-Capsule-AI Phase 1: Dataset Analysis")
    parser.add_argument("--config-dir", type=str, default="config", help="Path to config directory")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config_dir)

    # Initialize reproducible seeds
    set_seed(cfg.seed)

    # Setup Logging
    logger = setup_logger(
        name="DatasetAnalysis",
        log_dir=cfg.paths.logs_dir,
        level=10  # DEBUG level to capture comprehensive logs
    )
    logger.info("Initializing Dataset Analysis Module...")
    logger.info(f"Loaded configuration files from directory: '{args.config_dir}'")
    logger.info(f"Project Name: {cfg.project_name} | Seed: {cfg.seed}")

    dataset_dir = cfg.analysis.dataset_dir
    plots_dir = cfg.paths.plots_dir
    reports_dir = cfg.paths.reports_dir

    # Verify directory structure existence
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Perform analysis
    logger.info(f"Starting crawl of raw splits in directory: {dataset_dir}")
    try:
        df, stats = perform_dataset_analysis(dataset_dir)
    except Exception as e:
        logger.error(f"Analysis orchestrator failed due to an exception: {e}", exc_info=True)
        return

    logger.info(f"Metrics collection successfully completed. Checked: {stats['total_images']} frames.")
    logger.info(f"Corrupted image files found: {stats['corrupted_count']}.")
    
    # Save CSV of raw stats
    csv_path = os.path.join(reports_dir, "dataset_metrics.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Raw image properties compiled and saved to: {csv_path}")

    # Generate Visualizations
    logger.info("Generating publication-ready distribution charts...")
    
    class_dist_plot = os.path.join(plots_dir, "class_distribution.png")
    plot_class_distribution(df, class_dist_plot)
    logger.info(f"Saved: {class_dist_plot}")

    res_plot = os.path.join(plots_dir, "resolution_distribution.png")
    plot_resolution_distribution(df, res_plot)
    logger.info(f"Saved: {res_plot}")

    rgb_plot = os.path.join(plots_dir, "rgb_distribution.png")
    plot_rgb_distribution(df, rgb_plot)
    logger.info(f"Saved: {rgb_plot}")

    brightness_blur_plot = os.path.join(plots_dir, "brightness_blur_distributions.png")
    plot_brightness_blur_distributions(df, brightness_blur_plot)
    logger.info(f"Saved: {brightness_blur_plot}")

    grid_size = cfg.analysis.sample_grid_size
    grid_plot = os.path.join(plots_dir, "sample_grid.png")
    plot_random_sample_grid(df, dataset_dir, grid_plot, grid_size=grid_size)
    logger.info(f"Saved representative sample grid matrix: {grid_plot}")

    # Generate Reports
    report_path = os.path.join(reports_dir, "dataset_analysis_report.md")
    logger.info(f"Writing dataset diagnostic summary to: {report_path}")
    generate_markdown_report(df, stats, report_path)
    logger.info("Dataset analysis report successfully exported.")

if __name__ == "__main__":
    main()
