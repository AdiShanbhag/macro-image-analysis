# src/utils/plotting.py
# Standalone helper functions for saving charts and visual outputs.
# These functions are kept separate from EDAService so they can be
# reused by other modules (e.g. Member 2's evaluation outputs) without
# importing the full EDA class.
#
# Author: Member 1
# Unit: Software Technology 1 (4483/8995)

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd


def save_sample_grid(
    dataframe: pd.DataFrame,
    output_path: Path,
    sample_count: int = 9,
    random_seed: int = 42,
) -> None:
    """Save a grid of sample macroinvertebrate images for visual inspection.

    Randomly selects images from the DataFrame and arranges them in a
    3x3 grid. Each image is shown with its class label as the title.
    Images that cannot be read are skipped silently.

    Args:
        dataframe: DataFrame with at least 'file_path' and 'label' columns.
        output_path: Path where the grid image will be saved.
        sample_count: Number of images to include. Defaults to 9.
        random_seed: Seed for reproducible random sampling. Defaults to 42.
    """
    if dataframe.empty:
        print("[WARNING] Cannot save sample grid: DataFrame is empty.")
        return

    sample_df = dataframe.sample(
        min(sample_count, len(dataframe)), random_state=random_seed
    )

    cols = 3
    rows = (len(sample_df) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    plotted = 0
    for ax, (_, row) in zip(axes_flat, sample_df.iterrows()):
        image = cv2.imread(row["file_path"])
        if image is None:
            ax.axis("off")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
        ax.set_title(row["label"], fontsize=9)
        ax.axis("off")
        plotted += 1

    # Hide any unused axes
    for ax in list(axes_flat)[plotted:]:
        ax.axis("off")

    plt.suptitle("Sample Macroinvertebrate Images", fontsize=14, y=1.01)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Sample grid saved to {output_path}")


def save_bar_chart(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    rotate_labels: bool = True,
) -> None:
    """Save a simple bar chart from a Pandas Series.

    Args:
        series: Series where the index is the x-axis and values are heights.
        title: Chart title.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        output_path: Path where the chart PNG will be saved.
        rotate_labels: Whether to rotate x-axis labels 90 degrees.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    series.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if rotate_labels:
        plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Chart saved to {output_path}")


def save_summary_table(summary: dict, output_path: Path) -> None:
    """Save a dataset summary dictionary as a formatted text file.

    Args:
        summary: Dictionary of metric names to values.
        output_path: Path where the text file will be saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Dataset Summary", "=" * 40]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"  {key:<25} {value:.2f}")
        else:
            lines.append(f"  {key:<25} {value}")
    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")
    print(f"[INFO] Summary table saved to {output_path}")
