# src/services/eda_service.py
# EDA service for the Macroinvertebrate Image Analysis System.
# This class generates all Stage 1 exploratory data analysis outputs
# including charts, summary statistics, and a class-level breakdown.
#
# All outputs are saved to disk so they can be used in the Implementation
# Summary, the README, and the Week 13 presentation.
#
# Author: Member 1
# Unit: Software Technology 1 (4483/8995)

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import EDA_OUTPUT_DIR, RANDOM_SEED, SAMPLE_GRID_COUNT
from src.utils.plotting import save_sample_grid, save_summary_table


class EDAService:
    """Generate and save EDA outputs for the indexed macroinvertebrate dataset.

    This service accepts a Pandas DataFrame produced by DatasetIndexer and
    produces a series of visual and statistical outputs that describe the
    dataset. These outputs inform decisions about class balance, image
    normalisation, and model design in later stages.

    Attributes:
        dataframe: The indexed image DataFrame with label and dimension columns.
        output_dir: Directory where all EDA output files will be saved.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        output_dir: Path = EDA_OUTPUT_DIR,
    ) -> None:
        """Initialise the EDA service.

        Args:
            dataframe: Indexed image DataFrame from DatasetIndexer.
            output_dir: Folder where charts and reports will be saved.
                        Defaults to EDA_OUTPUT_DIR from config.
        """
        self.dataframe = dataframe
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def build_summary(self) -> dict[str, float | int]:
        """Compute and return key dataset statistics.

        Returns:
            A dictionary with total_images, total_classes, mean_width,
            mean_height, min_width, max_width, min_height, and max_height.
        """
        if self.dataframe.empty:
            print("[WARNING] DataFrame is empty. Summary will be zeros.")
            return {}

        summary = {
            "total_images": int(len(self.dataframe)),
            "total_classes": int(self.dataframe["label"].nunique()),
            "mean_width": float(self.dataframe["width"].mean()),
            "mean_height": float(self.dataframe["height"].mean()),
            "min_width": int(self.dataframe["width"].min()),
            "max_width": int(self.dataframe["width"].max()),
            "min_height": int(self.dataframe["height"].min()),
            "max_height": int(self.dataframe["height"].max()),
        }
        return summary

    def build_class_summary(self) -> pd.DataFrame:
        """Return a per-class breakdown of image counts and mean dimensions.

        Returns:
            A DataFrame grouped by label with count, mean width,
            and mean height columns.
        """
        if self.dataframe.empty:
            return pd.DataFrame()

        grouped = (
            self.dataframe.groupby("label")
            .agg(
                count=("file_path", "count"),
                mean_width=("width", "mean"),
                mean_height=("height", "mean"),
            )
            .reset_index()
            .sort_values("count", ascending=False)
        )
        return grouped

    # ------------------------------------------------------------------
    # Chart outputs
    # ------------------------------------------------------------------

    def save_class_distribution(self) -> None:
        """Save a bar chart showing the number of images per class.

        Reveals class imbalance which can affect model training decisions.
        Output saved to: outputs/eda/class_distribution.png
        """
        if self.dataframe.empty:
            print("[WARNING] Skipping class distribution: DataFrame is empty.")
            return

        order = self.dataframe["label"].value_counts().index

        plt.figure(figsize=(14, 6))
        sns.countplot(
            data=self.dataframe,
            x="label",
            order=order,
            palette="Blues_d",
        )
        plt.xticks(rotation=90, ha="right", fontsize=8)
        plt.title("Number of Images per Macroinvertebrate Class", fontsize=14)
        plt.xlabel("Class (Macroinvertebrate Type)", fontsize=11)
        plt.ylabel("Image Count", fontsize=11)
        plt.tight_layout()

        output_path = self.output_dir / "class_distribution.png"
        plt.savefig(output_path)
        plt.close()
        print(f"[INFO] Class distribution chart saved to {output_path}")

    def save_image_size_distribution(self) -> None:
        """Save histograms showing the distribution of image widths and heights.

        Inconsistent image sizes indicate that preprocessing (resizing)
        is necessary before model training.
        Output saved to: outputs/eda/image_size_distribution.png
        """
        if self.dataframe.empty:
            print("[WARNING] Skipping size distribution: DataFrame is empty.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.histplot(
            self.dataframe["width"],
            bins=30,
            ax=axes[0],
            color="steelblue",
            kde=True,
        )
        axes[0].set_title("Image Width Distribution", fontsize=12)
        axes[0].set_xlabel("Width (pixels)")
        axes[0].set_ylabel("Frequency")

        sns.histplot(
            self.dataframe["height"],
            bins=30,
            ax=axes[1],
            color="coral",
            kde=True,
        )
        axes[1].set_title("Image Height Distribution", fontsize=12)
        axes[1].set_xlabel("Height (pixels)")
        axes[1].set_ylabel("Frequency")

        plt.suptitle("Image Dimension Distributions", fontsize=14)
        plt.tight_layout()

        output_path = self.output_dir / "image_size_distribution.png"
        plt.savefig(output_path)
        plt.close()
        print(f"[INFO] Image size distribution saved to {output_path}")

    def save_class_imbalance_pie(self) -> None:
        """Save a pie chart showing proportional class representation.

        Useful for quickly identifying dominant and minority classes.
        Output saved to: outputs/eda/class_imbalance_pie.png
        """
        if self.dataframe.empty:
            print("[WARNING] Skipping pie chart: DataFrame is empty.")
            return

        counts = self.dataframe["label"].value_counts()

        # Merge small classes into 'Other' for readability
        threshold = counts.sum() * 0.02
        large = counts[counts >= threshold]
        small_total = counts[counts < threshold].sum()

        if small_total > 0:
            large["Other (small classes)"] = small_total

        plt.figure(figsize=(10, 8))
        plt.pie(
            large,
            labels=large.index,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"fontsize": 8},
        )
        plt.title("Class Proportion in Dataset", fontsize=14)
        plt.tight_layout()

        output_path = self.output_dir / "class_imbalance_pie.png"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Class imbalance pie chart saved to {output_path}")

    def save_aspect_ratio_distribution(self) -> None:
        """Save a histogram of image aspect ratios (width / height).

        Helps identify whether images are consistently landscape, portrait,
        or square, which informs how images should be padded or cropped.
        Output saved to: outputs/eda/aspect_ratio_distribution.png
        """
        if self.dataframe.empty:
            print("[WARNING] Skipping aspect ratio: DataFrame is empty.")
            return

        aspect_ratios = self.dataframe["width"] / self.dataframe["height"]

        plt.figure(figsize=(10, 5))
        sns.histplot(aspect_ratios, bins=30, color="mediumseagreen", kde=True)
        plt.axvline(x=1.0, color="red", linestyle="--", label="Square (1:1)")
        plt.title("Image Aspect Ratio Distribution", fontsize=14)
        plt.xlabel("Aspect Ratio (Width / Height)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()

        output_path = self.output_dir / "aspect_ratio_distribution.png"
        plt.savefig(output_path)
        plt.close()
        print(f"[INFO] Aspect ratio distribution saved to {output_path}")

    def save_sample_grid(self) -> None:
        """Save a grid of randomly sampled macroinvertebrate images.

        Provides a quick visual check of image quality, labelling,
        and class variety within the dataset.
        Output saved to: outputs/eda/sample_grid.png
        """
        output_path = self.output_dir / "sample_grid.png"
        save_sample_grid(
            self.dataframe,
            output_path,
            sample_count=SAMPLE_GRID_COUNT,
            random_seed=RANDOM_SEED,
        )

    def save_class_summary_csv(self) -> None:
        """Save the per-class breakdown table as a CSV file.

        Output saved to: outputs/eda/class_summary.csv
        """
        class_summary = self.build_class_summary()
        if class_summary.empty:
            print("[WARNING] Skipping class summary CSV: no data.")
            return

        output_path = self.output_dir / "class_summary.csv"
        class_summary.to_csv(output_path, index=False)
        print(f"[INFO] Class summary CSV saved to {output_path}")

    # ------------------------------------------------------------------
    # Run all outputs at once
    # ------------------------------------------------------------------

    def run_all(self) -> dict[str, float | int]:
        """Generate all EDA outputs and return the summary dictionary.

        This is the main method to call when running the full Stage 1
        pipeline. It produces all charts, saves the CSV, prints the
        summary, and saves a text summary file.

        Returns:
            Summary dictionary from build_summary().
        """
        print("\n[EDA] Starting Stage 1 Exploratory Data Analysis...")
        print(f"[EDA] Output directory: {self.output_dir}\n")

        self.save_class_distribution()
        self.save_image_size_distribution()
        self.save_class_imbalance_pie()
        self.save_aspect_ratio_distribution()
        self.save_sample_grid()
        self.save_class_summary_csv()

        summary = self.build_summary()

        # Save summary as text file
        save_summary_table(
            summary,
            self.output_dir / "dataset_summary.txt",
        )

        # Print summary to console
        print("\n[EDA] Dataset Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key:<25} {value:.2f}")
            else:
                print(f"  {key:<25} {value}")

        print("\n[EDA] Stage 1 complete. All outputs saved.\n")
        return summary
