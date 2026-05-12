# main_eda.py
# Standalone entry point for Member 1's Stage 1 EDA pipeline.
# Run this file directly to index the dataset and generate all
# exploratory data analysis outputs independently of other members.
#
# Usage:
#   python main_eda.py
#
# Prerequisites:
#   1. Download the Kaggle Stream Macroinvertebrates dataset
#   2. Place the unzipped dataset inside:  data/raw/
#      Each class should be in its own subfolder, for example:
#        data/raw/Baetidae/image1.jpg
#        data/raw/Elmidae/image1.jpg
#   3. Install dependencies:  pip install -r requirements.txt
#
# Outputs (saved to outputs/eda/):
#   class_distribution.png       - bar chart of images per class
#   image_size_distribution.png  - histograms of image widths and heights
#   class_imbalance_pie.png      - pie chart of class proportions
#   aspect_ratio_distribution.png - histogram of width-to-height ratios
#   sample_grid.png              - 3x3 grid of sample images
#   class_summary.csv            - per-class count and mean dimensions
#   dataset_summary.txt          - overall dataset statistics
#
# Unit: Software Technology 1 (4483/8995)

from src.config import EDA_OUTPUT_DIR, RAW_DATA_DIR
from src.services.dataset_indexer import DatasetIndexer
from src.services.eda_service import EDAService


def main() -> None:
    """Run the complete Stage 1 EDA pipeline.

    Steps:
        1. Check that the data directory exists and is not empty.
        2. Build the image index using DatasetIndexer.
        3. Run all EDA outputs using EDAService.
        4. Print a final confirmation message.
    """
    print("=" * 60)
    print("  Macroinvertebrate Image Analysis System")
    print("  Stage 1: Exploratory Data Analysis")
    print("=" * 60)
    print(f"\n[INFO] Looking for images in: {RAW_DATA_DIR}")
    print(f"[INFO] EDA outputs will be saved to: {EDA_OUTPUT_DIR}\n")

    # Step 1: Index the dataset
    indexer = DatasetIndexer()
    dataframe = indexer.build_dataframe()

    if dataframe.empty:
        print(
            "\n[ERROR] No images were found in the data directory.\n"
            "Please check that the dataset has been placed inside:\n"
            f"  {RAW_DATA_DIR}\n"
            "Each class should be in its own subfolder, for example:\n"
            "  data/raw/Baetidae/image001.jpg\n"
            "  data/raw/Elmidae/image001.jpg\n"
        )
        return

    # Step 2: Run all EDA outputs
    eda = EDAService(dataframe)
    eda.run_all()

    print("=" * 60)
    print("  Stage 1 complete.")
    print(f"  {len(dataframe)} images indexed.")
    print(f"  {dataframe['label'].nunique()} classes found.")
    print(f"  All outputs saved to: {EDA_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
