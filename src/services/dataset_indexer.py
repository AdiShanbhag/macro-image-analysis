# src/services/dataset_indexer.py
# Scans the raw dataset folder and builds a structured Pandas DataFrame
# containing one row per image with its file path, label, and dimensions.
#
# The label for each image is taken from its immediate parent folder name.
# This matches the Kaggle Stream Macroinvertebrates dataset structure where
# each class has its own subfolder.
#
# Author: Member 1
# Unit: Software Technology 1 (4483/8995)

from pathlib import Path

import cv2
import pandas as pd

from src.config import RAW_DATA_DIR, SUPPORTED_EXTENSIONS
from src.models.records import ImageRecord


class DatasetIndexer:
    """Scan the dataset folder and build a tabular image index.

    This class is responsible for discovering all valid image files inside
    the raw data directory, reading their dimensions using OpenCV, and
    returning the results as a Pandas DataFrame suitable for EDA and
    model training.

    Attributes:
        data_dir: Path to the root folder containing class subfolders.
    """

    def __init__(self, data_dir: Path = RAW_DATA_DIR) -> None:
        """Initialise the indexer with a target data directory.

        Args:
            data_dir: Path to scan for image files.
                      Defaults to RAW_DATA_DIR from config.
        """
        self.data_dir = data_dir

    def _is_valid_image(self, file_path: Path) -> bool:
        """Check whether a file has a supported image extension.

        Args:
            file_path: Path object for the file to check.

        Returns:
            True if the file extension is in SUPPORTED_EXTENSIONS.
        """
        return file_path.suffix.lower() in SUPPORTED_EXTENSIONS

    def _read_image_record(self, file_path: Path) -> ImageRecord | None:
        """Attempt to read one image file and return an ImageRecord.

        Uses OpenCV to open the image and extract width, height, and
        channel count. Returns None if the image cannot be read.

        Args:
            file_path: Path to the image file.

        Returns:
            An ImageRecord instance if the image is readable, else None.
        """
        image = cv2.imread(str(file_path))
        if image is None:
            return None

        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        label = file_path.parent.name

        return ImageRecord(
            file_path=file_path,
            label=label,
            width=width,
            height=height,
            channels=channels,
        )

    def build_records(self) -> list[ImageRecord]:
        """Scan the data directory and return a list of ImageRecord objects.

        Recursively searches for all files matching the supported
        extensions. Files that cannot be read by OpenCV are skipped
        with a warning printed to the console.

        Returns:
            A list of ImageRecord instances for all readable image files.
        """
        records: list[ImageRecord] = []

        if not self.data_dir.exists():
            print(f"[WARNING] Data directory not found: {self.data_dir}")
            return records

        for file_path in sorted(self.data_dir.rglob("*")):
            if not self._is_valid_image(file_path):
                continue

            record = self._read_image_record(file_path)
            if record is None:
                print(f"[WARNING] Could not read image: {file_path}")
                continue

            records.append(record)

        print(f"[INFO] Indexed {len(records)} images from {self.data_dir}")
        return records

    def build_dataframe(self) -> pd.DataFrame:
        """Build and return a DataFrame from all indexed image records.

        Each row represents one image. Columns are: file_path, label,
        width, height, and channels.

        Returns:
            A Pandas DataFrame with one row per valid image file.
            Returns an empty DataFrame if no valid images are found.
        """
        records = self.build_records()

        if not records:
            print("[WARNING] No valid images found. Returning empty DataFrame.")
            return pd.DataFrame(
                columns=["file_path", "label", "width", "height", "channels"]
            )

        rows = [
            {
                "file_path": str(record.file_path),
                "label": record.label,
                "width": record.width,
                "height": record.height,
                "channels": record.channels,
            }
            for record in records
        ]

        dataframe = pd.DataFrame(rows)
        print(
            f"[INFO] DataFrame built with {len(dataframe)} rows "
            f"and {dataframe['label'].nunique()} unique classes."
        )
        return dataframe

    def save_index(self, output_path: Path) -> None:
        """Build the DataFrame and save it to a CSV file.

        Args:
            output_path: Path where the CSV file should be saved.
        """
        dataframe = self.build_dataframe()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)
        print(f"[INFO] Index saved to {output_path}")
