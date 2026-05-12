# src/models/records.py
# Data model for a single indexed macroinvertebrate image.
# Using a dataclass keeps the structure clean and avoids writing
# boilerplate __init__ and __repr__ methods manually.
#
# Author: Member 1
# Unit: Software Technology 1 (4483/8995)

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImageRecord:
    """Store the core metadata for one indexed macroinvertebrate image.

    Each instance represents a single image file found during dataset
    scanning. The label is derived from the parent folder name, which
    matches the class structure of the Kaggle Stream Macroinvertebrates
    dataset.

    Attributes:
        file_path: Absolute path to the image file on disk.
        label: Class name derived from the parent folder name.
        width: Width of the image in pixels.
        height: Height of the image in pixels.
        channels: Number of colour channels (1 for grayscale, 3 for RGB).
    """

    file_path: Path
    label: str
    width: int
    height: int
    channels: int

    def aspect_ratio(self) -> float:
        """Return the width-to-height aspect ratio of the image.

        Returns:
            Float ratio of width divided by height.
            Returns 0.0 if height is zero to avoid division errors.
        """
        if self.height == 0:
            return 0.0
        return self.width / self.height

    def is_square(self) -> bool:
        """Return True if the image width and height are equal.

        Returns:
            True if width equals height, False otherwise.
        """
        return self.width == self.height

    def pixel_count(self) -> int:
        """Return the total number of pixels in the image.

        Returns:
            Integer product of width and height.
        """
        return self.width * self.height

    def __str__(self) -> str:
        """Return a readable summary of the image record.

        Returns:
            A formatted string with label and image dimensions.
        """
        return (
            f"ImageRecord(label={self.label!r}, "
            f"size={self.width}x{self.height}, "
            f"channels={self.channels})"
        )
