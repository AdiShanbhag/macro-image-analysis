import cv2
import numpy as np
from pathlib import Path

class ImagePreprocessor:
    """
    Convert raw macroinvertebrate images into normalised numeric features
    suitable for model training and prediction.

    Reads images in grayscale, resizes them to a fixed dimension,
    normalises pixel values to the range [0, 1], and flattens the result
    into a 1D feature vector.
    """

    def __init__(self, image_size: tuple[int, int] = (128, 128)) -> None:
        """
        Initialise the preprocessor with a target image size.

        Args:
            image_size: Target (width, height) to resize all images to.
                        Defaults to (128, 128).
        """
        self.image_size = image_size

    def transform(self, file_path: str | Path) -> np.ndarray:
        """
        Load, resize, normalise, and flatten a single image.

        Args:
            file_path: Path to the image file (PNG, JPG, BMP supported).

        Returns:
            A 1D float32 numpy array of length image_size[0] * image_size[1].

        Raises:
            FileNotFoundError: If the file does not exist at the given path.
            ValueError: If the file exists but cannot be read as an image.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Could not read image (unsupported format or corrupt file): {path}")

        resized = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        normalised = resized.astype("float32") / 255.0

        return normalised.flatten()

    def transform_batch(self, file_paths: list[str | Path]) -> np.ndarray:
        """
        Transform a list of image paths into a 2D feature matrix.

        Skips and logs any image that fails to load rather than stopping
        the entire batch. This handles corrupt or missing files in the
        dataset gracefully.

        Args:
            file_paths: List of paths to image files.

        Returns:
            A 2D float32 numpy array of shape (n_valid_images, n_features).
            Also returns the list of valid indices so callers can align labels.
        """
        features = []
        valid_indices = []

        for i, path in enumerate(file_paths):
            try:
                features.append(self.transform(path))
                valid_indices.append(i)
            except (FileNotFoundError, ValueError) as e:
                print(f"[WARNING] Skipping image at index {i}: {e}")

        return np.array(features, dtype="float32"), valid_indices