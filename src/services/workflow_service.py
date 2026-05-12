"""
workflow_service.py

Member 3 — Deployed Application.

This module is the coordination layer for the deployed Tkinter application.
It is intentionally self-contained: it ships its own minimal DatasetIndexer
and ImagePreprocessor so the app can run end-to-end without depending on
Member 1's or Member 2's source files. It only relies on their *artifacts*
(saved model in outputs/models/, EDA charts in outputs/eda/) when those
artifacts exist, and can regenerate either side if they do not.

The preprocessing here matches the (128, 128) grayscale-and-flatten pipeline
documented in src/config.py and used by Member 2's ImagePreprocessor, so a
model trained anywhere in the team produces compatible inputs at predict
time.

Public class:
    WorkflowService — coordinates dataset summary, EDA generation, training,
                      prediction, and chart discovery for the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import joblib
import matplotlib

# Use a non-interactive backend for chart generation so this works
# whether or not a display is attached when EDA runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import (
    EDA_OUTPUT_DIR,
    IMAGE_SIZE,
    MODEL_FILENAME,
    MODEL_OUTPUT_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
    SUPPORTED_EXTENSIONS,
    TEST_SIZE,
)


# ---------------------------------------------------------------------------
# Lightweight self-contained components
# ---------------------------------------------------------------------------

@dataclass
class ImageRecord:
    """Minimal record describing one indexed image on disk.

    Kept simple on purpose — the deployed app only needs path, label, and
    dimensions for summary / EDA. This mirrors the structure expected by
    Member 1's richer model so anything we produce here is drop-in compatible.
    """

    file_path: str
    label: str
    width: int
    height: int
    channels: int


class _LocalDatasetIndexer:
    """Self-contained dataset scanner used by the deployed app.

    Walks the raw data directory recursively, treats each immediate parent
    folder as the class label, and emits a tidy ``pandas.DataFrame``. Files
    that cannot be opened by OpenCV are silently skipped so a few corrupt
    samples never break the whole scan.

    This is intentionally a duplicate of Member 1's DatasetIndexer interface
    so the deployment never breaks if Member 1 has not yet pushed their work.
    """

    def __init__(self, data_dir: Path = RAW_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)

    def build_dataframe(self) -> pd.DataFrame:
        """Return one row per indexed image with file path, label, and size.

        Returns:
            DataFrame with columns: file_path, label, width, height, channels.

        Raises:
            FileNotFoundError: If the configured raw data directory is
                missing — the user is told exactly where to put the dataset.
            ValueError: If the directory exists but contains no supported
                images.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.data_dir}\n"
                "Place the extracted Kaggle dataset under data/raw/ "
                "so that each class has its own subfolder."
            )

        records: list[dict] = []
        for file_path in self.data_dir.rglob("*"):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            image = cv2.imread(str(file_path))
            if image is None:
                # Unreadable file — skip rather than crash the whole scan.
                continue

            height, width = image.shape[:2]
            channels = image.shape[2] if image.ndim == 3 else 1

            records.append(
                {
                    "file_path": str(file_path),
                    "label": file_path.parent.name,
                    "width": int(width),
                    "height": int(height),
                    "channels": int(channels),
                }
            )

        if not records:
            raise ValueError(
                f"No supported images found in {self.data_dir}. "
                f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        return pd.DataFrame(records)


class _LocalImagePreprocessor:
    """Self-contained preprocessor mirroring the team's baseline pipeline.

    Loads each image in grayscale, resizes to ``image_size``, normalises to
    [0, 1] and flattens. Identical feature shape to Member 2's
    ``ImagePreprocessor``, which is what keeps Member 2's saved model usable
    here without any wrapper code.
    """

    def __init__(self, image_size: tuple[int, int] = IMAGE_SIZE) -> None:
        self.image_size = image_size

    def transform(self, file_path: str | Path) -> np.ndarray:
        """Load → grayscale → resize → normalise → flatten.

        Args:
            file_path: Path to an image file.

        Returns:
            1-D float32 numpy array of length ``image_size[0] * image_size[1]``.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the file exists but OpenCV cannot decode it.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(
                f"Could not read image (unsupported format or corrupt file): {path}"
            )

        resized = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        normalised = resized.astype("float32") / 255.0
        return normalised.flatten()


# ---------------------------------------------------------------------------
# WorkflowService — the single coordination layer the GUI talks to
# ---------------------------------------------------------------------------

class WorkflowService:
    """Coordinate dataset, EDA, training, and prediction for the GUI.

    The service is the only thing the Tkinter app talks to. That keeps the
    UI thin and lets us swap the underlying implementation later — for
    example, calling Member 2's ``ClassifierService`` directly — without
    touching ``app.py``.

    Public surface:
        - is_dataset_available()
        - is_model_available()
        - load_dataframe() / show_summary()
        - generate_eda()
        - list_eda_charts()
        - train_model()
        - predict_image(file_path)
    """

    def __init__(
        self,
        data_dir: Path = RAW_DATA_DIR,
        eda_output_dir: Path = EDA_OUTPUT_DIR,
        model_output_dir: Path = MODEL_OUTPUT_DIR,
        model_filename: str = MODEL_FILENAME,
        image_size: tuple[int, int] = IMAGE_SIZE,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.eda_output_dir = Path(eda_output_dir)
        self.model_output_dir = Path(model_output_dir)
        self.model_filename = model_filename

        self.eda_output_dir.mkdir(parents=True, exist_ok=True)
        self.model_output_dir.mkdir(parents=True, exist_ok=True)

        self.indexer = _LocalDatasetIndexer(self.data_dir)
        self.preprocessor = _LocalImagePreprocessor(image_size)

        self._dataframe: Optional[pd.DataFrame] = None
        self._model = None  # populated on first successful load / train

    # ---------- Availability checks (used by the GUI to disable buttons) ----

    def is_dataset_available(self) -> bool:
        """True when the configured raw data folder has at least one image."""
        if not self.data_dir.exists():
            return False
        for path in self.data_dir.rglob("*"):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                return True
        return False

    @property
    def model_path(self) -> Path:
        """Full path to the saved model artifact this service will load."""
        return self.model_output_dir / self.model_filename

    def is_model_available(self) -> bool:
        """True when a saved model artifact exists on disk."""
        return self.model_path.exists()

    # ---------- Dataset + summary --------------------------------------------

    def load_dataframe(self, force_refresh: bool = False) -> pd.DataFrame:
        """Index the dataset once and cache the resulting DataFrame.

        Args:
            force_refresh: When True, rebuild the index even if a cached copy
                exists. Useful after the user adds new images to data/raw/.
        """
        if self._dataframe is None or force_refresh:
            self._dataframe = self.indexer.build_dataframe()
        return self._dataframe

    def show_summary(self) -> dict:
        """Return key dataset statistics for display in the GUI status panel."""
        dataframe = self.load_dataframe()
        return {
            "total_images": int(len(dataframe)),
            "total_classes": int(dataframe["label"].nunique()),
            "mean_width": float(dataframe["width"].mean()),
            "mean_height": float(dataframe["height"].mean()),
            "images_per_class": dataframe["label"].value_counts().to_dict(),
        }

    # ---------- EDA -----------------------------------------------------------

    def generate_eda(self) -> list[Path]:
        """Produce the standard EDA chart set into ``outputs/eda/``.

        Generates a class distribution bar chart, a width/height histogram
        pair, and a 3x3 sample grid. Returns the list of saved file paths so
        the GUI can preview them straight away.
        """
        dataframe = self.load_dataframe()
        self.eda_output_dir.mkdir(parents=True, exist_ok=True)

        produced: list[Path] = []
        produced.append(self._save_class_distribution(dataframe))
        produced.append(self._save_image_size_distribution(dataframe))
        produced.append(self._save_sample_grid(dataframe))
        return produced

    def list_eda_charts(self) -> list[Path]:
        """Return every PNG chart currently sitting in ``outputs/eda/``.

        The GUI uses this to populate its chart picker. Anything generated
        by Member 1's EDAService also shows up here automatically because
        we only look at the filesystem, not at who wrote the files.
        """
        if not self.eda_output_dir.exists():
            return []
        return sorted(self.eda_output_dir.glob("*.png"))

    def _save_class_distribution(self, dataframe: pd.DataFrame) -> Path:
        """Bar chart of image counts per class, sorted descending."""
        plt.figure(figsize=(12, 6))
        order = dataframe["label"].value_counts().index
        sns.countplot(data=dataframe, x="label", order=order)
        plt.xticks(rotation=90)
        plt.title("Macroinvertebrate Images per Class")
        plt.tight_layout()
        output_path = self.eda_output_dir / "class_distribution.png"
        plt.savefig(output_path, dpi=120)
        plt.close()
        return output_path

    def _save_image_size_distribution(self, dataframe: pd.DataFrame) -> Path:
        """Side-by-side histograms of image width and height."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(dataframe["width"], bins=20, ax=axes[0])
        sns.histplot(dataframe["height"], bins=20, ax=axes[1])
        axes[0].set_title("Image Width Distribution")
        axes[1].set_title("Image Height Distribution")
        plt.tight_layout()
        output_path = self.eda_output_dir / "image_size_distribution.png"
        plt.savefig(output_path, dpi=120)
        plt.close()
        return output_path

    def _save_sample_grid(
        self, dataframe: pd.DataFrame, sample_count: int = 9
    ) -> Path:
        """3x3 grid of randomly sampled images with their class labels."""
        sample_df = dataframe.sample(
            min(sample_count, len(dataframe)), random_state=RANDOM_STATE
        )
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))

        for ax, (_, row) in zip(axes.flat, sample_df.iterrows()):
            image = cv2.imread(row["file_path"])
            if image is None:
                ax.axis("off")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            ax.set_title(row["label"], fontsize=9)
            ax.axis("off")

        for ax in axes.flat[len(sample_df):]:
            ax.axis("off")

        plt.tight_layout()
        output_path = self.eda_output_dir / "sample_grid.png"
        plt.savefig(output_path, dpi=120)
        plt.close()
        return output_path

    # ---------- Training ------------------------------------------------------

    def train_model(self) -> dict:
        """Train a baseline Random Forest, persist it, and return metrics.

        Uses the self-contained preprocessor so the feature shape is
        guaranteed to match what ``predict_image`` will produce later. Any
        unreadable image is skipped rather than aborting the run.

        Returns:
            Dict containing ``accuracy``, ``report`` (str), ``confusion_matrix``
            (np.ndarray), ``labels`` (list[str]), and ``model_path`` (str).
        """
        # Import lazily so the GUI can import this module without sklearn
        # installed for users who only want to run prediction with a
        # pre-shipped model.
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
        )
        from sklearn.model_selection import train_test_split

        dataframe = self.load_dataframe()

        features: list[np.ndarray] = []
        labels: list[str] = []
        skipped = 0
        for _, row in dataframe.iterrows():
            try:
                features.append(self.preprocessor.transform(row["file_path"]))
                labels.append(row["label"])
            except (FileNotFoundError, ValueError):
                skipped += 1

        if not features:
            raise ValueError(
                "No images could be processed for training. "
                "Check that data/raw/ contains valid images."
            )

        X = np.array(features, dtype="float32")
        y = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        ordered_labels = sorted(np.unique(y).tolist())

        # Persist model so subsequent app launches can predict immediately.
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        self._model = model

        return {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "report": classification_report(y_test, predictions, zero_division=0),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=ordered_labels
            ),
            "labels": ordered_labels,
            "model_path": str(self.model_path),
            "skipped_images": skipped,
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        }

    # ---------- Prediction ---------------------------------------------------

    def _ensure_model_loaded(self):
        """Load the saved model from disk on first prediction.

        Raises:
            FileNotFoundError: If no saved model exists at all. The GUI
                catches this and prompts the user to train or to point at a
                different ``outputs/models/`` folder.
        """
        if self._model is None:
            if not self.is_model_available():
                raise FileNotFoundError(
                    f"No trained model found at {self.model_path}. "
                    "Train the model first from the Training tab."
                )
            self._model = joblib.load(self.model_path)
        return self._model

    def predict_image(self, file_path: str | Path) -> dict:
        """Predict the class of one image and return prediction + confidence.

        Args:
            file_path: Path to an image file on disk.

        Returns:
            Dict with ``predicted_class`` (str) and ``confidence`` (float
            in [0, 1]). Confidence falls back to 0.0 for models that do not
            expose ``predict_proba``.
        """
        model = self._ensure_model_loaded()
        features = self.preprocessor.transform(file_path).reshape(1, -1)
        predicted = model.predict(features)[0]

        confidence = 0.0
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(features).max())

        return {
            "predicted_class": str(predicted),
            "confidence": confidence,
        }
