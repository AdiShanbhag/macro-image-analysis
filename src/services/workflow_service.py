"""
workflow_service.py

Member 3 — Deployed Application.

Coordination layer for the deployed Tkinter application. Self-contained:
ships its own minimal DatasetIndexer and ImagePreprocessor so the app runs
end-to-end without depending on Member 1's or Member 2's source files at
import time. It only relies on their *artifacts* (saved model in
outputs/models/, EDA charts in outputs/eda/) when those artifacts exist,
and can regenerate either side if they do not.

The preprocessing pipeline matches the (128, 128) grayscale-and-flatten
pipeline in src/config.py and Member 2's ImagePreprocessor, so a model
trained anywhere in the team produces compatible inputs at predict time.

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
    REPORTS_OUTPUT_DIR,
    SUPPORTED_EXTENSIONS,
    TEST_SIZE,
)


# ---------------------------------------------------------------------------
# Lightweight self-contained components
# ---------------------------------------------------------------------------

@dataclass
class ImageRecord:
    """Minimal record describing one indexed image on disk."""

    file_path: str
    label: str
    width: int
    height: int
    channels: int


class _LocalDatasetIndexer:
    """Self-contained dataset scanner used by the deployed app."""

    def __init__(self, data_dir: Path = RAW_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)

    def build_dataframe(self) -> pd.DataFrame:
        """Return one row per indexed image with file path, label, and size.

        Raises:
            FileNotFoundError: If the raw data directory is missing.
            ValueError: If the directory exists but contains no supported images.
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
                continue
            height, width = image.shape[:2]
            channels = image.shape[2] if image.ndim == 3 else 1
            records.append({
                "file_path": str(file_path),
                "label": file_path.parent.name,
                "width": int(width),
                "height": int(height),
                "channels": int(channels),
            })

        if not records:
            raise ValueError(
                f"No supported images found in {self.data_dir}. "
                f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        return pd.DataFrame(records)


class _LocalImagePreprocessor:
    """Self-contained preprocessor mirroring the team's baseline pipeline."""

    def __init__(self, image_size: tuple[int, int] = IMAGE_SIZE) -> None:
        self.image_size = image_size

    def transform(self, file_path: str | Path) -> np.ndarray:
        """Load, grayscale, resize, normalise, and flatten one image.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If OpenCV cannot decode the file.
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
# WorkflowService
# ---------------------------------------------------------------------------

class WorkflowService:
    """Coordinate dataset, EDA, training, and prediction for the GUI.

    The GUI talks only to this class. All underlying logic is encapsulated
    here so the UI stays thin and testable.

    Public surface:
        - list_available_classes()
        - is_dataset_available()
        - is_model_available()
        - load_dataframe() / show_summary()
        - generate_eda()
        - list_eda_charts()
        - train_model(selected_classes)
        - predict_image(file_path)
    """

    def __init__(
        self,
        data_dir: Path = RAW_DATA_DIR,
        eda_output_dir: Path = EDA_OUTPUT_DIR,
        model_output_dir: Path = MODEL_OUTPUT_DIR,
        reports_output_dir: Path = REPORTS_OUTPUT_DIR,
        model_filename: str = MODEL_FILENAME,
        image_size: tuple[int, int] = IMAGE_SIZE,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.eda_output_dir = Path(eda_output_dir)
        self.model_output_dir = Path(model_output_dir)
        self.reports_output_dir = Path(reports_output_dir)
        self.model_filename = model_filename

        self.eda_output_dir.mkdir(parents=True, exist_ok=True)
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_output_dir.mkdir(parents=True, exist_ok=True)

        self.indexer = _LocalDatasetIndexer(self.data_dir)
        self.preprocessor = _LocalImagePreprocessor(image_size)

        self._dataframe: Optional[pd.DataFrame] = None
        self._model = None

    # ---------- Class discovery ---------------------------------------------

    def list_available_classes(self) -> list[str]:
        """Return sorted list of class folder names found in data/raw/.

        Only folders that contain at least one supported image are included.
        Returns an empty list if the data directory does not exist.
        """
        if not self.data_dir.exists():
            return []

        classes = []
        for folder in self.data_dir.iterdir():
            if not folder.is_dir():
                continue
            has_image = any(
                f.suffix.lower() in SUPPORTED_EXTENSIONS
                for f in folder.iterdir()
                if f.is_file()
            )
            if has_image:
                classes.append(folder.name)

        return sorted(classes)

    # ---------- Availability checks -----------------------------------------

    def is_dataset_available(self) -> bool:
        """True when the raw data folder has at least one supported image."""
        if not self.data_dir.exists():
            return False
        for path in self.data_dir.rglob("*"):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                return True
        return False

    @property
    def model_path(self) -> Path:
        """Full path to the saved model artifact."""
        return self.model_output_dir / self.model_filename

    def is_model_available(self) -> bool:
        """True when a saved model artifact exists on disk."""
        return self.model_path.exists()

    # ---------- Dataset + summary -------------------------------------------

    def load_dataframe(self, force_refresh: bool = False) -> pd.DataFrame:
        """Index the dataset once and cache the result.

        Args:
            force_refresh: Rebuild the index even if a cached copy exists.
        """
        if self._dataframe is None or force_refresh:
            self._dataframe = self.indexer.build_dataframe()
        return self._dataframe

    def show_summary(self) -> dict:
        """Return key dataset statistics for display in the GUI."""
        dataframe = self.load_dataframe()
        return {
            "total_images": int(len(dataframe)),
            "total_classes": int(dataframe["label"].nunique()),
            "mean_width": float(dataframe["width"].mean()),
            "mean_height": float(dataframe["height"].mean()),
            "images_per_class": dataframe["label"].value_counts().to_dict(),
        }

    # ---------- EDA ---------------------------------------------------------

    def generate_eda(self) -> list[Path]:
        """Produce the standard EDA chart set into outputs/eda/.

        Returns the list of saved file paths so the GUI can preview them.
        """
        dataframe = self.load_dataframe()
        self.eda_output_dir.mkdir(parents=True, exist_ok=True)

        produced: list[Path] = []
        produced.append(self._save_class_distribution(dataframe))
        produced.append(self._save_image_size_distribution(dataframe))
        produced.append(self._save_sample_grid(dataframe))
        return produced

    def list_eda_charts(self) -> list[Path]:
        """Return every PNG chart currently in outputs/eda/."""
        if not self.eda_output_dir.exists():
            return []
        return sorted(self.eda_output_dir.glob("*.png"))

    def _save_class_distribution(self, dataframe: pd.DataFrame) -> Path:
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

    def _save_sample_grid(self, dataframe: pd.DataFrame, sample_count: int = 9) -> Path:
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

    # ---------- Training ----------------------------------------------------

    def train_model(self, selected_classes: list[str] | None = None) -> dict:
        """Train a Random Forest on selected class folders and save artifacts.

        Args:
            selected_classes: List of class folder names to train on.
                              If None or empty, trains on all available classes.

        Returns:
            Dict with accuracy, report, confusion_matrix, labels, model_path,
            skipped_images, training_samples, test_samples.

        Raises:
            ValueError: If no images can be processed for the selected classes.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
        )
        from sklearn.model_selection import train_test_split

        dataframe = self.load_dataframe()

        # Filter to selected classes if specified
        if selected_classes:
            dataframe = dataframe[
                dataframe["label"].isin(selected_classes)
            ].reset_index(drop=True)

            if dataframe.empty:
                raise ValueError(
                    "No images found for the selected classes. "
                    "Check that the selected folders contain supported images."
                )

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
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
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

        # Save model
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        self._model = model

        cm = confusion_matrix(y_test, predictions, labels=ordered_labels)

        # Save confusion matrix as PNG for GUI display
        self._save_confusion_matrix_plot(cm, ordered_labels)

        # Save classification report as text
        report_str = classification_report(y_test, predictions, zero_division=0)
        self._save_classification_report(
            accuracy_score(y_test, predictions), report_str
        )

        return {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "report": report_str,
            "confusion_matrix": cm,
            "labels": ordered_labels,
            "model_path": str(self.model_path),
            "skipped_images": skipped,
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        }

    def _save_confusion_matrix_plot(
        self, matrix: np.ndarray, labels: list[str]
    ) -> Path:
        """Save confusion matrix heatmap to outputs/reports/."""
        self.reports_output_dir.mkdir(parents=True, exist_ok=True)

        display_labels = [
            label[:15] + "..." if len(label) > 15 else label
            for label in labels
        ]

        fig_size = max(10, len(labels))
        plt.figure(figsize=(fig_size, fig_size - 2))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=display_labels,
            yticklabels=display_labels,
            linewidths=0.5,
        )
        plt.title("Confusion Matrix — Macroinvertebrate Classifier", fontsize=13, pad=15)
        plt.xlabel("Predicted Class", fontsize=11)
        plt.ylabel("Actual Class", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        output_path = self.reports_output_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

    def _save_classification_report(self, accuracy: float, report: str) -> Path:
        """Save the classification report text to outputs/reports/."""
        self.reports_output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.reports_output_dir / "classification_report.txt"
        content = f"Model Accuracy: {accuracy:.4f}\n{'=' * 60}\n\n{report}"
        report_path.write_text(content, encoding="utf-8")
        return report_path

    # ---------- Prediction --------------------------------------------------

    def _ensure_model_loaded(self):
        """Load the saved model from disk on first prediction.

        Raises:
            FileNotFoundError: If no saved model exists.
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
        """Predict the class of one image.

        Args:
            file_path: Path to an image file on disk.

        Returns:
            Dict with predicted_class (str) and confidence (float in [0, 1]).
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