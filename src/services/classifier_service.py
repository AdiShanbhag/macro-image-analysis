import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import (
    MODEL_OUTPUT_DIR,
    MODEL_FILENAME,
    RANDOM_STATE,
    TEST_SIZE,
    N_ESTIMATORS,
)
from src.services.image_preprocessor import ImagePreprocessor


class ClassifierService:
    """
    Manage the full classification pipeline for macroinvertebrate images.

    Responsibilities:
        - Convert an indexed image dataframe into feature matrices
        - Train a Random Forest classifier
        - Evaluate the trained model and return results
        - Save and load model artifacts to/from disk

    The preprocessor and model output directory are injected at construction
    so this class can be tested or reconfigured without changing its internals.
    """

    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        model_output_dir: Path = MODEL_OUTPUT_DIR,
    ) -> None:
        """
        Initialise the classifier service.

        Args:
            preprocessor: An ImagePreprocessor instance used to convert
                          raw image files into feature vectors.
            model_output_dir: Directory where trained model artifacts are saved.
        """
        self.preprocessor = preprocessor
        self.model_output_dir = model_output_dir
        self.model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )
        self._is_trained = False

    def prepare_features(
        self, dataframe: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a dataframe of image records into feature and label arrays.

        Uses transform_batch to skip unreadable images and align labels
        to only the successfully processed rows.

        Args:
            dataframe: Must contain columns 'file_path' and 'label'.

        Returns:
            A tuple of (features, labels) as numpy arrays.

        Raises:
            ValueError: If no images could be successfully processed.
        """
        file_paths = dataframe["file_path"].tolist()
        all_labels = dataframe["label"].tolist()

        features, valid_indices = self.preprocessor.transform_batch(file_paths)

        if len(valid_indices) == 0:
            raise ValueError(
                "No images could be processed. Check that RAW_DATA_DIR is correct "
                "and that the dataset has been extracted properly."
            )

        labels = np.array([all_labels[i] for i in valid_indices])

        skipped = len(file_paths) - len(valid_indices)
        if skipped > 0:
            print(f"[INFO] {skipped} image(s) skipped during feature preparation.")

        print(f"[INFO] Features prepared: {features.shape[0]} images, "
              f"{features.shape[1]} features each.")

        return features, labels

    def train(self, dataframe: pd.DataFrame) -> dict[str, object]:
        """
        Train the Random Forest classifier on the indexed image dataset.

        Splits data into train/test sets, fits the model, and returns
        evaluation outputs including accuracy, classification report,
        confusion matrix, and class labels.

        Args:
            dataframe: Indexed image dataframe with 'file_path' and 'label' columns.

        Returns:
            A dictionary with keys:
                - 'accuracy': float
                - 'report': str (classification report)
                - 'confusion_matrix': np.ndarray
                - 'labels': list[str] (sorted unique class names)
        """
        print("[INFO] Preparing features for training...")
        X, y = self.prepare_features(dataframe)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        print(f"[INFO] Training on {len(X_train)} samples, "
              f"testing on {len(X_test)} samples.")
        print("[INFO] Fitting Random Forest classifier — this may take a minute...")

        self.model.fit(X_train, y_train)
        self._is_trained = True

        predictions = self.model.predict(X_test)
        labels = sorted(list(np.unique(y)))

        results = {
            "accuracy": accuracy_score(y_test, predictions),
            "report": classification_report(y_test, predictions, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels),
            "labels": labels,
        }

        print(f"[INFO] Training complete. Accuracy: {results['accuracy']:.4f}")

        return results

    def predict(self, file_path: str | Path) -> dict[str, object]:
        """
        Predict the class of a single image.

        Args:
            file_path: Path to the image file.

        Returns:
            A dictionary with keys:
                - 'predicted_class': str
                - 'confidence': float (probability of the top class)

        Raises:
            RuntimeError: If the model has not been trained or loaded yet.
        """
        if not self._is_trained:
            raise RuntimeError(
                "Model is not trained. Run train() or load_model() before predicting."
            )

        features = self.preprocessor.transform(file_path).reshape(1, -1)
        predicted_class = self.model.predict(features)[0]

        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            confidence = float(self.model.predict_proba(features).max())

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
        }

    def save_model(self, filename: str = MODEL_FILENAME) -> Path:
        """
        Save the trained model artifact to disk.

        Args:
            filename: Name of the output file. Defaults to MODEL_FILENAME from config.

        Returns:
            The full path where the model was saved.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save an untrained model.")

        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.model_output_dir / filename
        joblib.dump(self.model, output_path)
        print(f"[INFO] Model saved to {output_path}")

        return output_path

    def load_model(self, filename: str = MODEL_FILENAME) -> None:
        """
        Load a previously saved model artifact from disk.

        Args:
            filename: Name of the model file to load.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        model_path = self.model_output_dir / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"No saved model found at {model_path}. "
                "Train the model first using train()."
            )

        self.model = joblib.load(model_path)
        self._is_trained = True
        print(f"[INFO] Model loaded from {model_path}")