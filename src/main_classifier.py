"""
main_classifier.py

Member 2 standalone entry point.

Runs the full training pipeline independently:
    1. Scans the raw dataset folder
    2. Builds an image index
    3. Preprocesses images and trains the Random Forest classifier
    4. Saves the trained model to outputs/models/
    5. Saves evaluation outputs to outputs/reports/

Usage:
    python -m src.main_classifier

Note:
    This script uses its own lightweight dataset scanner so it has no
    dependency on Member 1's DatasetIndexer during development. Once the
    group integrates, the WorkflowService will coordinate both.
"""

from pathlib import Path
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    MODEL_OUTPUT_DIR,
    REPORTS_OUTPUT_DIR,
    SUPPORTED_EXTENSIONS,
)
from src.services.image_preprocessor import ImagePreprocessor
from src.services.classifier_service import ClassifierService
from src.services.evaluation_utils import (
    save_classification_report,
    save_confusion_matrix_plot,
    save_accuracy_summary,
)


def scan_dataset(data_dir: Path) -> pd.DataFrame:
    """
    Lightweight dataset scanner for standalone use by Member 2.

    Recursively finds all supported image files and uses the parent
    folder name as the class label.

    Args:
        data_dir: Root directory of the raw dataset.

    Returns:
        A dataframe with columns 'file_path' and 'label'.

    Raises:
        FileNotFoundError: If the data directory does not exist.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            "Make sure you have extracted the Kaggle dataset into data/raw/"
        )

    records = []
    for file_path in data_dir.rglob("*"):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            records.append({
                "file_path": str(file_path),
                "label": file_path.parent.name,
            })

    if not records:
        raise ValueError(
            f"No supported images found in {data_dir}. "
            f"Supported extensions: {SUPPORTED_EXTENSIONS}"
        )

    df = pd.DataFrame(records)
    print(f"[INFO] Dataset scanned: {len(df)} images across "
          f"{df['label'].nunique()} classes.")

    return df


def main() -> None:
    """Run the full Member 2 training and evaluation pipeline."""

    print("=" * 60)
    print("Macroinvertebrate Classifier — Training Pipeline")
    print("=" * 60)

    # Step 1: Scan dataset
    print("\n[STEP 1] Scanning dataset...")
    dataframe = scan_dataset(RAW_DATA_DIR)

    # Step 2: Initialise services
    print("\n[STEP 2] Initialising preprocessor and classifier...")
    preprocessor = ImagePreprocessor()
    classifier = ClassifierService(preprocessor, MODEL_OUTPUT_DIR)

    # Step 3: Train
    print("\n[STEP 3] Training classifier...")
    results = classifier.train(dataframe)

    # Step 4: Save model
    print("\n[STEP 4] Saving model...")
    classifier.save_model()

    # Step 5: Save evaluation outputs
    print("\n[STEP 5] Saving evaluation outputs...")
    save_classification_report(results, REPORTS_OUTPUT_DIR)
    save_confusion_matrix_plot(results, REPORTS_OUTPUT_DIR)
    save_accuracy_summary(results, REPORTS_OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"Pipeline complete.")
    print(f"  Accuracy : {results['accuracy']:.4f}")
    print(f"  Model    : {MODEL_OUTPUT_DIR / 'macro_classifier.joblib'}")
    print(f"  Reports  : {REPORTS_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()