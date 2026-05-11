import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def save_classification_report(results: dict, output_dir: Path) -> Path:
    """
    Write the classification report string to a text file.

    Args:
        results: Output dictionary from ClassifierService.train().
        output_dir: Directory to save the report in.

    Returns:
        Path to the saved report file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "classification_report.txt"

    content = (
        f"Model Accuracy: {results['accuracy']:.4f}\n"
        f"{'=' * 60}\n\n"
        f"{results['report']}"
    )

    report_path.write_text(content, encoding="utf-8")
    print(f"[INFO] Classification report saved to {report_path}")

    return report_path


def save_confusion_matrix_plot(results: dict, output_dir: Path) -> Path:
    """
    Save a confusion matrix heatmap as a PNG image.

    Uses abbreviated labels if there are many classes to keep the
    chart readable during presentation.

    Args:
        results: Output dictionary from ClassifierService.train().
        output_dir: Directory to save the plot in.

    Returns:
        Path to the saved plot file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = results["labels"]
    matrix = results["confusion_matrix"]

    # Abbreviate long class names for readability on the chart
    display_labels = [label[:15] + "..." if len(label) > 15 else label for label in labels]

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

    plt.title("Confusion Matrix — Macroinvertebrate Classifier", fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("Actual Class", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    output_path = output_dir / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[INFO] Confusion matrix saved to {output_path}")

    return output_path


def save_accuracy_summary(results: dict, output_dir: Path) -> Path:
    """
    Save a simple bar chart showing per-class precision from the
    classification report as a quick visual summary.

    Args:
        results: Output dictionary from ClassifierService.train().
        output_dir: Directory to save the chart in.

    Returns:
        Path to the saved chart file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse per-class precision from the report string
    lines = results["report"].strip().split("\n")
    class_names = []
    precisions = []

    for line in lines:
        parts = line.split()
        # Valid class rows have at least 5 parts: name, precision, recall, f1, support
        if len(parts) >= 5:
            try:
                precision = float(parts[-4])
                class_name = " ".join(parts[:-4])
                # Skip summary rows
                if class_name not in ("accuracy", "macro avg", "weighted avg"):
                    class_names.append(class_name[:20])
                    precisions.append(precision)
            except ValueError:
                continue

    if not class_names:
        print("[WARNING] Could not parse per-class data for accuracy summary chart.")
        return None

    x = np.arange(len(class_names))
    plt.figure(figsize=(max(10, len(class_names)), 6))
    bars = plt.bar(x, precisions, color="steelblue", edgecolor="white")
    plt.xticks(x, class_names, rotation=45, ha="right", fontsize=8)
    plt.ylim(0, 1.1)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Per-Class Precision — Macroinvertebrate Classifier", fontsize=13)

    for bar, val in zip(bars, precisions):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    plt.tight_layout()
    output_path = output_dir / "per_class_precision.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[INFO] Per-class precision chart saved to {output_path}")

    return output_path