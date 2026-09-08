import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)


def calculate_metrics(y_true, y_pred):
    """Calculate standard multiclass classification metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }


def save_classification_report(y_true, y_pred, output_path: Path):
    """Save the full sklearn classification report as JSON."""
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return report
