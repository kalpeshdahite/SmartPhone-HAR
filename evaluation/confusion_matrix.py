from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_confusion_matrix(
    y_true,
    y_pred,
    output_path: Path,
    class_names=None,
):
    """Create and save a confusion matrix figure."""
    matrix = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    figure, axis = plt.subplots(figsize=(7, 7))
    display.plot(ax=axis, values_format="d")
    axis.set_title("ResNet-SE-5 — UCI Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

    return matrix
