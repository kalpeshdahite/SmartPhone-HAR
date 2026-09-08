import sys
from pathlib import Path

import json
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import METRICS_DIR, RESULTS_DIR


def main():
    metrics_path = (
        METRICS_DIR / "resnet_se5_uci_supervised.json"
    )

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}\n"
            "Run scripts/run_supervised.py first."
        )

    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    history = metrics["history"]
    epochs = range(1, len(history["train_loss"]) + 1)

    output_dir = RESULTS_DIR / "learning_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, history["train_loss"], label="Train Loss")
    axis.plot(epochs, history["val_loss"], label="Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("ResNet-SE-5 — Training and Validation Loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "resnet_se5_uci_loss.png",
        dpi=200,
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        epochs,
        history["train_accuracy"],
        label="Train Accuracy",
    )
    axis.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.set_title("ResNet-SE-5 — Training and Validation Accuracy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "resnet_se5_uci_accuracy.png",
        dpi=200,
    )
    plt.close(figure)

    print("Learning curves saved to:", output_dir)


if __name__ == "__main__":
    main()
