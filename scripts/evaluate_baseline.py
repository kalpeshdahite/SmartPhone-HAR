import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    MODEL_DIR,
    METRICS_DIR,
    CONFUSION_MATRIX_DIR,
)

from preprocessing.load_data import load_daghar_splits
from preprocessing.preprocess import dataframe_to_numpy
from preprocessing.dataset import HARDataset
from models.resnet_se import ResNetSE5
from training.supervised import evaluate
from evaluation.metrics import calculate_metrics, save_classification_report
from evaluation.confusion_matrix import save_confusion_matrix


def main():
    print("\n" + "=" * 70)
    print("RESNET-SE-5 BASELINE EVALUATION")
    print("=" * 70)

    print("Device:", DEVICE)

    train_df, val_df, test_df = load_daghar_splits(
        TRAIN_FILE,
        VAL_FILE,
        TEST_FILE,
    )

    X_train, y_train = dataframe_to_numpy(train_df)
    X_val, y_val = dataframe_to_numpy(val_df)
    X_test, y_test = dataframe_to_numpy(test_df)

    unique_labels = sorted(set(y_train.tolist()))
    label_mapping = {
        label: index
        for index, label in enumerate(unique_labels)
    }

    train_dataset = HARDataset(
        X_train, y_train, label_mapping=label_mapping
    )
    test_dataset = HARDataset(
        X_test, y_test, label_mapping=label_mapping
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = ResNetSE5(
        in_channels=6,
        num_classes=len(unique_labels),
    ).to(DEVICE)

    checkpoint_path = MODEL_DIR / "resnet_se5_uci_supervised.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run scripts/run_supervised.py first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = torch.nn.CrossEntropyLoss()

    (
        test_loss,
        test_accuracy,
        predictions,
        labels,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE,
    )

    metrics = calculate_metrics(labels, predictions)

    print("\nTest Loss:", f"{test_loss:.4f}")
    print("Test Accuracy:", f"{test_accuracy:.4f}")
    print("Weighted Precision:", f"{metrics['precision_weighted']:.4f}")
    print("Weighted Recall:", f"{metrics['recall_weighted']:.4f}")
    print("Weighted F1:", f"{metrics['f1_weighted']:.4f}")

    report_path = (
        METRICS_DIR / "resnet_se5_uci_classification_report.json"
    )

    save_classification_report(
        labels,
        predictions,
        report_path,
    )

    class_names = [
        str(label)
        for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])
    ]

    cm_path = (
        CONFUSION_MATRIX_DIR
        / "resnet_se5_uci_confusion_matrix.png"
    )

    save_confusion_matrix(
        labels,
        predictions,
        cm_path,
        class_names=class_names,
    )

    print("\nSaved classification report:", report_path)
    print("Saved confusion matrix:", cm_path)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
