import sys
from pathlib import Path

import json
import torch
from torch.utils.data import DataLoader


# --------------------------------------------------
# Project path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# Project imports
# --------------------------------------------------

from config import (
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    DEVICE,
    MODEL_DIR,
    METRICS_DIR,
)

from preprocessing.load_data import load_daghar_splits
from preprocessing.preprocess import dataframe_to_numpy
from preprocessing.dataset import HARDataset

from models.resnet_se import ResNetSE5

from training.supervised import (
    train_one_epoch,
    evaluate,
)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("RESNET-SE-5 SUPERVISED TRAINING")
    print("=" * 70)

    print("\nDevice:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    print("\nLoading dataset...")

    train_df, val_df, test_df = load_daghar_splits(
        TRAIN_FILE,
        VAL_FILE,
        TEST_FILE
    )

    # --------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------

    X_train, y_train = dataframe_to_numpy(
        train_df
    )

    X_val, y_val = dataframe_to_numpy(
        val_df
    )

    X_test, y_test = dataframe_to_numpy(
        test_df
    )

    # --------------------------------------------------
    # Label mapping
    # --------------------------------------------------

    unique_labels = sorted(
        set(y_train.tolist())
    )

    label_mapping = {
        label: index
        for index, label in enumerate(unique_labels)
    }

    print("\nLabel mapping:")
    print(label_mapping)

    # --------------------------------------------------
    # Create datasets
    # --------------------------------------------------

    train_dataset = HARDataset(
        X_train,
        y_train,
        label_mapping=label_mapping
    )

    val_dataset = HARDataset(
        X_val,
        y_val,
        label_mapping=label_mapping
    )

    test_dataset = HARDataset(
        X_test,
        y_test,
        label_mapping=label_mapping
    )

    # --------------------------------------------------
    # Create dataloaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("\nDataset sizes:")
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = ResNetSE5(
        in_channels=6,
        num_classes=len(unique_labels)
    )

    model = model.to(DEVICE)

    # --------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    best_val_accuracy = 0.0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        (
            val_loss,
            val_accuracy,
            _,
            _
        ) = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        # Save history
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"\nEpoch {epoch}/{NUM_EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_accuracy:.4f}"
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            model_path = (
                MODEL_DIR
                / "resnet_se5_uci_supervised.pt"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_mapping": label_mapping,
                    "val_accuracy": val_accuracy,
                    "epoch": epoch
                },
                model_path
            )

            print(
                f"Best model saved: {model_path}"
            )

    # --------------------------------------------------
    # Load best model
    # --------------------------------------------------

    print("\nLoading best model...")

    checkpoint = torch.load(
        MODEL_DIR / "resnet_se5_uci_supervised.pt",
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------
    # Test evaluation
    # --------------------------------------------------

    (
        test_loss,
        test_accuracy,
        test_predictions,
        test_labels
    ) = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"\nTest Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: {test_accuracy:.4f}"
    )

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------

    metrics = {
        "model": "ResNet-SE-5",
        "dataset": "DAGHAR UCI",
        "num_classes": len(unique_labels),
        "input_shape": [
            6,
            150
        ],
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "best_validation_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "label_mapping": label_mapping,
        "history": history
    }

    metrics_path = (
        METRICS_DIR
        / "resnet_se5_uci_supervised.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        f"\nMetrics saved: {metrics_path}"
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()