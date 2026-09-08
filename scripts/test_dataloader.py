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
)
from preprocessing.load_data import load_daghar_splits
from preprocessing.preprocess import dataframe_to_numpy
from preprocessing.dataset import HARDataset


def main():

    print("\n" + "=" * 70)
    print("PYTORCH DATALOADER TEST")
    print("=" * 70)

    # --------------------------------------------------
    # Load CSV files
    # --------------------------------------------------

    train_df, val_df, test_df = load_daghar_splits(
        TRAIN_FILE,
        VAL_FILE,
        TEST_FILE
    )

    # --------------------------------------------------
    # Convert data to NumPy
    # --------------------------------------------------

    X_train, y_train = dataframe_to_numpy(train_df)
    X_val, y_val = dataframe_to_numpy(val_df)
    X_test, y_test = dataframe_to_numpy(test_df)

    # --------------------------------------------------
    # Create label mapping using TRAINING SET ONLY
    # --------------------------------------------------

    unique_labels = sorted(set(y_train.tolist()))

    label_mapping = {
        label: index
        for index, label in enumerate(unique_labels)
    }

    print("\nLabel mapping:")
    print(label_mapping)

    # --------------------------------------------------
    # Create PyTorch datasets
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

    print("\nDataset sizes:")
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    # --------------------------------------------------
    # Create DataLoaders
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

    # --------------------------------------------------
    # Test one batch
    # --------------------------------------------------

    X_batch, y_batch = next(iter(train_loader))

    print("\nFirst training batch:")
    print("X shape:", X_batch.shape)
    print("y shape:", y_batch.shape)

    print("\nData type:")
    print("X:", X_batch.dtype)
    print("y:", y_batch.dtype)

    print("\nLabel values in batch:")
    print(torch.unique(y_batch))

    print("\n" + "=" * 70)
    print("DATALOADER TEST SUCCESSFUL")
    print("=" * 70)


if __name__ == "__main__":
    main()