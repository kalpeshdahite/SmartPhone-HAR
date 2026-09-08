import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    TRAIN_FILE,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
)

from preprocessing.load_data import load_csv
from preprocessing.preprocess import dataframe_to_numpy
from preprocessing.dataset import HARDataset

from models.resnet_se import ResNetSE5


def main():

    print("\n" + "=" * 70)
    print("RESNET-SE-5 MODEL TEST")
    print("=" * 70)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    print("\nDevice:")
    print(DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)
    else:
        print("GPU: Not available")

    # --------------------------------------------------
    # Load training data
    # --------------------------------------------------

    train_df = load_csv(TRAIN_FILE)

    X_train, y_train = dataframe_to_numpy(train_df)

    # --------------------------------------------------
    # Label mapping
    # --------------------------------------------------

    unique_labels = sorted(set(y_train.tolist()))

    label_mapping = {
        label: index
        for index, label in enumerate(unique_labels)
    }

    print("\nLabel mapping:")
    print(label_mapping)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_dataset = HARDataset(
        X_train,
        y_train,
        label_mapping=label_mapping
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------
    # Get one batch
    # --------------------------------------------------

    X_batch, y_batch = next(iter(train_loader))

    print("\nInput batch:")
    print("X:", X_batch.shape)
    print("y:", y_batch.shape)

    # --------------------------------------------------
    # Move data to GPU/CPU
    # --------------------------------------------------

    X_batch = X_batch.to(DEVICE)
    y_batch = y_batch.to(DEVICE)

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = ResNetSE5(
        in_channels=6,
        num_classes=5
    )

    model = model.to(DEVICE)

    print("\nModel created successfully.")

    # --------------------------------------------------
    # Parameter count
    # --------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Total parameters:", f"{total_parameters:,}")
    print("Trainable parameters:", f"{trainable_parameters:,}")

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    model.train()

    outputs = model(X_batch)

    print("\nForward pass:")
    print("Output shape:", outputs.shape)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(outputs, y_batch)

    print("Loss:", loss.item())

    # --------------------------------------------------
    # Backward pass
    # --------------------------------------------------

    loss.backward()

    print("\nBackward pass:")
    print("Gradients calculated successfully.")

    # --------------------------------------------------
    # GPU synchronization
    # --------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print("GPU computation synchronized successfully.")

    print("\n" + "=" * 70)
    print("RESNET-SE-5 TEST SUCCESSFUL")
    print("=" * 70)


if __name__ == "__main__":
    main()