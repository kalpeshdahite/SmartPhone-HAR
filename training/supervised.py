import torch


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
    """
    Train the model for one epoch.

    Returns:
        average_loss
        accuracy
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for X, y in dataloader:

        X = X.to(device)
        y = y.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X)

        # Calculate loss
        loss = criterion(outputs, y)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Statistics
        running_loss += loss.item() * X.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == y).sum().item()
        total += y.size(0)

    average_loss = running_loss / total
    accuracy = correct / total

    return average_loss, accuracy


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device
):
    """
    Evaluate the model.

    Returns:
        average_loss
        accuracy
        all_predictions
        all_labels
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    for X, y in dataloader:

        X = X.to(device)
        y = y.to(device)

        # Forward pass
        outputs = model(X)

        # Loss
        loss = criterion(outputs, y)

        running_loss += loss.item() * X.size(0)

        # Predictions
        predictions = outputs.argmax(dim=1)

        correct += (predictions == y).sum().item()
        total += y.size(0)

        # Store predictions and labels
        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_labels.extend(
            y.cpu().numpy().tolist()
        )

    average_loss = running_loss / total
    accuracy = correct / total

    return (
        average_loss,
        accuracy,
        all_predictions,
        all_labels
    )