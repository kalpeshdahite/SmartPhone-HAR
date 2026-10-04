import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _loss_acc(model, loader, criterion, device):
    model.eval()
    loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss += criterion(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return loss / n, correct / n


def train_model(model, loaders, device, epochs=50, lr=1e-3, weight_decay=0.0):
    """Adam + CrossEntropy. Keeps the weights of the best VALIDATION accuracy epoch."""
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_epoch, best_state = -1.0, 0, None
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        tl, tc, n = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(y)
            tc += (out.argmax(1) == y).sum().item()
            n += len(y)
        vl, va = _loss_acc(model, loaders["validation"], criterion, device)
        hist["train_loss"].append(tl / n)
        hist["train_acc"].append(tc / n)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        if va > best_acc:
            best_acc, best_epoch = va, ep
            best_state = copy.deepcopy(model.state_dict())
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"epoch {ep:3d} | train loss {tl/n:.4f} acc {tc/n:.4f} "
                  f"| val loss {vl:.4f} acc {va:.4f}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)
    return model, hist, best_epoch, train_time


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


@torch.no_grad()
def measure_inference_ms(model, loader, device, repeats=3):
    """Average milliseconds per window (batches of 64, after warm-up)."""
    model.eval()
    xs = torch.cat([x for x, _ in loader]).to(device)
    model(xs[:64])
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(repeats):
        for i in range(0, len(xs), 64):
            model(xs[i:i + 64])
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / (repeats * len(xs)) * 1000


def compute_metrics(y_true, y_pred, n_classes=5):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(n_classes))).tolist(),
    }


def save_confusion_png(cm, class_names, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_curves_png(hist, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(hist["train_loss"], label="train")
    axes[0].plot(hist["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(hist["train_acc"], label="train")
    axes[1].plot(hist["val_acc"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)