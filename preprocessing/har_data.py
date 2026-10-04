from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path("data/processed/daghar/standardized_view/UCI")
STATS_PATH = Path("data/processed/norm_stats_uci.npz")
SENSORS = ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
T = 150
LABEL_COL = "standard activity code"
# Provisional names: verify against DAGHAR/UCI documentation
CLASS_NAMES = ["sitting", "standing", "walking", "upstairs", "downstairs"]


def load_split(split):
    """Returns X (N, 6, 150) float32 and y (N,) int64 for 'train'/'validation'/'test'."""
    df = pd.read_csv(DATA_DIR / f"{split}.csv")
    X = np.stack(
        [df[[f"{s}-{t}" for t in range(T)]].values for s in SENSORS], axis=1
    ).astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int64)
    return X, y


def load_norm_stats():
    d = np.load(STATS_PATH)
    return d["mean"].astype(np.float32), d["std"].astype(np.float32)


def normalize(X, mean, std):
    """Per-channel standardization using TRAIN statistics only."""
    return (X - mean[None, :, None]) / std[None, :, None]


def stratified_subset(y, fraction, seed=42):
    """Indices of a class-balanced random subset containing `fraction` of each class."""
    if fraction >= 1.0:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        k = max(1, int(round(len(c_idx) * fraction)))
        idx.append(rng.choice(c_idx, size=k, replace=False))
    return np.sort(np.concatenate(idx))


class HARWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def get_loaders(batch_size=64, label_fraction=1.0, seed=42):
    """Train/val/test loaders. label_fraction < 1 subsamples ONLY the train set."""
    mean, std = load_norm_stats()
    out = {}
    for split in ["train", "validation", "test"]:
        X, y = load_split(split)
        X = normalize(X, mean, std)
        if split == "train":
            idx = stratified_subset(y, label_fraction, seed)
            X, y = X[idx], y[idx]
        out[split] = DataLoader(
            HARWindowDataset(X, y),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,  # 0 is the safest on Windows
        )
    return out