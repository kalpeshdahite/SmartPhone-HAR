import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocessing.har_data import (
    get_loaders, load_split, load_norm_stats, normalize, CLASS_NAMES, SENSORS
)

mean, std = load_norm_stats()
print("=== Dataset summary ===")
for split in ["train", "validation", "test"]:
    X, y = load_split(split)
    Xn = normalize(X, mean, std)
    print(f"{split:10s} X={Xn.shape} y={y.shape} classes={np.unique(y).tolist()}")
    print(f"           normalized mean per channel: {np.round(Xn.mean(axis=(0, 2)), 3)}")
    print(f"           normalized std  per channel: {np.round(Xn.std(axis=(0, 2)), 3)}")
print(f"Channels: {len(SENSORS)} | Window size: 150 steps | Classes: {len(CLASS_NAMES)}")

print("\n=== Label-fraction subsets (train) ===")
for frac in [1.0, 0.10, 0.05, 0.01]:
    ld = get_loaders(label_fraction=frac)["train"]
    counts = np.bincount(ld.dataset.y.numpy(), minlength=5).tolist()
    print(f"{int(frac*100):3d}% -> {len(ld.dataset):5d} windows, per-class {counts}")

print("\n=== One batch ===")
loaders = get_loaders()
xb, yb = next(iter(loaders["train"]))
print("X batch:", tuple(xb.shape), xb.dtype, "| y batch:", tuple(yb.shape), yb.dtype)

# One example window per class
X, y = load_split("train")
Xn = normalize(X, mean, std)
fig, axes = plt.subplots(5, 1, figsize=(9, 10), sharex=True)
for c, ax in enumerate(axes):
    w = Xn[np.where(y == c)[0][0]]
    for ch in range(6):
        ax.plot(w[ch], label=SENSORS[ch], lw=0.9)
    ax.set_ylabel(f"{c}: {CLASS_NAMES[c]}", fontsize=8)
axes[0].legend(ncol=6, fontsize=6)
fig.suptitle("Normalized example window per class")
fig.tight_layout()
Path("results/figures").mkdir(parents=True, exist_ok=True)
fig.savefig("results/figures/example_windows_per_class.png", dpi=120)
print("\nSaved results/figures/example_windows_per_class.png")