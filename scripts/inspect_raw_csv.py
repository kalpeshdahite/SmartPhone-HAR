import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path("data/processed/daghar/standardized_view/UCI")
OUT = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)
pat = re.compile(r"^(.*?)[-_]?(\d+)$")   # e.g. accel-x-0 -> ("accel-x", 0)

dfs = {}
for split in ["train", "validation", "test"]:
    f = DATA_DIR / f"{split}.csv"
    dfs[split] = pd.read_csv(f)
    print(f"{split:10s} shape = {dfs[split].shape}")

df = dfs["train"]
print("\nFirst 10 columns :", list(df.columns[:10]))
print("Last 10 columns  :", list(df.columns[-10:]))

# Group columns like 'accel-x-0..N' into channels
channels = {}
meta_cols = []
for c in df.columns:
    m = pat.match(c)
    if m and pd.api.types.is_numeric_dtype(df[c]):
        channels.setdefault(m.group(1), []).append((int(m.group(2)), c))
    else:
        meta_cols.append(c)

print("\nDetected sensor channels (name -> timesteps):")
for name, cols in channels.items():
    print(f"  {name:12s} {len(cols)}")
print("Non-signal columns:", meta_cols)

# Label column guess + class counts
label_candidates = [c for c in meta_cols if "activ" in c.lower() or "label" in c.lower()]
print("\nLabel column candidates:", label_candidates)
for lc in label_candidates:
    for split, d in dfs.items():
        print(f"  {lc} [{split}]:", d[lc].value_counts().sort_index().to_dict())

# Sanity checks
print("\nNaNs in train:", int(df.isna().sum().sum()))
for name, cols in channels.items():
    cols_sorted = [c for _, c in sorted(cols)]
    v = df[cols_sorted].values
    print(f"  {name:12s} mean={v.mean():9.4f} std={v.std():9.4f} "
          f"min={v.min():9.4f} max={v.max():9.4f}")

# Plot one example window
names = list(channels.keys())
fig, axes = plt.subplots(len(names), 1, figsize=(9, 1.6*len(names)), sharex=True)
axes = np.atleast_1d(axes)
for ax, name in zip(axes, names):
    cols_sorted = [c for _, c in sorted(channels[name])]
    ax.plot(df.loc[0, cols_sorted].values.astype(float)); ax.set_ylabel(name, fontsize=8)
if label_candidates:
    fig.suptitle(f"Example window 0, label = {df.loc[0, label_candidates[0]]}")
fig.tight_layout(); fig.savefig(OUT / "example_window.png", dpi=120)
print("\nSaved:", OUT / "example_window.png")