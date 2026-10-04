from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed/daghar/standardized_view/UCI")
SENSORS = ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
T = 150

def load(split):
    df = pd.read_csv(DATA_DIR / f"{split}.csv")
    X = np.stack(
        [df[[f"{s}-{t}" for t in range(T)]].values for s in SENSORS], axis=1
    ).astype(np.float32)                      # (N, 6, 150)
    return df, X

data = {s: load(s) for s in ["train", "validation", "test"]}
df_tr, X_tr = data["train"]

print("=== Shapes ===")
for k, (df, X) in data.items():
    print(f"{k:10s} X={X.shape}")

print("\n=== Metadata sample (train, first 5 rows) ===")
meta = ["index", "activity code", "standard activity code", "serial", "txt", "user", "window"]
print(df_tr[meta].head().to_string())

print("\n=== Label code pairing (activity code vs standard activity code) ===")
print(pd.crosstab(df_tr["activity code"], df_tr["standard activity code"]))

print("\n=== Users per split ===")
users = {k: set(df["user"].unique()) for k, (df, _) in data.items()}
for k, u in users.items():
    print(f"{k:10s} n_users={len(u)}  users={sorted(u)[:15]}")
print("train∩validation:", sorted(users["train"] & users["validation"]))
print("train∩test      :", sorted(users["train"] & users["test"]))
print("validation∩test :", sorted(users["validation"] & users["test"]))

print("\n=== Recording overlap (user, serial, txt) across splits ===")
rec = {k: set(map(tuple, df[["user", "serial", "txt"]].values)) for k, (df, _) in data.items()}
print("train∩test recordings:", len(rec["train"] & rec["test"]))
print("train∩validation recordings:", len(rec["train"] & rec["validation"]))

print("\n=== Per-class signal statistics (train) ===")
acc = X_tr[:, :3, :]
gyr = X_tr[:, 3:, :]
acc_norm = np.linalg.norm(acc, axis=1)       # (N,150)
gyr_norm = np.linalg.norm(gyr, axis=1)
rows = []
for c in sorted(df_tr["standard activity code"].unique()):
    m = (df_tr["standard activity code"] == c).values
    rows.append({
        "std_code": c,
        "acc_motion(std of |acc|)": acc_norm[m].std(axis=1).mean(),
        "gyro_energy(mean |gyro|)": gyr_norm[m].mean(),
        "mean_ax": acc[m, 0].mean(), "mean_ay": acc[m, 1].mean(), "mean_az": acc[m, 2].mean(),
    })
print(pd.DataFrame(rows).round(4).to_string(index=False))

print("\n=== Normalization stats (train only, per channel) ===")
mean = X_tr.mean(axis=(0, 2))
std = X_tr.std(axis=(0, 2))
for s, m_, sd in zip(SENSORS, mean, std):
    print(f"{s:8s} mean={m_:9.4f} std={sd:8.4f}")
Path("data/processed").mkdir(parents=True, exist_ok=True)
np.savez("data/processed/norm_stats_uci.npz", mean=mean, std=std)
print("Saved data/processed/norm_stats_uci.npz")