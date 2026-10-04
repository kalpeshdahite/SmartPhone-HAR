import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from preprocessing.har_data import load_split, stratified_subset, CLASS_NAMES
from training.train_eval import compute_metrics, save_confusion_png


def extract_features(X):
    """(N, 6, 150) -> (N, 64): 8 statistics for each of 6 raw channels + |acc| + |gyro|."""
    acc_mag = np.linalg.norm(X[:, :3, :], axis=1, keepdims=True)
    gyr_mag = np.linalg.norm(X[:, 3:, :], axis=1, keepdims=True)
    S = np.concatenate([X, acc_mag, gyr_mag], axis=1)       # (N, 8, 150)
    feats = [S.mean(2), S.std(2), S.min(2), S.max(2), np.median(S, 2),
             np.percentile(S, 25, axis=2), np.percentile(S, 75, axis=2),
             np.sqrt((S ** 2).mean(2))]
    return np.concatenate(feats, axis=1)


ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--label_fraction", type=float, default=1.0)
args = ap.parse_args()
name = f"rf_uci_frac{int(args.label_fraction * 100)}_seed{args.seed}"

Xtr, ytr = load_split("train")
Xva, yva = load_split("validation")
Xte, yte = load_split("test")
idx = stratified_subset(ytr, args.label_fraction, args.seed)
Xtr, ytr = Xtr[idx], ytr[idx]

t0 = time.perf_counter()
rf = RandomForestClassifier(n_estimators=300, random_state=args.seed, n_jobs=-1)
rf.fit(extract_features(Xtr), ytr)
train_time = time.perf_counter() - t0

val_acc = float((rf.predict(extract_features(Xva)) == yva).mean())
t0 = time.perf_counter()
y_pred = rf.predict(extract_features(Xte))
inf_ms = (time.perf_counter() - t0) / len(yte) * 1000
m = compute_metrics(yte, y_pred, n_classes=len(CLASS_NAMES))

result = {"name": name, "model": "RandomForest", "label_fraction": args.label_fraction,
          "seed": args.seed, "n_train_windows": int(len(ytr)), "val_acc": val_acc,
          "train_time_s": train_time, "inference_ms_per_window": inf_ms, "test": m}
Path("results/metrics").mkdir(parents=True, exist_ok=True)
with open(f"results/metrics/{name}.json", "w") as f:
    json.dump(result, f, indent=2)
save_confusion_png(m["confusion_matrix"], CLASS_NAMES,
                   f"results/confusion_matrices/{name}.png", title=name)

print(f"=== {name} ===")
print(f"val acc {val_acc:.4f} | train {train_time:.1f}s | inference {inf_ms:.4f} ms/window")
print(f"TEST accuracy {m['accuracy']:.4f} | precision {m['precision_macro']:.4f} | "
      f"recall {m['recall_macro']:.4f} | F1 {m['f1_macro']:.4f}")
print("confusion matrix (rows=true, cols=pred):")
for row in m["confusion_matrix"]:
    print(row)