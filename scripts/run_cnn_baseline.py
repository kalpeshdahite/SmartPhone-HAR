import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from preprocessing.har_data import get_loaders, CLASS_NAMES
from models.cnn1d import CNN1D
from training.train_eval import (set_seed, train_model, predict, compute_metrics,
                                 measure_inference_ms, count_params,
                                 save_confusion_png, save_curves_png)

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--label_fraction", type=float, default=1.0)
args = ap.parse_args()

name = f"cnn1d_uci_frac{int(args.label_fraction * 100)}_seed{args.seed}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)
print(f"Run: {name} | device: {device}")

loaders = get_loaders(batch_size=64, label_fraction=args.label_fraction, seed=args.seed)
model = CNN1D().to(device)
print("Trainable parameters:", count_params(model))

model, hist, best_epoch, train_time = train_model(
    model, loaders, device, epochs=args.epochs, lr=args.lr)

y_true, y_pred = predict(model, loaders["test"], device)
m = compute_metrics(y_true, y_pred, n_classes=len(CLASS_NAMES))
inf_ms = measure_inference_ms(model, loaders["test"], device)

result = {
    "name": name, "model": "CNN1D", "label_fraction": args.label_fraction,
    "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
    "best_epoch": best_epoch, "best_val_acc": max(hist["val_acc"]),
    "n_train_windows": len(loaders["train"].dataset),
    "params": count_params(model), "train_time_s": train_time,
    "inference_ms_per_window": inf_ms, "test": m, "history": hist,
}
Path("results/metrics").mkdir(parents=True, exist_ok=True)
Path("checkpoints").mkdir(exist_ok=True)
with open(f"results/metrics/{name}.json", "w") as f:
    json.dump(result, f, indent=2)
torch.save({"model_state": model.state_dict(), "args": vars(args)},
           f"checkpoints/{name}.pt")
save_confusion_png(m["confusion_matrix"], CLASS_NAMES,
                   f"results/confusion_matrices/{name}.png", title=name)
save_curves_png(hist, f"results/learning_curves/{name}.png", title=name)

print("\n=== TEST RESULTS ===")
print(f"best epoch {best_epoch} | train time {train_time:.1f}s | "
      f"inference {inf_ms:.4f} ms/window")
print(f"accuracy {m['accuracy']:.4f} | precision {m['precision_macro']:.4f} | "
      f"recall {m['recall_macro']:.4f} | F1 {m['f1_macro']:.4f}")
print("confusion matrix (rows=true, cols=pred):")
for row in m["confusion_matrix"]:
    print(row)