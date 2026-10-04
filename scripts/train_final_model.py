import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from preprocessing.har_data import get_loaders, load_norm_stats, CLASS_NAMES
from models.cnn1d import CNN1D
from training.train_eval import (set_seed, train_model, predict,
                                 compute_metrics, count_params)

ap = argparse.ArgumentParser()
ap.add_argument("--init", choices=["scratch", "ssl"], default="scratch")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)
loaders = get_loaders(batch_size=64, label_fraction=1.0, seed=args.seed)
model = CNN1D().to(device)
if args.init == "ssl":
    model.encoder.load_state_dict(
        torch.load("checkpoints/simclr_encoder.pt", map_location=device))
print(f"Training final model (init={args.init}), params={count_params(model)}")

model, hist, best_epoch, train_time = train_model(
    model, loaders, device, epochs=args.epochs, lr=1e-3)
y_true, y_pred = predict(model, loaders["test"], device)
m = compute_metrics(y_true, y_pred, n_classes=len(CLASS_NAMES))

mean, std = load_norm_stats()
Path("checkpoints").mkdir(exist_ok=True)
torch.save({
    "model_state": model.state_dict(), "init": args.init, "seed": args.seed,
    "best_epoch": best_epoch, "mean": mean.tolist(), "std": std.tolist(),
    "class_names": CLASS_NAMES,
    "test_accuracy": m["accuracy"], "test_f1_macro": m["f1_macro"],
}, "checkpoints/final_model.pt")
print(f"\nSaved checkpoints/final_model.pt | best epoch {best_epoch}")
print(f"OFFLINE test accuracy {m['accuracy']:.4f} | F1 {m['f1_macro']:.4f}")