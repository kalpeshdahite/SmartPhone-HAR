import argparse
import contextlib
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing.har_data import load_split, load_norm_stats, normalize, get_loaders
from models.cnn1d import CNN1D
from training.train_eval import set_seed, train_model, predict, compute_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--pretrain_epochs", type=int, default=100)
ap.add_argument("--finetune_epochs", type=int, default=50)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--skip_pretrain", action="store_true")
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENC_PATH = Path("checkpoints/simclr_encoder.pt")
Path("checkpoints").mkdir(exist_ok=True)
Path("results/tables").mkdir(parents=True, exist_ok=True)


# ---------- SimCLR pieces ----------
def augment(x):
    """Two random views of a window: noise + per-channel scaling + circular time shift."""
    B, C, T = x.shape
    x = x + 0.1 * torch.randn_like(x)
    x = x * (1 + 0.1 * torch.randn(B, C, 1, device=x.device))
    shifts = torch.randint(0, T, (B,))
    return torch.stack([torch.roll(x[i], int(shifts[i]), dims=1) for i in range(B)])


def nt_xent(z1, z2, tau=0.5):
    n = z1.size(0)
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def pretrain():
    set_seed(42)
    mean, std = load_norm_stats()
    X, _ = load_split("train")                       # labels are NOT used here
    X = torch.from_numpy(normalize(X, mean, std)).to(device)
    enc = CNN1D().encoder.to(device)
    head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64)).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)
    losses = []
    for ep in range(1, args.pretrain_epochs + 1):
        enc.train(); head.train()
        perm = torch.randperm(len(X), device=device)
        total, nb = 0.0, 0
        for i in range(0, len(X), 256):
            xb = X[perm[i:i + 256]]
            if len(xb) < 8:
                continue
            loss = nt_xent(head(enc(augment(xb))), head(enc(augment(xb))))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        losses.append(total / nb)
        if ep == 1 or ep % 10 == 0:
            print(f"pretrain epoch {ep:3d} | NT-Xent loss {losses[-1]:.4f}")
    torch.save(enc.state_dict(), ENC_PATH)
    plt.figure(figsize=(5, 3.5)); plt.plot(losses)
    plt.xlabel("epoch"); plt.ylabel("NT-Xent loss"); plt.title("SimCLR pretraining")
    plt.tight_layout(); plt.savefig("results/learning_curves/simclr_pretrain_loss.png", dpi=120)
    plt.close()


# ---------- fine-tune / scratch ----------
def run(frac, seed, pretrained):
    set_seed(seed)
    loaders = get_loaders(batch_size=64, label_fraction=frac, seed=seed)
    model = CNN1D().to(device)
    if pretrained:
        model.encoder.load_state_dict(torch.load(ENC_PATH, map_location=device))
    with contextlib.redirect_stdout(io.StringIO()):
        model, hist, best_ep, _ = train_model(model, loaders, device,
                                              epochs=args.finetune_epochs, lr=1e-3)
    y, p = predict(model, loaders["test"], device)
    m = compute_metrics(y, p)
    return {"model": "SSL+CNN" if pretrained else "CNN (scratch)",
            "labels_pct": int(frac * 100), "seed": seed,
            "n_windows": len(loaders["train"].dataset), "best_epoch": best_ep,
            "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]}


if args.skip_pretrain and ENC_PATH.exists():
    print("Using existing encoder:", ENC_PATH)
else:
    pretrain()

rows = []
for frac in [1.0, 0.10, 0.05, 0.01]:
    for seed in args.seeds:
        for pre in [False, True]:
            r = run(frac, seed, pre)
            rows.append(r)
            print(f"{r['model']:14s} labels={r['labels_pct']:3d}% seed={seed} "
                  f"acc={r['accuracy']:.4f} f1={r['f1_macro']:.4f}")

df = pd.DataFrame(rows)
df.to_csv("results/tables/label_efficiency_raw.csv", index=False)
summary = (df.groupby(["model", "labels_pct"])[["accuracy", "f1_macro"]]
             .agg(["mean", "std"]).round(4))
summary.to_csv("results/tables/label_efficiency_summary.csv")
print("\n=== SUMMARY (mean/std over seeds, test set) ===")
print(summary.to_string())

plt.figure(figsize=(6, 4))
for model_name, g in df.groupby("model"):
    s = g.groupby("labels_pct")["accuracy"].agg(["mean", "std"]).sort_index()
    plt.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", capsize=3, label=model_name)
plt.xscale("log"); plt.xticks([1, 5, 10, 100], ["1", "5", "10", "100"])
plt.xlabel("% of labeled training data"); plt.ylabel("Test accuracy")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("results/figures/label_efficiency.png", dpi=150)
print("Saved results/figures/label_efficiency.png")