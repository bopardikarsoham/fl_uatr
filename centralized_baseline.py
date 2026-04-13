"""
baseline_uatr.py — Centralized baseline for UATR (ShipsEar)

Trains ResNet-18 on the full merged dataset (no federation).
This is the accuracy ceiling — the best possible result if privacy didn't matter.

Data structure expected:
  centralized/
    train.npz   — data: (1778, 3, 224, 224) float32,  targets: (1778,) int64
    test.npz    — data: (445,  3, 224, 224) float32,  targets: (445,)  int64

Classes:
  0 = Small Working   (369 total)
  1 = Small Rec/Utility (301 total)
  2 = Passenger Ferry (843 total)
  3 = Large Commercial (486 total)
  4 = Background Noise (224 total)

Usage:
  python baseline_uatr.py
  python baseline_uatr.py --data_dir ./my_centralized_folder
"""

import os, time, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
DATA_DIR     = "./centralized_data"
RESULTS_DIR  = "./centralized_results"
PLOTS_DIR    = "./centralized_plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

EPOCHS       = 75
BATCH_SIZE   = 32
LR           = 1e-3
ES_PATIENCE  = 10       # early stopping on val F1
NUM_CLASSES  = 5
WEIGHT_DECAY = 1e-4

CLASS_NAMES = {
    0: "Small Working", 1: "Small Rec/Utility",
    2: "Passenger Ferry", 3: "Large Commercial", 4: "Background Noise",
}

# ──────────────────────────────────────────
# Device
# ──────────────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Device : {device}")

# ──────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────
class NPZDataset(Dataset):
    def __init__(self, npz_path):
        d            = np.load(npz_path)
        self.data    = torch.from_numpy(d["data"])      # (N, 3, 224, 224) float32
        self.targets = torch.from_numpy(d["targets"])   # (N,) int64
        print(f"  Loaded {npz_path}")
        print(f"    Samples : {len(self.targets)}")
        counts = np.bincount(d["targets"], minlength=NUM_CLASSES)
        for i, c in enumerate(counts):
            print(f"    Class {i} ({CLASS_NAMES[i]}): {c}")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx].long()

# ──────────────────────────────────────────
# Model
# ──────────────────────────────────────────
def get_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(512, NUM_CLASSES),
    )
    return model

# ──────────────────────────────────────────
# Evaluate
# ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out   = model(images)
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(out, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y, yh, yp = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    acc  = float(accuracy_score(y, yh))
    f1   = float(f1_score(y, yh, average="macro", zero_division=0))
    prec = float(precision_score(y, yh, average="macro", zero_division=0))
    rec  = float(recall_score(y, yh,    average="macro", zero_division=0))
    cm   = confusion_matrix(y, yh, labels=list(range(NUM_CLASSES))).tolist()
    try:
        auc = float(roc_auc_score(y, yp, multi_class="ovr", average="macro"))
    except ValueError:
        auc = 0.0
    # Per-class F1
    f1_per_class = f1_score(y, yh, average=None, zero_division=0).tolist()
    return {
        "accuracy": acc, "f1": f1, "auc": auc,
        "precision": prec, "recall": rec,
        "cm": cm, "f1_per_class": f1_per_class,
    }

# ──────────────────────────────────────────
# Plot training curves
# ──────────────────────────────────────────
def plot_curves(log_df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Centralized Baseline — Training Curves", fontsize=13, fontweight="bold")

    specs = [
        (axes[0], "accuracy", 100, "%",  "#2E75B6", "Accuracy", "Accuracy (%)"),
        (axes[1], "f1",       1,   "",   "#ED7D31", "Macro F1", "Macro F1"),
        (axes[2], "auc",      1,   "",   "#70AD47", "AUC-ROC",  "AUC-ROC"),
    ]
    for ax, col, scale, unit, color, label, ylabel in specs:
        vals = log_df[col] * scale
        ax.plot(log_df["epoch"], vals, color=color, linewidth=2, label=label)
        best_val = vals.max()
        best_ep  = log_df.loc[vals.idxmax(), "epoch"]
        ax.axhline(best_val, color=color, linestyle="--", alpha=0.4, linewidth=1)
        ax.scatter([best_ep], [best_val], color=color, s=60, zorder=5)
        ax.set_title(f"{label} per epoch")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.text(0.98, 0.05, f"Best: {best_val:.2f}{unit} (ep {int(best_ep)})",
                transform=ax.transAxes, ha="right", fontsize=9, color=color)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "baseline_uatr_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot saved: plots/baseline_uatr_curves.png")


def plot_confusion_matrix(cm, title="Centralized Baseline — Confusion Matrix"):
    cm_arr = np.array(cm)
    pct    = cm_arr / cm_arr.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Row %")

    labels = [f"C{i}\n{CLASS_NAMES[i][:8]}" for i in range(NUM_CLASSES)]
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")

    for r in range(NUM_CLASSES):
        for c in range(NUM_CLASSES):
            color = "white" if pct[r, c] > 55 else "black"
            ax.text(c, r, f"{cm_arr[r,c]}\n({pct[r,c]:.1f}%)",
                    ha="center", va="center", fontsize=8,
                    color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "baseline_uatr_confusion.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot saved: plots/baseline_uatr_confusion.png")


def plot_per_class_f1(f1_per_class):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = ["#2E75B6", "#70AD47", "#ED7D31", "#7030A0", "#888780"]
    labels  = [f"C{i}: {CLASS_NAMES[i]}" for i in range(NUM_CLASSES)]
    bars    = ax.bar(labels, f1_per_class, color=colors, alpha=0.85)
    for bar, val in zip(bars, f1_per_class):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.set_ylabel("F1 Score")
    ax.set_title("Centralized Baseline — Per-Class F1", fontweight="bold")
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "baseline_uatr_per_class_f1.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot saved: plots/baseline_uatr_per_class_f1.png")

# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    args = parser.parse_args()

    print(f"\nLoading centralized dataset from {args.data_dir}/")
    train_ds = NPZDataset(os.path.join(args.data_dir, "train.npz"))
    test_ds  = NPZDataset(os.path.join(args.data_dir, "test.npz"))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model = get_model().to(device)
    total_p   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: ResNet-18 (pretrained)")
    print(f"  Total params    : {total_p:,}")
    print(f"  Trainable params: {trainable:,}  ({100*trainable/total_p:.1f}%)")

    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler  = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(EPOCHS // 3, 1), T_mult=2
    )

    print(f"\nTraining: {EPOCHS} epochs max  |  Early stopping patience={ES_PATIENCE}\n")
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'Acc':>8}  {'F1':>8}  {'AUC':>8}  {'LR':>10}")
    print(f"  {'-'*55}")

    best_f1    = 0.0
    best_wts   = None
    best_metrics = None
    no_improve = 0
    epoch_log  = []
    t0         = time.time()

    for epoch in range(1, EPOCHS + 1):

        # Train
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Evaluate
        m   = evaluate(model, test_loader)
        lr  = optimizer.param_groups[0]["lr"]
        row = {
            "epoch":    epoch,
            "loss":     total_loss / len(train_loader),
            "accuracy": m["accuracy"],
            "f1":       m["f1"],
            "auc":      m["auc"],
            "lr":       lr,
        }
        epoch_log.append(row)

        flag = ""
        if m["f1"] > best_f1:
            best_f1      = m["f1"]
            best_wts     = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = m
            no_improve   = 0
            flag         = " ← best"
        else:
            no_improve += 1

        print(f"  {epoch:>5}  {row['loss']:>8.4f}  "
              f"{m['accuracy']*100:>7.2f}%  "
              f"{m['f1']:>8.4f}  "
              f"{m['auc']:>8.4f}  "
              f"{lr:>10.2e}"
              f"{flag}")

        if no_improve >= ES_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no F1 improvement for {ES_PATIENCE} epochs)")
            break

    elapsed = time.time() - t0
    model.load_state_dict(best_wts)

    # Save weights
    torch.save(model.state_dict(),
               os.path.join(RESULTS_DIR, "best_baseline_uatr.pth"))

    # Save epoch log
    log_df = pd.DataFrame(epoch_log)
    log_df.to_csv(os.path.join(RESULTS_DIR, "baseline_uatr_epochs.csv"), index=False)

    # Save metrics
    with open(os.path.join(RESULTS_DIR, "baseline_uatr_metrics.json"), "w") as f:
        json.dump(best_metrics, f, indent=2)

    # Summary
    print(f"\n{'='*55}")
    print(f"  CENTRALIZED BASELINE — FINAL RESULTS")
    print(f"{'='*55}")
    print(f"  Accuracy  : {best_metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1  : {best_metrics['f1']:.4f}")
    print(f"  AUC-ROC   : {best_metrics['auc']:.4f}")
    print(f"  Precision : {best_metrics['precision']:.4f}")
    print(f"  Recall    : {best_metrics['recall']:.4f}")
    print(f"\n  Per-class F1:")
    for i, f1_val in enumerate(best_metrics["f1_per_class"]):
        print(f"    Class {i} ({CLASS_NAMES[i]:<20}): {f1_val:.4f}")
    print(f"\n  Training time : {elapsed/60:.1f} min")
    print(f"{'='*55}")

    # Plots
    print(f"\nGenerating plots...")
    plot_curves(log_df)
    plot_confusion_matrix(best_metrics["cm"])
    plot_per_class_f1(best_metrics["f1_per_class"])

    print(f"\nAll done!")
    print(f"  Results : {RESULTS_DIR}/")
    print(f"  Plots   : {PLOTS_DIR}/")