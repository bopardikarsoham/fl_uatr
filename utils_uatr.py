"""
utils_uatr.py — Shared utilities for UATR FL experiments
==========================================================

Imported by: run_all_uatr.py, fedprox_uatr.py, ditto_uatr.py, perfedavg_uatr.py

Contains:
  - Global config (DATA_ROOT, hyperparams, combinations)
  - NPZDataset
  - get_model()
  - evaluate_model()
  - set_seed()
  - Tee (stdout logger)
  - progress_bar()
  - All plot functions
"""

import os, sys, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# ─────────────────────────────────────────
# Global config — single source of truth
# Change these here and all algorithm files pick them up
# ─────────────────────────────────────────
DATA_ROOT    = "./shipsear_fl"
RESULTS_DIR  = "./results"
PLOTS_DIR    = "./plots"
LOGS_DIR     = "./logs"

NUM_ROUNDS   = 30
LOCAL_EPOCHS = 3
BATCH_SIZE   = 32
NUM_CLASSES  = 5

# Tuned via Optuna on centralized baseline
LR           = 3.05e-4
WEIGHT_DECAY = 1.73e-4
DROPOUT      = 0.342
# unfrozen_layers=3: layer2 + layer3 + layer4

SEEDS = [42]   # single seed for development; use [42, 123, 2025] for final paper run

COMBINATIONS = [
    (10.0, 5), (10.0, 10),
    (1.0,  5), (1.0,  10),
    (0.1,  5), (0.1,  10),
]

COLORS = {
    (10.0, 5):  "#2E75B6", (10.0, 10): "#378ADD",
    (1.0,  5):  "#70AD47", (1.0,  10): "#639922",
    (0.1,  5):  "#ED7D31", (0.1,  10): "#C00000",
}
LABELS = {
    (10.0, 5):  "a=10.0 c5",  (10.0, 10): "a=10.0 c10",
    (1.0,  5):  "a=1.0  c5",  (1.0,  10): "a=1.0  c10",
    (0.1,  5):  "a=0.1  c5",  (0.1,  10): "a=0.1  c10",
}

CLASS_NAMES = {
    0: "Small Working", 1: "Small Rec/Utility",
    2: "Passenger Ferry", 3: "Large Commercial", 4: "Background Noise",
}

for d in [RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

# ─────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

# ─────────────────────────────────────────
# Tee — stdout to terminal + log file
# ─────────────────────────────────────────
class Tee:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log      = open(filepath, "w", buffering=1)
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()
        sys.stdout = self.terminal

# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────
class NPZDataset(Dataset):
    def __init__(self, npz_path):
        d            = np.load(npz_path)
        self.data    = torch.from_numpy(d["data"])
        self.targets = torch.from_numpy(d["targets"])
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx].long()

def get_client_loaders(data_dir: str, client_id: int):
    td = NPZDataset(os.path.join(data_dir, "train", str(client_id) + ".npz"))
    vd = NPZDataset(os.path.join(data_dir, "test",  str(client_id) + ".npz"))
    return (DataLoader(td, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0),
            DataLoader(vd, batch_size=BATCH_SIZE, shuffle=False, num_workers=0))

# ─────────────────────────────────────────
# Model — Optuna-tuned config
# ─────────────────────────────────────────
def get_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    # unfrozen_layers=3: layer2 + layer3 + layer4
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.layer3.parameters():
        p.requires_grad = True
    for p in model.layer2.parameters():
        p.requires_grad = True
    model.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(512, NUM_CLASSES),
    )
    return model

# ─────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────
def evaluate_model(model, loader):
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
    return acc, f1, auc, prec, rec, cm

# ─────────────────────────────────────────
# Progress bar
# ─────────────────────────────────────────
def progress_bar(current, total, width=30):
    filled = int(width * current / total)
    bar    = chr(9608) * filled + chr(9617) * (width - filled)
    return "[" + bar + "] " + str(current) + "/" + str(total)

# ─────────────────────────────────────────
# Plot: per-run convergence curves
# ─────────────────────────────────────────
def plot_run(log, run_name):
    df = pd.DataFrame(log)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(run_name.replace("_", " "), fontsize=11, fontweight="bold")
    for ax, col, scale, unit, color, title, ylabel in [
        (ax1, "accuracy", 100, "%",  "#2E75B6", "Accuracy per round",  "Accuracy (%)"),
        (ax2, "f1",       1,   "",   "#ED7D31", "Macro F1 per round",   "Macro F1"),
    ]:
        vals = df[col] * scale
        ax.plot(df["round"], vals, color=color, linewidth=2)
        ax.axhline(vals.max(), color=color, linestyle="--", alpha=0.4, linewidth=1)
        ax.set_title(title); ax.set_xlabel("Round"); ax.set_ylabel(ylabel)
        ax.text(0.98, 0.05, "Best: " + str(round(vals.max(), 2)) + unit,
                transform=ax.transAxes, ha="right", fontsize=9, color=color)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax2.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, run_name + "_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

# ─────────────────────────────────────────
# Plot: seed variance (shaded mean ± std)
# ─────────────────────────────────────────
def plot_seed_variance(seed_logs: Dict[int, List], run_name_base: str):
    all_acc = np.array([[r["accuracy"] for r in lg] for lg in seed_logs.values()])
    all_f1  = np.array([[r["f1"]       for r in lg] for lg in seed_logs.values()])
    ma, sa  = all_acc.mean(0) * 100, all_acc.std(0) * 100
    mf, sf  = all_f1.mean(0),        all_f1.std(0)
    rounds  = np.arange(1, len(ma) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    n = len(seed_logs)
    fig.suptitle(run_name_base + " — mean +/- std (" + str(n) + " seeds)",
                 fontsize=11, fontweight="bold")
    for ax, mean, std, raw_fn, color, ylabel, title, ylim in [
        (ax1, ma, sa, lambda lg: [r["accuracy"]*100 for r in lg],
         "#2E75B6", "Accuracy (%)", "Accuracy", None),
        (ax2, mf, sf, lambda lg: [r["f1"] for r in lg],
         "#ED7D31", "Macro F1", "Macro F1", (0, 1)),
    ]:
        for lg in seed_logs.values():
            ax.plot(rounds, raw_fn(lg), color=color, linewidth=0.8, alpha=0.35)
        ax.plot(rounds, mean, color=color, linewidth=2.5, label="mean")
        ax.fill_between(rounds, mean - std, mean + std,
                        alpha=0.2, color=color, label="+/- 1 std")
        ax.set_title(title); ax.set_xlabel("Round"); ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, run_name_base + "_seed_variance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Seed variance: plots/" + run_name_base + "_seed_variance.png")

# ─────────────────────────────────────────
# Plot: combined all configs
# ─────────────────────────────────────────
def plot_combined(mean_logs: Dict, algo: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(algo.upper() + " — All Configurations (mean across seeds)",
                 fontsize=12, fontweight="bold")
    for key, df in mean_logs.items():
        _, alpha, nc = key
        color = COLORS[(alpha, nc)]
        label = LABELS[(alpha, nc)]
        ls    = "-" if nc == 5 else "--"
        axes[0].plot(df["round"], df["accuracy"]*100,
                     color=color, linestyle=ls, linewidth=2, label=label)
        axes[1].plot(df["round"], df["f1"],
                     color=color, linestyle=ls, linewidth=2, label=label)
    axes[0].set_title("Accuracy vs Round")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Accuracy (%)")
    axes[1].set_title("Macro F1 vs Round")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim([0, 1])
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        h, l = ax.get_legend_handles_labels()
        h += [plt.Line2D([0],[0],color="gray",ls="-",lw=1.5),
              plt.Line2D([0],[0],color="gray",ls="--",lw=1.5)]
        l += ["— 5 clients", "-- 10 clients"]
        ax.legend(h, l, fontsize=8, framealpha=0.8, ncol=2)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, algo + "_combined_accuracy_f1.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Combined: plots/" + algo + "_combined_accuracy_f1.png")

# ─────────────────────────────────────────
# Plot: summary bar with error bars
# ─────────────────────────────────────────
def plot_summary_bar(rows, algo: str):
    df     = pd.DataFrame(rows).sort_values("mean_best_accuracy", ascending=False)
    labels = df["config"].tolist()
    x      = np.arange(len(labels))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, df["mean_best_accuracy"]*100, width,
           yerr=df["std_best_accuracy"]*100, capsize=4,
           label="Best Accuracy (%)", color="#2E75B6", alpha=0.85)
    ax.bar(x + width/2, df["mean_best_f1"]*100, width,
           yerr=df["std_best_f1"]*100, capsize=4,
           label="Best F1 x 100", color="#ED7D31", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(algo.upper() + " — Best Accuracy & F1 (mean +/- std across seeds)",
                 fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, algo + "_summary_bar.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Summary bar: plots/" + algo + "_summary_bar.png")