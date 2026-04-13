"""
run_all_uatr.py — Runs all 6 FedAvg combinations for UATR ShipsEar.

Combinations: alpha x clients — (0.1,5), (0.1,10), (1.0,5), (1.0,10), (10.0,5), (10.0,10)

For each run:
  - Streams output to terminal AND saves to logs/<run_name>.out
  - Saves round metrics to results/<run_name>_rounds.csv
  - Saves per-run plots to plots/<run_name>_curves.png  (accuracy, F1, AUC)

After all runs:
  - plots/combined_accuracy_f1.png  — all 6 accuracy + F1 curves
  - plots/summary_bar.png           — best accuracy & F1 bar chart
  - results/all_runs_summary.csv    — final comparison table

Usage:
  python run_all_uatr.py
  python run_all_uatr.py --alpha 0.1 10.0 --clients 5
"""

import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import flwr as fl
from flwr.simulation import start_simulation
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, Context
from typing import List

os.environ["RAY_DEDUP_LOGS"]      = "0"
os.environ["RAY_DISABLE_METRICS"] = "1"

# ──────────────────────────────────────────
# GLOBAL CONFIG
# ──────────────────────────────────────────
DATA_ROOT    = "./shipsear_fl"
RESULTS_DIR  = "./results"
PLOTS_DIR    = "./plots"
LOGS_DIR     = "./logs"
NUM_ROUNDS   = 50
LOCAL_EPOCHS = 3
BATCH_SIZE   = 32
LR           = 1e-3
NUM_CLASSES  = 5

for d in [RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

CLASS_NAMES = {
    0: "Small Working", 1: "Small Rec/Utility",
    2: "Passenger Ferry", 3: "Large Commercial", 4: "Background Noise",
}

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
    (10.0, 5):  "α=10.0, c5",  (10.0, 10): "α=10.0, c10",
    (1.0,  5):  "α=1.0,  c5",  (1.0,  10): "α=1.0,  c10",
    (0.1,  5):  "α=0.1,  c5",  (0.1,  10): "α=0.1,  c10",
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

# ──────────────────────────────────────────
# Tee — stdout to terminal + log file
# ──────────────────────────────────────────
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

# ──────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────
class NPZDataset(Dataset):
    def __init__(self, npz_path):
        d            = np.load(npz_path)
        self.data    = torch.from_numpy(d["data"])
        self.targets = torch.from_numpy(d["targets"])

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
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(512, NUM_CLASSES))
    return model

# ──────────────────────────────────────────
# Evaluate
# ──────────────────────────────────────────
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
    rec  = float(recall_score(y, yh, average="macro", zero_division=0))
    cm   = confusion_matrix(y, yh, labels=list(range(NUM_CLASSES))).tolist()
    try:
        auc = float(roc_auc_score(y, yp, multi_class="ovr", average="macro"))
    except ValueError:
        auc = 0.0
    return acc, f1, auc, prec, rec, cm

def progress_bar(current, total, width=30):
    filled = int(width * current / total)
    return f"[{'█'*filled}{'░'*(width-filled)}] {current}/{total}"

# ──────────────────────────────────────────
# Per-run plot (accuracy, F1, AUC)
# ──────────────────────────────────────────
def plot_run(log, run_name, alpha, num_clients):
    df  = pd.DataFrame(log)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(f"FedAvg — α={alpha}, {num_clients} clients",
                 fontsize=13, fontweight="bold")

    specs = [
        (axes[0], "accuracy", 100, "%",    "#2E75B6", "Accuracy per round",  "Accuracy (%)"),
        (axes[1], "f1",       1,   "",     "#ED7D31", "Macro F1 per round",   "Macro F1"),
        (axes[2], "auc",      1,   "",     "#70AD47", "AUC-ROC per round",    "AUC-ROC"),
    ]
    for ax, col, scale, unit, color, title, ylabel in specs:
        vals = df[col] * scale
        ax.plot(df["round"], vals, color=color, linewidth=2)
        ax.axhline(vals.max(), color=color, linestyle="--", alpha=0.4, linewidth=1)
        ax.set_title(title); ax.set_xlabel("Round"); ax.set_ylabel(ylabel)
        ax.text(0.98, 0.05, f"Best: {vals.max():.2f}{unit}",
                transform=ax.transAxes, ha="right", fontsize=9, color=color)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[1].set_ylim([0, 1])
    if df["auc"].max() == 0:
        axes[2].text(0.5, 0.5, "AUC=0\n(missing classes\nin small clients)",
                     transform=axes[2].transAxes, ha="center", va="center",
                     fontsize=9, color="gray")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{run_name}_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: plots/{run_name}_curves.png")

# ──────────────────────────────────────────
# Combined plots
# ──────────────────────────────────────────
def plot_combined(all_logs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("FedAvg — All Configurations Comparison",
                 fontsize=13, fontweight="bold")

    for key, log in all_logs.items():
        df    = pd.DataFrame(log)
        color = COLORS[key]
        label = LABELS[key]
        ls    = "-" if key[1] == 5 else "--"
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
        handles, labels_leg = ax.get_legend_handles_labels()
        handles += [plt.Line2D([0],[0],color="gray",ls="-",lw=1.5),
                    plt.Line2D([0],[0],color="gray",ls="--",lw=1.5)]
        labels_leg += ["— 5 clients", "-- 10 clients"]
        ax.legend(handles, labels_leg, fontsize=8, framealpha=0.8, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "combined_accuracy_f1.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Combined plot saved: plots/combined_accuracy_f1.png")


def plot_summary_bar(summary_rows):
    df     = pd.DataFrame(summary_rows).sort_values("best_accuracy", ascending=False)
    labels = df["run"].str.replace("fedavg_alpha", "α=").str.replace("_c", " c").tolist()
    x      = np.arange(len(labels))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, df["best_accuracy"]*100, width,
                label="Best Accuracy (%)", color="#2E75B6", alpha=0.85)
    b2 = ax.bar(x + width/2, df["best_f1"]*100,       width,
                label="Best F1 × 100",    color="#ED7D31", alpha=0.85)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Best Accuracy & F1 — All Configurations", fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "summary_bar.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Summary bar saved: plots/summary_bar.png")

# ──────────────────────────────────────────
# Single simulation run
# ──────────────────────────────────────────
def run_simulation(alpha, num_clients):
    run_name = f"fedavg_alpha{alpha}_c{num_clients}"
    data_dir = os.path.join(DATA_ROOT, f"niid_alpha{alpha}_c{num_clients}")

    def get_client_loaders(client_id):
        train_ds = NPZDataset(os.path.join(data_dir, "train", f"{client_id}.npz"))
        test_ds  = NPZDataset(os.path.join(data_dir, "test",  f"{client_id}.npz"))
        return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0),
                DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0))

    run_log: List[dict]  = []
    round_times: List[float] = []
    round_start = [0.0]

    class _Strategy(FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            round_start[0] = time.time()
            return super().aggregate_fit(server_round, results, failures)

        def aggregate_evaluate(self, server_round, results, failures):
            aggregated = super().aggregate_evaluate(server_round, results, failures)
            elapsed    = time.time() - round_start[0]
            round_times.append(elapsed)

            accs, f1s, aucs = [], [], []
            for _, r in results:
                m = r.metrics
                if "accuracy" in m: accs.append(m["accuracy"])
                if "f1"       in m: f1s.append(m["f1"])
                if "auc"      in m: aucs.append(m["auc"])

            avg_acc = float(np.mean(accs)) * 100 if accs else 0.0
            avg_f1  = float(np.mean(f1s))         if f1s  else 0.0
            avg_auc = float(np.mean(aucs))         if aucs else 0.0

            run_log.append({
                "run": run_name, "round": server_round,
                "alpha": alpha, "num_clients": num_clients,
                "accuracy": avg_acc / 100, "f1": avg_f1,
                "auc": avg_auc, "time": elapsed,
            })

            bar   = progress_bar(server_round, NUM_ROUNDS)
            eta   = np.mean(round_times) * (NUM_ROUNDS - server_round)
            prev  = run_log[-2]["accuracy"]*100 if len(run_log) > 1 else avg_acc
            delta = f"  ({avg_acc - prev:+.2f}%)"

            print(f"  Round {server_round:02d}/{NUM_ROUNDS}  {bar}")
            print(f"           Accuracy  : {avg_acc:.2f}%{delta}")
            print(f"           F1 (macro): {avg_f1:.4f}  |  AUC: {avg_auc:.4f}")
            print(f"           Time      : {elapsed:.1f}s  |  ETA: {eta:.0f}s\n")

            if server_round == NUM_ROUNDS:
                best_acc = max(r["accuracy"] for r in run_log) * 100
                best_f1  = max(r["f1"]       for r in run_log)
                print(f"{'─'*58}")
                print(f"  Training complete!")
                print(f"  Best Accuracy : {best_acc:.2f}%")
                print(f"  Best F1       : {best_f1:.4f}")
                print(f"  Total time    : {sum(round_times):.0f}s")
                print(f"{'─'*58}\n")

            return aggregated

    class _Client(fl.client.NumPyClient):
        def __init__(self, client_id):
            self.model       = get_model().to(device)
            self.criterion   = nn.CrossEntropyLoss()
            self.optimizer   = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=LR, weight_decay=1e-4,
            )
            self.train_loader, self.test_loader = get_client_loaders(client_id)
            print(f"  Client {client_id}: "
                  f"{len(self.train_loader.dataset)} train  |  "
                  f"{len(self.test_loader.dataset)} test")

        def get_parameters(self, config):
            return [v.cpu().numpy() for v in self.model.state_dict().values()]

        def set_parameters(self, params):
            sd = dict(zip(self.model.state_dict().keys(),
                          [torch.tensor(p) for p in params]))
            self.model.load_state_dict(sd, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            self.model.train()
            for _ in range(config.get("local_epochs", LOCAL_EPOCHS)):
                for images, labels in self.train_loader:
                    images, labels = images.to(device), labels.to(device)
                    self.optimizer.zero_grad()
                    self.criterion(self.model(images), labels).backward()
                    self.optimizer.step()
            return self.get_parameters(config), len(self.train_loader.dataset), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            acc, f1, auc, prec, rec, cm = evaluate_model(self.model, self.test_loader)
            return float(1 - acc), len(self.test_loader.dataset), {
                "accuracy": float(acc), "f1": float(f1), "auc": float(auc),
                "precision": float(prec), "recall": float(rec),
            }

    def client_fn(context: Context) -> fl.client.Client:
        return _Client(int(context.node_config["partition-id"])).to_client()

    initial_params = ndarrays_to_parameters(
        [v.cpu().numpy() for v in get_model().state_dict().values()]
    )
    strategy = _Strategy(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_params,
        on_fit_config_fn=lambda r: {"local_epochs": LOCAL_EPOCHS, "round": r},
    )

    start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    csv_path = os.path.join(RESULTS_DIR, f"{run_name}_rounds.csv")
    pd.DataFrame(run_log).to_csv(csv_path, index=False)
    print(f"  CSV saved: results/{run_name}_rounds.csv")

    plot_run(run_log, run_name, alpha, num_clients)

    return run_log

# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha",   nargs="+", type=float,
                        default=[10.0, 10.0, 1.0, 1.0, 0.1, 0.1])
    parser.add_argument("--clients", nargs="+", type=int,
                        default=[5, 10, 5, 10, 5, 10])
    args = parser.parse_args()

    combos = list(dict.fromkeys(zip(args.alpha, args.clients)))
    combos = sorted(combos, key=lambda x: (-x[0], x[1]))

    print(f"\n{'='*58}")
    print(f"  UATR FL — Running {len(combos)} configurations")
    print(f"  Device: {device}")
    print(f"  Rounds: {NUM_ROUNDS}  |  Local epochs: {LOCAL_EPOCHS}")
    for c in combos:
        print(f"    {LABELS[c]}")
    print(f"{'='*58}\n")

    all_logs     = {}
    summary_rows = []
    t_total      = time.time()

    for alpha, num_clients in combos:
        run_name = f"fedavg_alpha{alpha}_c{num_clients}"
        log_path = os.path.join(LOGS_DIR, f"{run_name}.out")
        tee      = Tee(log_path)
        sys.stdout = tee

        print(f"\n{'='*58}")
        print(f"  START: α={alpha}, {num_clients} clients")
        print(f"  Log : {log_path}")
        print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*58}\n")

        t0      = time.time()
        log     = run_simulation(alpha, num_clients)
        elapsed = time.time() - t0

        print(f"\n  Finished in {elapsed/60:.1f} min\n")
        tee.close()
        sys.stdout = tee.terminal

        all_logs[(alpha, num_clients)] = log

        best_acc = max(r["accuracy"] for r in log)
        best_f1  = max(r["f1"]       for r in log)
        best_auc = max(r["auc"]       for r in log)
        summary_rows.append({
            "run":            run_name,
            "alpha":          alpha,
            "num_clients":    num_clients,
            "best_accuracy":  best_acc,
            "best_f1":        best_f1,
            "best_auc":       best_auc,
            "final_accuracy": log[-1]["accuracy"],
            "final_f1":       log[-1]["f1"],
            "total_time_s":   int(elapsed),
        })

        print(f"  [{run_name}] done — "
              f"best acc: {best_acc*100:.2f}%  "
              f"best F1: {best_f1:.4f}  "
              f"({elapsed/60:.1f} min)")

    print(f"\n{'='*58}")
    print("  Generating combined plots...")
    plot_combined(all_logs)
    plot_summary_bar(summary_rows)

    pd.DataFrame(summary_rows).sort_values(
        "best_accuracy", ascending=False
    ).to_csv(os.path.join(RESULTS_DIR, "all_runs_summary.csv"), index=False)

    total_elapsed = time.time() - t_total
    print(f"\n{'='*58}")
    print(f"  ALL RUNS COMPLETE — {total_elapsed/60:.1f} min total")
    print(f"{'='*58}")
    print(f"\n  {'Run':<30} {'Best Acc':>10} {'Best F1':>9} {'Time':>8}")
    print(f"  {'-'*57}")
    for row in summary_rows:
        print(f"  {row['run']:<30} "
              f"{row['best_accuracy']*100:>9.2f}%  "
              f"{row['best_f1']:>9.4f}  "
              f"{row['total_time_s']//60:>5}m{row['total_time_s']%60:02d}s")
    print(f"\n  Results : {RESULTS_DIR}/")
    print(f"  Plots   : {PLOTS_DIR}/")
    print(f"  Logs    : {LOGS_DIR}/")