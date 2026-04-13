"""
run_all_uatr.py - Multi-seed FL experiment runner for UATR (ShipsEar)

Runs all algorithm x alpha x clients x seed combinations and produces
publication-quality results with mean +/- std across seeds.

Output:
  logs/    <run_name>.out              - full terminal log
  results/ <run_name>_rounds.csv       - per-round metrics
  results/ all_runs_summary.csv        - aggregated results
  results/ publication_table.csv       - mean +/- std table (copy into paper)
  plots/   <run_name>_curves.png       - per-run convergence
  plots/   <config>_seed_variance.png  - shaded mean +/- std across seeds
  plots/   combined_accuracy_f1.png    - all configs overlaid
  plots/   summary_bar.png             - bar chart with error bars

Usage:
  python run_all_uatr.py                   # run everything (18 runs, 3 seeds)
  python run_all_uatr.py --seeds 1         # quick single-seed test
  python run_all_uatr.py --resume          # skip completed runs
  python run_all_uatr.py --algos fedavg --alpha 1.0 --clients 5 --seeds 42
"""

import os, sys, time, argparse, random
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
import flwr as fl
from flwr.simulation import start_simulation
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, Context
from typing import List, Dict, Tuple

os.environ["RAY_DEDUP_LOGS"]      = "0"
os.environ["RAY_DISABLE_METRICS"] = "1"

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATA_ROOT    = "./shipsear_fl"
RESULTS_DIR  = "./fedavg_optuna_results"
PLOTS_DIR    = "./fedavg_optuna_plots"
LOGS_DIR     = "./fedavg_optuna_logs"
NUM_ROUNDS   = 50
LOCAL_EPOCHS = 3
BATCH_SIZE   = 32
LR           = 3.05e-4
NUM_CLASSES  = 5
SEEDS        = [42]

# Add "fedprox", "perfedavg", "ditto" here as you implement them
ALGORITHMS   = ["fedavg"]

COMBINATIONS = [
    (10.0, 5), (10.0, 10),
    (1.0,  5), (1.0,  10),
    (0.1,  5), (0.1,  10),
]
COLORS = {
    (10.0, 5): "#2E75B6", (10.0, 10): "#378ADD",
    (1.0,  5): "#70AD47", (1.0,  10): "#639922",
    (0.1,  5): "#ED7D31", (0.1,  10): "#C00000",
}
LABELS = {
    (10.0, 5):  "a=10.0 c5",  (10.0, 10): "a=10.0 c10",
    (1.0,  5):  "a=1.0  c5",  (1.0,  10): "a=1.0  c10",
    (0.1,  5):  "a=0.1  c5",  (0.1,  10): "a=0.1  c10",
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
# Tee - stdout to terminal + .out file
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

# ─────────────────────────────────────────
# Model
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
    model.fc = nn.Sequential(nn.Dropout(0.342), nn.Linear(512, NUM_CLASSES))
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

def progress_bar(current, total, width=30):
    filled = int(width * current / total)
    bar    = chr(9608) * filled + chr(9617) * (width - filled)
    return "[" + bar + "] " + str(current) + "/" + str(total)

# ─────────────────────────────────────────
# Per-run plot
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
    plt.savefig(os.path.join(PLOTS_DIR, run_name + "_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

# ─────────────────────────────────────────
# Seed variance plot
# ─────────────────────────────────────────
def plot_seed_variance(seed_logs: Dict[int, List], run_name_base: str):
    all_acc = np.array([[r["accuracy"] for r in lg] for lg in seed_logs.values()])
    all_f1  = np.array([[r["f1"]       for r in lg] for lg in seed_logs.values()])
    ma, sa  = all_acc.mean(axis=0) * 100, all_acc.std(axis=0) * 100
    mf, sf  = all_f1.mean(axis=0),        all_f1.std(axis=0)
    rounds  = np.arange(1, len(ma) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    n = len(seed_logs)
    fig.suptitle(run_name_base + " - mean +/- std (" + str(n) + " seeds)",
                 fontsize=11, fontweight="bold")

    for ax, mean, std, raw_fn, color, ylabel, title, ylim in [
        (ax1, ma, sa, lambda lg: [r["accuracy"] * 100 for r in lg],
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
# Combined convergence plot
# ─────────────────────────────────────────
def plot_combined(mean_logs: Dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("All Configurations - mean across seeds",
                 fontsize=12, fontweight="bold")
    for key, df in mean_logs.items():
        _, alpha, nc = key
        color = COLORS[(alpha, nc)]
        label = LABELS[(alpha, nc)]
        ls    = "-" if nc == 5 else "--"
        axes[0].plot(df["round"], df["accuracy"] * 100,
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
        h += [plt.Line2D([0], [0], color="gray", ls="-",  lw=1.5),
              plt.Line2D([0], [0], color="gray", ls="--", lw=1.5)]
        l += ["- 5 clients", "-- 10 clients"]
        ax.legend(h, l, fontsize=8, framealpha=0.8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "combined_accuracy_f1.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Combined: plots/combined_accuracy_f1.png")

# ─────────────────────────────────────────
# Summary bar chart
# ─────────────────────────────────────────
def plot_summary_bar(rows):
    df     = pd.DataFrame(rows).sort_values("mean_best_accuracy", ascending=False)
    labels = df["config"].tolist()
    x      = np.arange(len(labels))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, df["mean_best_accuracy"] * 100, width,
           yerr=df["std_best_accuracy"] * 100, capsize=4,
           label="Best Accuracy (%)", color="#2E75B6", alpha=0.85)
    ax.bar(x + width/2, df["mean_best_f1"] * 100, width,
           yerr=df["std_best_f1"] * 100, capsize=4,
           label="Best F1 x 100", color="#ED7D31", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Best Accuracy & F1 - mean +/- std across seeds", fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "summary_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Summary bar: plots/summary_bar.png")

# ─────────────────────────────────────────
# Single simulation run
# ─────────────────────────────────────────
def run_simulation(algo: str, alpha: float, num_clients: int, seed: int) -> List[dict]:
    set_seed(seed)
    run_name = algo + "_alpha" + str(alpha) + "_c" + str(num_clients) + "_seed" + str(seed)
    data_dir = os.path.join(DATA_ROOT, "niid_alpha" + str(alpha) + "_c" + str(num_clients))

    def get_client_loaders(client_id):
        td = NPZDataset(os.path.join(data_dir, "train", str(client_id) + ".npz"))
        vd = NPZDataset(os.path.join(data_dir, "test",  str(client_id) + ".npz"))
        return (DataLoader(td, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0),
                DataLoader(vd, batch_size=BATCH_SIZE, shuffle=False, num_workers=0))

    run_log: List[dict]      = []
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
                "run": run_name, "algo": algo, "round": server_round,
                "alpha": alpha, "num_clients": num_clients, "seed": seed,
                "accuracy": avg_acc / 100, "f1": avg_f1,
                "auc": avg_auc, "time": elapsed,
            })

            bar   = progress_bar(server_round, NUM_ROUNDS)
            eta   = np.mean(round_times) * (NUM_ROUNDS - server_round)
            prev  = run_log[-2]["accuracy"] * 100 if len(run_log) > 1 else avg_acc
            delta = "(" + ("+" if avg_acc - prev >= 0 else "") + str(round(avg_acc - prev, 2)) + "%)"

            print("  Round " + str(server_round).zfill(2) + "/" + str(NUM_ROUNDS) + "  " + bar)
            print("           Accuracy  : " + str(round(avg_acc, 2)) + "%  " + delta)
            print("           F1 (macro): " + str(round(avg_f1, 4)) + "  |  AUC: " + str(round(avg_auc, 4)))
            print("           Time      : " + str(round(elapsed, 1)) + "s  |  ETA: " + str(round(eta, 0)) + "s")
            print("")

            if server_round == NUM_ROUNDS:
                best_acc = max(r["accuracy"] for r in run_log) * 100
                best_f1  = max(r["f1"]       for r in run_log)
                print("-" * 58)
                print("  Done!  Best Acc: " + str(round(best_acc, 2)) + "%  Best F1: " + str(round(best_f1, 4)))
                print("  Total time: " + str(round(sum(round_times), 0)) + "s")
                print("-" * 58)
                print("")

            return aggregated

    class _Client(fl.client.NumPyClient):
        def __init__(self, client_id):
            self.model     = get_model().to(device)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=LR, weight_decay=1.73e-4,
            )
            self.train_loader, self.test_loader = get_client_loaders(client_id)

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

    csv_path = os.path.join(RESULTS_DIR, run_name + "_rounds.csv")
    pd.DataFrame(run_log).to_csv(csv_path, index=False)
    plot_run(run_log, run_name)
    return run_log

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algos",   nargs="+", default=ALGORITHMS)
    parser.add_argument("--alpha",   nargs="+", type=float,
                        default=[a for a, _ in COMBINATIONS])
    parser.add_argument("--clients", nargs="+", type=int,
                        default=[c for _, c in COMBINATIONS])
    parser.add_argument("--seeds",   nargs="+", type=int, default=SEEDS)
    parser.add_argument("--resume",  action="store_true",
                        help="Skip runs whose CSV already exists")
    args = parser.parse_args()

    combos    = list(dict.fromkeys(zip(args.alpha, args.clients)))
    combos    = sorted(combos, key=lambda x: (-x[0], x[1]))
    total     = len(args.algos) * len(combos) * len(args.seeds)
    run_count = 0

    print("")
    print("=" * 62)
    print("  UATR FL - " + str(total) + " total runs")
    print("  Device  : " + str(device))
    print("  Algos   : " + str(args.algos))
    print("  Configs : " + str([LABELS.get(c, str(c)) for c in combos]))
    print("  Seeds   : " + str(args.seeds))
    print("  Rounds  : " + str(NUM_ROUNDS) + "  |  Local epochs: " + str(LOCAL_EPOCHS))
    print("=" * 62)
    print("")

    all_seed_logs: Dict[Tuple, Dict[int, List]] = {}
    summary_rows = []
    t_total = time.time()

    for algo in args.algos:
        for alpha, num_clients in combos:
            key = (algo, alpha, num_clients)
            all_seed_logs[key] = {}

            for seed in args.seeds:
                run_count += 1
                run_name  = algo + "_alpha" + str(alpha) + "_c" + str(num_clients) + "_seed" + str(seed)
                csv_path  = os.path.join(RESULTS_DIR, run_name + "_rounds.csv")

                if args.resume and os.path.exists(csv_path):
                    print("  [" + str(run_count) + "/" + str(total) + "] SKIP: " + run_name)
                    all_seed_logs[key][seed] = pd.read_csv(csv_path).to_dict("records")
                    continue

                log_path   = os.path.join(LOGS_DIR, run_name + ".out")
                tee        = Tee(log_path)
                sys.stdout = tee

                print("")
                print("=" * 62)
                print("  RUN " + str(run_count) + "/" + str(total) + ": " + run_name)
                print("  Seed: " + str(seed) + "  |  Log: " + log_path)
                print("  Time: " + time.strftime("%Y-%m-%d %H:%M:%S"))
                print("=" * 62)
                print("")

                t0      = time.time()
                log     = run_simulation(algo, alpha, num_clients, seed)
                elapsed = time.time() - t0

                print("  Finished in " + str(round(elapsed / 60, 1)) + " min")
                print("")
                tee.close()
                sys.stdout = tee.terminal

                all_seed_logs[key][seed] = log
                best_str = str(round(max(r["accuracy"] for r in log) * 100, 2))
                print("  [" + str(run_count) + "/" + str(total) + "] " + run_name +
                      " - best acc: " + best_str + "%  (" + str(round(elapsed / 60, 1)) + " min)")

            # Aggregate across seeds for this config
            sd = all_seed_logs[key]
            if not sd:
                continue

            best_accs = [max(r["accuracy"] for r in sd[s]) for s in sd]
            best_f1s  = [max(r["f1"]       for r in sd[s]) for s in sd]
            lbl = algo + " " + LABELS.get((alpha, num_clients), "a" + str(alpha) + "_c" + str(num_clients))

            summary_rows.append({
                "config":             lbl,
                "algo":               algo,
                "alpha":              alpha,
                "num_clients":        num_clients,
                "mean_best_accuracy": float(np.mean(best_accs)),
                "std_best_accuracy":  float(np.std(best_accs)),
                "mean_best_f1":       float(np.mean(best_f1s)),
                "std_best_f1":        float(np.std(best_f1s)),
                "n_seeds":            len(best_accs),
            })

            run_name_base = algo + "_alpha" + str(alpha) + "_c" + str(num_clients)
            plot_seed_variance(sd, run_name_base)

    # Publication table
    pub = pd.DataFrame(summary_rows)
    pub["accuracy_mean_std"] = pub.apply(
        lambda r: str(round(r["mean_best_accuracy"] * 100, 2)) +
                  " +/- " + str(round(r["std_best_accuracy"] * 100, 2)), axis=1
    )
    pub["f1_mean_std"] = pub.apply(
        lambda r: str(round(r["mean_best_f1"], 4)) +
                  " +/- " + str(round(r["std_best_f1"], 4)), axis=1
    )
    pub.to_csv(os.path.join(RESULTS_DIR, "all_runs_summary.csv"), index=False)
    pub[["config", "accuracy_mean_std", "f1_mean_std", "n_seeds"]].to_csv(
        os.path.join(RESULTS_DIR, "publication_table.csv"), index=False
    )

    # Combined plots
    print("")
    print("=" * 62)
    print("  Generating combined plots...")

    mean_logs = {}
    for key, sd in all_seed_logs.items():
        if not sd:
            continue
        frames  = [pd.DataFrame(lg) for lg in sd.values()]
        mean_df = frames[0].copy()
        for col in ["accuracy", "f1", "auc"]:
            mean_df[col] = np.mean([df[col].values for df in frames], axis=0)
        mean_logs[key] = mean_df

    plot_combined(mean_logs)
    plot_summary_bar(summary_rows)

    total_elapsed = time.time() - t_total
    print("")
    print("=" * 62)
    print("  ALL RUNS COMPLETE - " + str(round(total_elapsed / 60, 1)) + " min total")
    print("=" * 62)
    print("")
    print("  " + "Config".ljust(40) + "Acc mean+/-std".rjust(22) + "F1 mean+/-std".rjust(20))
    print("  " + "-" * 82)
    for row in summary_rows:
        acc_str = str(round(row["mean_best_accuracy"] * 100, 2)).rjust(8) + " +/- " + str(round(row["std_best_accuracy"] * 100, 2)).ljust(6)
        f1_str  = str(round(row["mean_best_f1"], 4)).rjust(7) + " +/- " + str(round(row["std_best_f1"], 4)).ljust(6)
        print("  " + row["config"].ljust(40) + acc_str + "  " + f1_str)
    print("")
    print("  Publication table: " + RESULTS_DIR + "/publication_table.csv")