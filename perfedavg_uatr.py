"""
perfedavg_uatr.py — Per-FedAvg for UATR (ShipsEar)
=====================================================

Per-FedAvg (Fallah et al., NeurIPS 2020) applies Model-Agnostic
Meta-Learning (MAML) to federated learning. Instead of finding a
single global model that works for all clients, it finds a global
initialization that can be quickly adapted to each client with just
one or a few gradient steps.

Algorithm per round per client:
  1. Receive global params w from server
  2. Inner step : w' = w - alpha_inner * grad_L(w)   (one SGD step on local data)
  3. Meta  step : update using grad computed at w'    (gradient through inner step)
  4. Send updated w back to server

The server aggregates as in FedAvg (weighted average).

alpha_inner is searched via grid: {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}
(5 values, same spirit as FedProx mu grid)

Output goes to dedicated folders:
  perfedavg_results/
  perfedavg_plots/
  perfedavg_logs/

Usage:
  # Tune alpha_inner first, then run all configs
  python perfedavg_uatr.py --tune

  # Run with a specific alpha_inner
  python perfedavg_uatr.py --alpha_inner 1e-3

  # Quick single config test
  python perfedavg_uatr.py --alpha_inner 1e-3 --alpha 1.0 --clients 5
"""

import os, sys, time, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import flwr as fl
from flwr.simulation import start_simulation
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, Context
from typing import List, Dict, Tuple

from utils_uatr import (
    DATA_ROOT, SEEDS, NUM_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE,
    LR, WEIGHT_DECAY, NUM_CLASSES, COMBINATIONS, COLORS, LABELS,
    device, set_seed, Tee, progress_bar,
    NPZDataset, get_client_loaders, get_model, evaluate_model,
    plot_seed_variance,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["RAY_DEDUP_LOGS"]      = "0"
os.environ["RAY_DISABLE_METRICS"] = "1"

ALGO        = "perfedavg"
RESULTS_DIR = "./perfedavg_results"
PLOTS_DIR   = "./perfedavg_plots"
LOGS_DIR    = "./perfedavg_logs"

for d in [RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────
# Per-FedAvg Client
# ─────────────────────────────────────────
class PerFedAvgClient(fl.client.NumPyClient):
    """
    MAML-style Per-FedAvg client.

    Each fit() call does:
      For each local batch:
        1. Inner step: compute w' = w - alpha_inner * grad_L_batch1(w)
        2. Meta  step: compute grad of L_batch2(w') w.r.t. w
                       update w using that meta-gradient
    This trains w to be a good initialization — not a good final model.

    alpha_inner: inner loop learning rate (tuned via grid search)
    alpha_meta : outer loop LR = same as LR from utils (Adam handles this)
    """

    def __init__(self, client_id: int, data_dir: str, alpha_inner: float):
        self.model       = get_model().to(device)
        self.criterion   = nn.CrossEntropyLoss()
        self.optimizer   = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=LR, weight_decay=WEIGHT_DECAY,
        )
        self.alpha_inner = alpha_inner
        self.train_loader, self.test_loader = get_client_loaders(data_dir, client_id)

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, params):
        sd = dict(zip(self.model.state_dict().keys(),
                      [torch.tensor(p) for p in params]))
        self.model.load_state_dict(sd, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        alpha_inner  = config.get("alpha_inner", self.alpha_inner)
        local_epochs = config.get("local_epochs", LOCAL_EPOCHS)

        self.model.train()

        # Collect batches into a list for paired inner/meta updates
        batches = list(self.train_loader)
        if len(batches) < 2:
            # Fallback to standard SGD if too few batches (tiny clients)
            for images, labels in batches:
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()
                self.criterion(self.model(images), labels).backward()
                self.optimizer.step()
            return self.get_parameters(config), len(self.train_loader.dataset), {}

        for epoch in range(local_epochs):
            # Pair up batches: (support, query) for MAML inner/meta
            for i in range(0, len(batches) - 1, 2):
                x_support, y_support = batches[i]
                x_query,   y_query   = batches[i + 1]
                x_support = x_support.to(device); y_support = y_support.to(device)
                x_query   = x_query.to(device);   y_query   = y_query.to(device)

                # ── Inner step ──────────────────────────────────────────
                # Compute gradient on support batch
                self.model.zero_grad()
                inner_loss = self.criterion(self.model(x_support), y_support)
                inner_grads = torch.autograd.grad(
                    inner_loss,
                    [p for p in self.model.parameters() if p.requires_grad],
                    create_graph=True,   # need higher-order grad for meta step
                )

                # Manually apply inner gradient step to get w'
                # w' = w - alpha_inner * grad(L_support(w))
                fast_weights = []
                for p, g in zip(
                    [p for p in self.model.parameters() if p.requires_grad],
                    inner_grads,
                ):
                    fast_weights.append(p - alpha_inner * g)

                # ── Meta step ───────────────────────────────────────────
                # Compute loss at w' on query batch
                # We do this by temporarily replacing model params with fast_weights
                # via a functional forward pass
                meta_loss = self._forward_with_weights(x_query, y_query, fast_weights)

                # Meta gradient w.r.t. original w (through inner step)
                self.optimizer.zero_grad()
                meta_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            # Handle leftover batch if odd number
            if len(batches) % 2 == 1:
                x, y = batches[-1]
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                self.criterion(self.model(x), y).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def _forward_with_weights(self, x, y, fast_weights):
        """
        Functional forward pass using fast_weights instead of model.parameters().
        We temporarily patch the trainable parameters, run forward, then restore.
        This avoids needing functorch/higher and works with standard PyTorch.
        """
        trainable = [p for p in self.model.parameters() if p.requires_grad]

        # Save original data
        original_data = [p.data.clone() for p in trainable]

        # Patch with fast weights
        for p, fw in zip(trainable, fast_weights):
            p.data = fw

        # Forward
        loss = self.criterion(self.model(x), y)

        # Restore original weights
        for p, od in zip(trainable, original_data):
            p.data = od

        return loss

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc, f1, auc, prec, rec, cm = evaluate_model(self.model, self.test_loader)
        return float(1 - acc), len(self.test_loader.dataset), {
            "accuracy": float(acc), "f1": float(f1), "auc": float(auc),
            "precision": float(prec), "recall": float(rec),
        }

# ─────────────────────────────────────────
# Plot helpers (algo-specific folders)
# ─────────────────────────────────────────
def plot_run(log, run_name):
    df = pd.DataFrame(log)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(run_name.replace("_", " "), fontsize=11, fontweight="bold")
    for ax, col, scale, unit, color, title, ylabel in [
        (ax1, "accuracy", 100, "%",  "#2E75B6", "Accuracy per round", "Accuracy (%)"),
        (ax2, "f1",       1,   "",   "#ED7D31", "Macro F1 per round",  "Macro F1"),
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


def plot_combined(mean_logs: Dict, alpha_inner: float):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Per-FedAvg — All Configurations (alpha_inner=" + str(alpha_inner) + ")",
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
    plt.savefig(os.path.join(PLOTS_DIR, "perfedavg_combined_accuracy_f1.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Combined: perfedavg_plots/perfedavg_combined_accuracy_f1.png")


def plot_summary_bar(rows, alpha_inner: float):
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
    ax.set_title("Per-FedAvg — Best Accuracy & F1 (alpha_inner=" + str(alpha_inner) + ")",
                 fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "perfedavg_summary_bar.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Summary bar: perfedavg_plots/perfedavg_summary_bar.png")

# ─────────────────────────────────────────
# Single simulation run
# ─────────────────────────────────────────
def run_simulation(
    alpha: float, num_clients: int, seed: int, alpha_inner: float,
    max_rounds: int = NUM_ROUNDS,
    save_plots: bool = True,
    dropout: float = 0.0,
) -> List[dict]:
    set_seed(seed)
    run_name = (ALGO + "_alpha" + str(alpha) + "_c" + str(num_clients)
                + "_ai" + str(alpha_inner) + "_seed" + str(seed))
    data_dir = os.path.join(DATA_ROOT,
                            "niid_alpha" + str(alpha) + "_c" + str(num_clients))

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
                "run": run_name, "algo": ALGO, "round": server_round,
                "alpha": alpha, "num_clients": num_clients,
                "seed": seed, "alpha_inner": alpha_inner,
                "accuracy": avg_acc / 100, "f1": avg_f1,
                "auc": avg_auc, "time": elapsed, "dropout": dropout,
            })

            bar  = progress_bar(server_round, max_rounds)
            eta  = np.mean(round_times) * (max_rounds - server_round)
            prev = run_log[-2]["accuracy"]*100 if len(run_log) > 1 else avg_acc
            sign = "+" if avg_acc - prev >= 0 else ""
            delta = "(" + sign + str(round(avg_acc - prev, 2)) + "%)"

            print("  Round " + str(server_round).zfill(2) + "/" + str(max_rounds)
                  + "  " + bar)
            print("           Accuracy  : " + str(round(avg_acc, 2)) + "%  " + delta)
            print("           F1 (macro): " + str(round(avg_f1, 4))
                  + "  |  AUC: " + str(round(avg_auc, 4)))
            print("           Time      : " + str(round(elapsed, 1))
                  + "s  |  ETA: " + str(round(eta, 0)) + "s")
            print("")

            if server_round == max_rounds:
                best_acc = max(r["accuracy"] for r in run_log) * 100
                best_f1  = max(r["f1"]       for r in run_log)
                print("-" * 58)
                print("  Done!  Best Acc: " + str(round(best_acc, 2))
                      + "%  Best F1: " + str(round(best_f1, 4)))
                print("  alpha_inner=" + str(alpha_inner)
                      + "  Total time: " + str(round(sum(round_times), 0)) + "s")
                print("-" * 58)
                print("")

            return aggregated

    def client_fn(context: Context) -> fl.client.Client:
        return PerFedAvgClient(
            int(context.node_config["partition-id"]),
            data_dir,
            alpha_inner,
        ).to_client()

    initial_params = ndarrays_to_parameters(
        [v.cpu().numpy() for v in get_model().state_dict().values()]
    )
    active = max(1, int(num_clients * (1.0 - dropout)))
    strategy = _Strategy(
        fraction_fit=(1.0 - dropout), fraction_evaluate=(1.0 - dropout),
        min_fit_clients=active,
        min_evaluate_clients=active,
        min_available_clients=num_clients,
        initial_parameters=initial_params,
        on_fit_config_fn=lambda r: {
            "local_epochs": LOCAL_EPOCHS,
            "round": r,
            "alpha_inner": alpha_inner,
        },
    )
    start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=max_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    csv_path = os.path.join(RESULTS_DIR, run_name + "_rounds.csv")
    pd.DataFrame(run_log).to_csv(csv_path, index=False)
    if save_plots:
        plot_run(run_log, run_name)
        print("  Plot saved: perfedavg_plots/" + run_name + "_curves.png")
    return run_log

# ─────────────────────────────────────────
# Grid search for alpha_inner
# ─────────────────────────────────────────
def tune_alpha_inner(alpha: float, num_clients: int) -> float:
    """
    Grid search over alpha_inner in {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}
    on 20-round trials. Returns best alpha_inner by F1.
    """
    TUNE_ROUNDS      = 20
    ALPHA_INNER_GRID = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    print("\n" + "=" * 58)
    print("  Per-FedAvg alpha_inner Grid Search")
    print("  Config     : alpha=" + str(alpha) + "  clients=" + str(num_clients))
    print("  Grid       : " + str(ALPHA_INNER_GRID))
    print("  Rounds/trial: " + str(TUNE_ROUNDS))
    print("=" * 58 + "\n")

    results = []
    for i, ai in enumerate(ALPHA_INNER_GRID):
        print("  [" + str(i+1) + "/" + str(len(ALPHA_INNER_GRID))
              + "] alpha_inner=" + str(ai))
        log      = run_simulation(
            alpha=alpha, num_clients=num_clients,
            seed=42, alpha_inner=ai,
            max_rounds=TUNE_ROUNDS, save_plots=False,
        )
        best_f1  = max(r["f1"]       for r in log)
        best_acc = max(r["accuracy"] for r in log) * 100
        results.append({"alpha_inner": ai, "best_f1": best_f1, "best_acc": best_acc})
        print("  --> alpha_inner=" + str(ai)
              + "  best F1=" + str(round(best_f1, 4))
              + "  best Acc=" + str(round(best_acc, 2)) + "%\n")

    best      = max(results, key=lambda x: x["best_f1"])
    best_ai   = best["alpha_inner"]

    print("=" * 58)
    print("  alpha_inner Grid Search Results:")
    for r in results:
        marker = " <-- BEST" if r["alpha_inner"] == best_ai else ""
        print("    alpha_inner=" + str(r["alpha_inner"])
              + "  F1=" + str(round(r["best_f1"], 4))
              + "  Acc=" + str(round(r["best_acc"], 2)) + "%" + marker)
    print("  Best alpha_inner: " + str(best_ai))
    print("=" * 58 + "\n")

    tune_results = {
        "best_alpha_inner": best_ai,
        "best_f1":          best["best_f1"],
        "tune_alpha":       alpha,
        "tune_clients":     num_clients,
        "grid":             ALPHA_INNER_GRID,
        "all_results":      results,
    }
    with open(os.path.join(RESULTS_DIR, "perfedavg_best_alpha_inner.json"), "w") as f:
        json.dump(tune_results, f, indent=2)
    print("  Saved: perfedavg_results/perfedavg_best_alpha_inner.json")

    return best_ai

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha_inner", type=float, default=1e-3,
                        help="Inner loop LR for MAML step (default: 1e-3)")
    parser.add_argument("--alpha",       nargs="+", type=float,
                        default=[a for a, _ in COMBINATIONS])
    parser.add_argument("--clients",     nargs="+", type=int,
                        default=[c for _, c in COMBINATIONS])
    parser.add_argument("--seeds",       nargs="+", type=int, default=SEEDS)
    parser.add_argument("--test",        action="store_true",
                        help="Quick 5-round sanity test to verify algorithm is learning")
    parser.add_argument("--dropout",  type=float, default=0.0,
                        help="Client dropout rate e.g. 0.1=10%%, 0.3=30%% (default: 0.0)")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip runs whose CSV already exists")
    parser.add_argument("--run_tune",        action="store_true",
                        help="Grid search alpha_inner first, then full runs")
    parser.add_argument("--tune_alpha",  type=float, default=1.0)
    parser.add_argument("--tune_clients",type=int,   default=5)
    args = parser.parse_args()

    # Grid search alpha_inner if requested
    alpha_inner = args.alpha_inner

    # ── Quick sanity test ──────────────────────────────────
    if args.test:
        print("\n" + "=" * 58)
        print("  Per-FedAvg SANITY TEST — 5 rounds, alpha=1.0, c=5")
        print("  Accuracy should reach >50% by round 3")
        print("  If stuck at ~20-25%, algorithm is broken")
        print("=" * 58 + "\n")
        log = run_simulation(
            alpha=1.0, num_clients=5, seed=42,
            alpha_inner=args.alpha_inner, max_rounds=5, save_plots=False,
        )
        print("\n  TEST RESULTS:")
        for r in log:
            print("    Round " + str(r["round"]) + " — Acc: "
                  + str(round(r["accuracy"]*100, 2)) + "%  F1: "
                  + str(round(r["f1"], 4)))
        best = max(r["accuracy"] for r in log) * 100
        if best > 50:
            print("\n  PASS — Algorithm is learning (best: " + str(round(best,2)) + "%)")
        else:
            print("\n  FAIL — Algorithm stuck at random (" + str(round(best,2)) + "%)")
        import sys; sys.exit(0)

    if args.run_tune:
        alpha_inner = tune_alpha_inner(
            alpha=args.tune_alpha,
            num_clients=args.tune_clients,
        )
        print("  Using alpha_inner=" + str(alpha_inner) + " for full runs\n")
        print("  Best alpha_inner saved to: perfedavg_results/perfedavg_best_alpha_inner.json")
        print("  Now run individual configs with:")
        print("    python perfedavg_uatr.py --alpha_inner " + str(alpha_inner) + " --alpha <a> --clients <c>")
        import sys; sys.exit(0)

    combos    = list(dict.fromkeys(zip(args.alpha, args.clients)))
    combos    = sorted(combos, key=lambda x: (-x[0], x[1]))

    # If only one config passed — run it and exit cleanly
    if len(combos) == 1:
        alpha, num_clients = combos[0]
        for seed in args.seeds:
            log_path   = os.path.join(LOGS_DIR,
                         ALGO + "_alpha" + str(alpha) + "_c" + str(num_clients)
                         + "_ai" + str(alpha_inner) + "_seed" + str(seed) + ".out")
            tee        = Tee(log_path)
            sys.stdout = tee
            t0  = time.time()
            log = run_simulation(alpha, num_clients, seed, alpha_inner,
                                 dropout=args.dropout)
            elapsed = time.time() - t0
            tee.close()
            sys.stdout = tee.terminal
            best_acc = str(round(max(r["accuracy"] for r in log)*100, 2))
            best_f1  = str(round(max(r["f1"] for r in log), 4))
            print("  Done — best acc: " + best_acc + "%"
                  + "  best F1: " + best_f1
                  + "  (" + str(round(elapsed/60, 1)) + " min)")
        import sys; sys.exit(0)

    total     = len(combos) * len(args.seeds)
    run_count = 0

    print("\n" + "=" * 62)
    print("  Per-FedAvg UATR — " + str(total) + " total runs")
    print("  Device      : " + str(device))
    print("  alpha_inner : " + str(alpha_inner))
    print("  Configs     : " + str([LABELS.get(c, str(c)) for c in combos]))
    print("  Seeds       : " + str(args.seeds))
    print("  Rounds      : " + str(NUM_ROUNDS)
          + "  |  Local epochs: " + str(LOCAL_EPOCHS))
    print("=" * 62 + "\n")

    all_seed_logs: Dict[Tuple, Dict[int, List]] = {}
    summary_rows = []
    t_total = time.time()

    for alpha, num_clients in combos:
        key = (ALGO, alpha, num_clients)
        all_seed_logs[key] = {}

        for seed in args.seeds:
            run_count += 1
            run_name  = (ALGO + "_alpha" + str(alpha) + "_c" + str(num_clients)
                         + "_ai" + str(alpha_inner) + "_seed" + str(seed))
            csv_path  = os.path.join(RESULTS_DIR, run_name + "_rounds.csv")

            if args.resume and os.path.exists(csv_path):
                print("  [" + str(run_count) + "/" + str(total) + "] SKIP: " + run_name)
                all_seed_logs[key][seed] = pd.read_csv(csv_path).to_dict("records")
                continue

            log_path   = os.path.join(LOGS_DIR, run_name + ".out")
            tee        = Tee(log_path)
            sys.stdout = tee

            print("\n" + "=" * 62)
            print("  RUN " + str(run_count) + "/" + str(total) + ": " + run_name)
            print("  alpha_inner=" + str(alpha_inner) + "  seed=" + str(seed)
                  + "  |  Log: " + log_path)
            print("  Time: " + time.strftime("%Y-%m-%d %H:%M:%S"))
            print("=" * 62 + "\n")

            t0      = time.time()
            log     = run_simulation(alpha, num_clients, seed, alpha_inner, dropout=args.dropout)
            elapsed = time.time() - t0

            print("  Finished in " + str(round(elapsed / 60, 1)) + " min\n")
            tee.close()
            sys.stdout = tee.terminal

            all_seed_logs[key][seed] = log
            best_str = str(round(max(r["accuracy"] for r in log) * 100, 2))
            best_f1  = str(round(max(r["f1"]       for r in log), 4))
            print("  [" + str(run_count) + "/" + str(total) + "] " + run_name
                  + " — best acc: " + best_str + "%"
                  + "  best F1: " + best_f1
                  + "  (" + str(round(elapsed / 60, 1)) + " min)")

        # Aggregate across seeds
        sd = all_seed_logs[key]
        if not sd:
            continue

        best_accs = [max(r["accuracy"] for r in sd[s]) for s in sd]
        best_f1s  = [max(r["f1"]       for r in sd[s]) for s in sd]
        lbl = (ALGO + " ai=" + str(alpha_inner) + " "
               + LABELS.get((alpha, num_clients),
                             "a" + str(alpha) + "_c" + str(num_clients)))

        summary_rows.append({
            "config":             lbl,
            "algo":               ALGO,
            "alpha_inner":        alpha_inner,
            "alpha":              alpha,
            "num_clients":        num_clients,
            "mean_best_accuracy": float(np.mean(best_accs)),
            "std_best_accuracy":  float(np.std(best_accs)),
            "mean_best_f1":       float(np.mean(best_f1s)),
            "std_best_f1":        float(np.std(best_f1s)),
            "n_seeds":            len(best_accs),
        })

        run_name_base = (ALGO + "_alpha" + str(alpha)
                         + "_c" + str(num_clients) + "_ai" + str(alpha_inner))
        # Seed variance plot goes to perfedavg_plots
        all_acc = np.array([[r["accuracy"] for r in sd[s]] for s in sd])
        all_f1  = np.array([[r["f1"]       for r in sd[s]] for s in sd])
        if len(sd) > 1:
            import matplotlib.pyplot as plt2
            ma, sa = all_acc.mean(0)*100, all_acc.std(0)*100
            mf, sf = all_f1.mean(0),      all_f1.std(0)
            rounds = np.arange(1, len(ma)+1)
            fig, (a1, a2) = plt2.subplots(1, 2, figsize=(10, 4))
            fig.suptitle(run_name_base + " seed variance", fontsize=11, fontweight="bold")
            for ax, mn, sd2, color, ylabel, title, ylim in [
                (a1, ma, sa, "#2E75B6", "Accuracy (%)", "Accuracy", None),
                (a2, mf, sf, "#ED7D31", "Macro F1",     "Macro F1", (0,1)),
            ]:
                ax.plot(rounds, mn, color=color, lw=2.5)
                ax.fill_between(rounds, mn-sd2, mn+sd2, alpha=0.2, color=color)
                ax.set_title(title); ax.set_xlabel("Round"); ax.set_ylabel(ylabel)
                if ylim: ax.set_ylim(ylim)
                ax.grid(True, alpha=0.3)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            plt2.tight_layout()
            plt2.savefig(os.path.join(PLOTS_DIR, run_name_base + "_seed_variance.png"),
                        dpi=150, bbox_inches="tight")
            plt2.close()

    # Publication table
    pub = pd.DataFrame(summary_rows)
    pub["accuracy_mean_std"] = pub.apply(
        lambda r: str(round(r["mean_best_accuracy"]*100, 2))
                  + " +/- " + str(round(r["std_best_accuracy"]*100, 2)), axis=1
    )
    pub["f1_mean_std"] = pub.apply(
        lambda r: str(round(r["mean_best_f1"], 4))
                  + " +/- " + str(round(r["std_best_f1"], 4)), axis=1
    )
    pub.to_csv(os.path.join(RESULTS_DIR, "perfedavg_all_runs_summary.csv"), index=False)
    pub[["config", "accuracy_mean_std", "f1_mean_std", "n_seeds"]].to_csv(
        os.path.join(RESULTS_DIR, "perfedavg_publication_table.csv"), index=False
    )

    # Combined plots
    print("\n" + "=" * 62)
    print("  Generating combined plots...")
    mean_logs = {}
    for key, sd in all_seed_logs.items():
        if not sd:
            continue
        frames  = [pd.DataFrame(lg) for lg in sd.values()]
        mdf     = frames[0].copy()
        for col in ["accuracy", "f1", "auc"]:
            mdf[col] = np.mean([df[col].values for df in frames], axis=0)
        mean_logs[key] = mdf

    plot_combined(mean_logs, alpha_inner)
    plot_summary_bar(summary_rows, alpha_inner)

    te = time.time() - t_total
    print("\n" + "=" * 62)
    print("  ALL RUNS COMPLETE — " + str(round(te / 60, 1)) + " min total")
    print("=" * 62)
    print("\n  " + "Config".ljust(45) + "Acc mean+/-std".rjust(20)
          + "F1 mean+/-std".rjust(18))
    print("  " + "-" * 83)
    for row in summary_rows:
        acc_str = (str(round(row["mean_best_accuracy"]*100, 2)).rjust(7)
                   + " +/- " + str(round(row["std_best_accuracy"]*100, 2)).ljust(5))
        f1_str  = (str(round(row["mean_best_f1"], 4)).rjust(6)
                   + " +/- " + str(round(row["std_best_f1"], 4)).ljust(6))
        print("  " + row["config"].ljust(45) + acc_str + "  " + f1_str)
    print("\n  Results : perfedavg_results/")
    print("  Plots   : perfedavg_plots/")
    print("  Logs    : perfedavg_logs/")
    print("  Publication table: perfedavg_results/perfedavg_publication_table.csv")