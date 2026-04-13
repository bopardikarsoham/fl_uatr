"""
fedprox_uatr.py — FedProx for UATR (ShipsEar)
===============================================

FedProx adds a proximal term to each client's local loss:
    L_FedProx = L_CE + (mu/2) * ||w - w_global||^2

This prevents clients from drifting too far from the global model
during local training, stabilizing convergence under non-IID data.

The server-side aggregation is identical to FedAvg.
Only the client fit() changes.

mu is passed from server to client via the fit config dict each round.
The best mu is found via Optuna (tune_fedprox_uatr.py) before full runs.

Usage:
  # Quick single run to verify
  python fedprox_uatr.py --alpha 1.0 --clients 5 --mu 0.01 --seeds 42

  # Full multi-seed run (all configs, Optuna-tuned mu)
  python fedprox_uatr.py --mu 0.01

  # With Optuna mu search first
  python fedprox_uatr.py --tune --n_trials 20
  python fedprox_uatr.py --mu <best_mu_from_tune>
"""

import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from flwr.simulation import start_simulation
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, Context
from typing import List, Dict, Tuple

from utils_uatr import (
    DATA_ROOT,
    NUM_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, NUM_CLASSES,
    COMBINATIONS, COLORS, LABELS, SEEDS,
    device, set_seed, Tee, progress_bar,
    NPZDataset, get_client_loaders, get_model, evaluate_model,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

os.environ["RAY_DEDUP_LOGS"]      = "0"
os.environ["RAY_DISABLE_METRICS"] = "1"

ALGO        = "fedprox"
RESULTS_DIR = "./fedprox_results"
PLOTS_DIR   = "./fedprox_plots"
LOGS_DIR    = "./fedprox_logs"

for d in [RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────
# Plot helpers (fedprox-specific folders)
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


def plot_combined(mean_logs, mu):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("FedProx — All Configurations (mu=" + str(mu) + ")",
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
    plt.savefig(os.path.join(PLOTS_DIR, "fedprox_combined_accuracy_f1.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Combined: fedprox_plots/fedprox_combined_accuracy_f1.png")


def plot_summary_bar(rows, mu):
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
    ax.set_title("FedProx — Best Accuracy & F1 (mu=" + str(mu) + ")",
                 fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fedprox_summary_bar.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Summary bar: fedprox_plots/fedprox_summary_bar.png")

# ─────────────────────────────────────────
# FedProx Client
# ─────────────────────────────────────────
class FedProxClient(fl.client.NumPyClient):
    """
    Identical to FedAvg client except fit() adds the proximal term:
        loss += (mu/2) * ||w_local - w_global||^2
    w_global is stored at the start of each round from set_parameters().
    """
    def __init__(self, client_id: int, data_dir: str):
        self.model          = get_model().to(device)
        self.criterion      = nn.CrossEntropyLoss()
        self.optimizer      = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=LR, weight_decay=WEIGHT_DECAY,
        )
        self.train_loader, self.test_loader = get_client_loaders(data_dir, client_id)
        self.global_params  = None   # stored at start of each round

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, params):
        sd = dict(zip(self.model.state_dict().keys(),
                      [torch.tensor(p) for p in params]))
        self.model.load_state_dict(sd, strict=True)
        # Store global params as detached tensors, one per named parameter
        # Only store trainable params — same order as model.parameters()
        self.global_params = [
            p.detach().clone()
            for p in self.model.parameters()
            if p.requires_grad
        ]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        mu           = config.get("mu", 0.01)
        local_epochs = config.get("local_epochs", LOCAL_EPOCHS)

        self.model.train()
        for _ in range(local_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()

                # Standard cross-entropy loss
                loss = self.criterion(self.model(images), labels)

                # Proximal term: (mu/2) * sum ||w_local - w_global||^2
                # Iterate only over trainable params — same order as global_params
                prox = torch.tensor(0.0, device=device)
                for w_local, w_global in zip(
                    (p for p in self.model.parameters() if p.requires_grad),
                    self.global_params,
                ):
                    prox = prox + torch.sum((w_local - w_global) ** 2)

                loss = loss + (mu / 2.0) * prox
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc, f1, auc, prec, rec, cm = evaluate_model(self.model, self.test_loader)
        return float(1 - acc), len(self.test_loader.dataset), {
            "accuracy": float(acc), "f1": float(f1), "auc": float(auc),
            "precision": float(prec), "recall": float(rec),
        }

# ─────────────────────────────────────────
# Single simulation run
# ─────────────────────────────────────────
def run_simulation(
    alpha: float, num_clients: int, seed: int, mu: float,
    max_rounds: int = NUM_ROUNDS,
    save_plots: bool = True,
) -> List[dict]:
    set_seed(seed)
    run_name = (ALGO + "_alpha" + str(alpha) + "_c" + str(num_clients)
                + "_mu" + str(round(mu, 4)) + "_seed" + str(seed))
    data_dir = os.path.join(DATA_ROOT,
                            "niid_alpha" + str(alpha) + "_c" + str(num_clients))

    run_log: List[dict]      = []
    round_times: List[float] = []
    round_start = [0.0]

    class _Strategy(FedAvg):
        """FedProx uses standard FedAvg aggregation — only the client changes."""
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
                "seed": seed, "mu": mu,
                "accuracy": avg_acc / 100, "f1": avg_f1,
                "auc": avg_auc, "time": elapsed,
            })

            bar   = progress_bar(server_round, max_rounds)
            eta   = np.mean(round_times) * (max_rounds - server_round)
            prev  = run_log[-2]["accuracy"]*100 if len(run_log) > 1 else avg_acc
            delta = ("(+" if avg_acc - prev >= 0 else "(") + str(round(avg_acc - prev, 2)) + "%)"

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
                print("  mu=" + str(mu)
                      + "  Total time: " + str(round(sum(round_times), 0)) + "s")
                print("-" * 58)
                print("")

            return aggregated

    def client_fn(context: Context) -> fl.client.Client:
        return FedProxClient(
            int(context.node_config["partition-id"]), data_dir
        ).to_client()

    initial_params = ndarrays_to_parameters(
        [v.cpu().numpy() for v in get_model().state_dict().values()]
    )
    strategy = _Strategy(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_params,
        on_fit_config_fn=lambda r: {
            "local_epochs": LOCAL_EPOCHS,
            "round": r,
            "mu": mu,          # passed to client fit() each round
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
        print("  Plot saved: fedprox_plots/" + run_name + "_curves.png")
    return run_log

# ─────────────────────────────────────────
# Grid search over paper's mu values
# ─────────────────────────────────────────
def tune_mu(alpha: float, num_clients: int, n_trials: int = 20) -> float:
    """
    Grid search over mu in {0.001, 0.01, 0.1, 0.5, 1.0} following
    Li et al. (FedProx, MLSys 2020) — the exact values used in the
    original paper. Each value is evaluated with 20-round trials on
    the specified config. Returns best mu.

    Citation: "Following Li et al. (FedProx), we search
               mu in {0.001, 0.01, 0.1, 0.5, 1.0}."
    """
    import json
    TUNE_ROUNDS  = 20
    MU_GRID      = [0.001, 0.01, 0.1, 0.5, 1.0]   # exact values from FedProx paper

    print("\n" + "=" * 58)
    print("  FedProx mu Grid Search (Li et al. MLSys 2020)")
    print("  Config : alpha=" + str(alpha) + "  clients=" + str(num_clients))
    print("  mu grid: " + str(MU_GRID))
    print("  Rounds per value: " + str(TUNE_ROUNDS))
    print("=" * 58 + "\n")

    results = []
    for i, mu in enumerate(MU_GRID):
        print("  [" + str(i+1) + "/" + str(len(MU_GRID)) + "] mu=" + str(mu))
        log     = run_simulation(
            alpha=alpha, num_clients=num_clients,
            seed=42, mu=mu, max_rounds=TUNE_ROUNDS,
            save_plots=False,   # don't clutter plots/ with tune trial curves
        )
        best_f1  = max(r["f1"]       for r in log)
        best_acc = max(r["accuracy"] for r in log) * 100
        results.append({"mu": mu, "best_f1": best_f1, "best_acc": best_acc})
        print("  --> mu=" + str(mu) + "  best F1=" + str(round(best_f1, 4))
              + "  best Acc=" + str(round(best_acc, 2)) + "%\n")

    # Pick best mu by F1
    best   = max(results, key=lambda x: x["best_f1"])
    best_mu = best["mu"]

    print("=" * 58)
    print("  mu Grid Search Results:")
    for r in results:
        marker = " <-- BEST" if r["mu"] == best_mu else ""
        print("    mu=" + str(r["mu"]) + "  F1=" + str(round(r["best_f1"], 4))
              + "  Acc=" + str(round(r["best_acc"], 2)) + "%" + marker)
    print("  Best mu: " + str(best_mu))
    print("=" * 58 + "\n")

    mu_results = {
        "best_mu":      best_mu,
        "best_f1":      best["best_f1"],
        "tune_alpha":   alpha,
        "tune_clients": num_clients,
        "mu_grid":      MU_GRID,
        "all_results":  results,
    }
    mu_path = os.path.join(RESULTS_DIR, "fedprox_best_mu.json")
    with open(mu_path, "w") as f:
        json.dump(mu_results, f, indent=2)
    print("  Saved: results/fedprox_best_mu.json")

    return best_mu

# ─────────────────────────────────────────
# Main — multi-seed full run
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu",       type=float, default=0.01,
                        help="Proximal coefficient (default: 0.01)")
    parser.add_argument("--alpha",    nargs="+", type=float,
                        default=[a for a, _ in COMBINATIONS])
    parser.add_argument("--clients",  nargs="+", type=int,
                        default=[c for _, c in COMBINATIONS])
    parser.add_argument("--seeds",    nargs="+", type=int, default=SEEDS)
    parser.add_argument("--test",        action="store_true",
                        help="Quick 5-round sanity test to verify algorithm is learning")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip runs whose CSV already exists")
    parser.add_argument("--run_tune",     action="store_true",
                        help="Run Optuna mu search first, then full runs")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of Optuna trials for mu search")
    parser.add_argument("--tune_alpha",   type=float, default=1.0,
                        help="Alpha config to use for mu tuning")
    parser.add_argument("--tune_clients", type=int, default=5,
                        help="Client count to use for mu tuning")
    args = parser.parse_args()

    # Optuna mu search if requested
    mu = args.mu

    # ── Quick sanity test ──────────────────────────────────
    if args.test:
        print("\n" + "=" * 58)
        print("  FedProx SANITY TEST — 5 rounds, alpha=1.0, c=5")
        print("  Accuracy should reach >50% by round 3")
        print("  If stuck at ~20-25%, algorithm is broken")
        print("=" * 58 + "\n")
        log = run_simulation(
            alpha=1.0, num_clients=5, seed=42,
            mu=args.mu, max_rounds=5, save_plots=False,
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
        mu = tune_mu(
            alpha=args.tune_alpha,
            num_clients=args.tune_clients,
            n_trials=args.n_trials,
        )
        print("\n  Using mu=" + str(round(mu, 5)) + " for full runs\n")
        print("  Best mu saved to: fedprox_results/fedprox_best_mu.json")
        print("  Now run individual configs with:")
        print("    python fedprox_uatr.py --mu " + str(round(mu, 5)) + " --alpha <a> --clients <c>")
        import sys; sys.exit(0)

    combos    = list(dict.fromkeys(zip(args.alpha, args.clients)))
    combos    = sorted(combos, key=lambda x: (-x[0], x[1]))
    total     = len(combos) * len(args.seeds)
    run_count = 0

    print("\n" + "=" * 62)
    print("  FedProx UATR — " + str(total) + " total runs")
    print("  Device  : " + str(device))
    print("  mu      : " + str(mu))
    print("  Configs : " + str([LABELS.get(c, str(c)) for c in combos]))
    print("  Seeds   : " + str(args.seeds))
    print("  Rounds  : " + str(NUM_ROUNDS) + "  |  Local epochs: " + str(LOCAL_EPOCHS))
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
                         + "_mu" + str(round(mu, 4)) + "_seed" + str(seed))
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
            print("  mu=" + str(mu) + "  seed=" + str(seed)
                  + "  |  Log: " + log_path)
            print("  Time: " + time.strftime("%Y-%m-%d %H:%M:%S"))
            print("=" * 62 + "\n")

            t0      = time.time()
            log     = run_simulation(alpha, num_clients, seed, mu)
            elapsed = time.time() - t0

            print("  Finished in " + str(round(elapsed / 60, 1)) + " min\n")
            tee.close()
            sys.stdout = tee.terminal

            all_seed_logs[key][seed] = log
            best_str = str(round(max(r["accuracy"] for r in log) * 100, 2))
            best_f1  = str(round(max(r["f1"] for r in log), 4))
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
        lbl = (ALGO + " mu=" + str(round(mu, 4)) + " "
               + LABELS.get((alpha, num_clients),
                             "a" + str(alpha) + "_c" + str(num_clients)))

        summary_rows.append({
            "config":             lbl,
            "algo":               ALGO,
            "mu":                 mu,
            "alpha":              alpha,
            "num_clients":        num_clients,
            "mean_best_accuracy": float(np.mean(best_accs)),
            "std_best_accuracy":  float(np.std(best_accs)),
            "mean_best_f1":       float(np.mean(best_f1s)),
            "std_best_f1":        float(np.std(best_f1s)),
            "n_seeds":            len(best_accs),
        })

        run_name_base = (ALGO + "_alpha" + str(alpha)
                         + "_c" + str(num_clients) + "_mu" + str(round(mu, 4)))
        plot_seed_variance(sd, run_name_base)

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
    pub.to_csv(os.path.join(RESULTS_DIR, ALGO + "_all_runs_summary.csv"), index=False)
    pub[["config", "accuracy_mean_std", "f1_mean_std", "n_seeds"]].to_csv(
        os.path.join(RESULTS_DIR, ALGO + "_publication_table.csv"), index=False
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

    plot_combined(mean_logs, mu)
    plot_summary_bar(summary_rows, mu)

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
    print("\n  Results : fedprox_results/")
    print("  Plots   : fedprox_plots/")
    print("  Logs    : fedprox_logs/")
    print("  Publication table: fedprox_results/fedprox_publication_table.csv")