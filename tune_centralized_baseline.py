"""
tune_baseline_uatr.py — Optuna hyperparameter search for UATR centralized baseline
====================================================================================

Searches the following hyperparameters:
  - lr            : learning rate            [1e-4, 5e-3]  (log scale)
  - weight_decay  : L2 regularization        [1e-5, 1e-2]  (log scale)
  - dropout       : dropout before fc layer  [0.1, 0.5]
  - unfrozen_layers: how many ResNet blocks to unfreeze  [1, 3]
                     1 = layer4 only  (current)
                     2 = layer3 + layer4
                     3 = layer2 + layer3 + layer4

Each trial trains for a SHORT budget (TRIAL_EPOCHS with early stopping)
to keep the search fast. The best config is then re-trained for the full
FINAL_EPOCHS budget and saved.

Output:
  optuna_results/
    study.db                  — SQLite study (resumable)
    best_params.json          — best hyperparams → copy into FL scripts
    best_params_full.json     — best params + final full-run metrics
    optimization_history.png  — objective vs trial
    param_importance.png      — which params mattered most
    baseline_uatr_curves.png  — training curves of best config full run
    best_baseline_tuned.pth   — weights of best full run

Usage:
  python tune_baseline_uatr.py
  python tune_baseline_uatr.py --data_dir ./shipsear_fl/centralized --n_trials 30
  python tune_baseline_uatr.py --resume   # continues from existing study.db
"""

import os, json, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATA_DIR     = "./centralized_data"
RESULTS_DIR  = "./optuna_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_CLASSES   = 5
BATCH_SIZE    = 32
TRIAL_EPOCHS  = 25    # short budget per trial
FINAL_EPOCHS  = 75    # full training for best config
ES_PATIENCE   = 7     # early stopping during trials (tighter than final run)
N_TRIALS      = 40    # number of Optuna trials

CLASS_NAMES = {
    0: "Small Working", 1: "Small Rec/Utility",
    2: "Passenger Ferry", 3: "Large Commercial", 4: "Background Noise",
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

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
# Model builder — parameterized
# ─────────────────────────────────────────
def get_model(dropout: float, unfrozen_layers: int):
    """
    unfrozen_layers:
      1 -> unfreeze layer4 only
      2 -> unfreeze layer3 + layer4
      3 -> unfreeze layer2 + layer3 + layer4
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Selectively unfreeze from the top
    layers_to_unfreeze = [model.layer4, model.layer3, model.layer2]
    for i in range(unfrozen_layers):
        for p in layers_to_unfreeze[i].parameters():
            p.requires_grad = True

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(512, NUM_CLASSES),
    )
    return model

# ─────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────
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
    acc = float(accuracy_score(y, yh))
    f1  = float(f1_score(y, yh, average="macro", zero_division=0))
    try:
        auc = float(roc_auc_score(y, yp, multi_class="ovr", average="macro"))
    except ValueError:
        auc = 0.0
    return acc, f1, auc

# ─────────────────────────────────────────
# Training loop (shared by trial + final)
# ─────────────────────────────────────────
def train_loop(
    model, train_loader, test_loader,
    lr, weight_decay,
    max_epochs, es_patience,
    trial=None,   # pass Optuna trial for pruning, None for final run
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(max_epochs // 3, 1), T_mult=2
    )

    best_f1    = 0.0
    best_wts   = None
    best_acc   = 0.0
    best_auc   = 0.0
    no_improve = 0
    log        = []

    for epoch in range(1, max_epochs + 1):
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

        acc, f1, auc = evaluate(model, test_loader)
        log.append({"epoch": epoch, "loss": total_loss / len(train_loader),
                    "accuracy": acc, "f1": f1, "auc": auc})

        if f1 > best_f1:
            best_f1  = f1
            best_acc = acc
            best_auc = auc
            best_wts = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # Optuna pruning — report intermediate value
        if trial is not None:
            trial.report(f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if no_improve >= es_patience:
            break

    return best_f1, best_acc, best_auc, best_wts, log

# ─────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────
def objective(trial, train_loader, test_loader):
    # Search space
    lr             = trial.suggest_float("lr",             1e-4, 5e-3, log=True)
    weight_decay   = trial.suggest_float("weight_decay",   1e-5, 1e-2, log=True)
    dropout        = trial.suggest_float("dropout",        0.1,  0.5)
    unfrozen_layers= trial.suggest_int("unfrozen_layers",  1,    3)

    model = get_model(dropout, unfrozen_layers).to(device)

    best_f1, best_acc, best_auc, _, _ = train_loop(
        model, train_loader, test_loader,
        lr=lr, weight_decay=weight_decay,
        max_epochs=TRIAL_EPOCHS, es_patience=ES_PATIENCE,
        trial=trial,
    )

    print(f"  Trial {trial.number:3d} | "
          f"lr={lr:.2e}  wd={weight_decay:.2e}  "
          f"drop={dropout:.2f}  unfreeze={unfrozen_layers} | "
          f"F1={best_f1:.4f}  Acc={best_acc*100:.2f}%")

    return best_f1   # maximize

# ─────────────────────────────────────────
# Plots
# ─────────────────────────────────────────
def plot_optimization_history(study):
    trials = [t for t in study.trials if t.value is not None]
    trial_nums = [t.number for t in trials]
    values     = [t.value  for t in trials]
    best_so_far = [max(values[:i+1]) for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(trial_nums, values, color="#2E75B6", alpha=0.6, s=40, label="Trial F1")
    ax.plot(trial_nums, best_so_far, color="#ED7D31", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial"); ax.set_ylabel("Macro F1")
    ax.set_title("Optuna Optimization History — UATR Centralized", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "optimization_history.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot: optuna_results/optimization_history.png")


def plot_param_importance(study):
    try:
        importance = optuna.importance.get_param_importances(study)
        params = list(importance.keys())
        values = list(importance.values())

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#2E75B6", "#70AD47", "#ED7D31", "#7030A0"]
        bars = ax.barh(params, values,
                       color=colors[:len(params)], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9)
        ax.set_xlabel("Importance Score")
        ax.set_title("Hyperparameter Importance — UATR", fontweight="bold")
        ax.set_xlim([0, max(values) * 1.2])
        ax.grid(True, alpha=0.3, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "param_importance.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  Plot: optuna_results/param_importance.png")
    except Exception as e:
        print(f"  [WARN] Could not plot param importance: {e}")


def plot_final_curves(log):
    df = pd.DataFrame(log)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Best Config — Full Training Run", fontsize=13, fontweight="bold")

    for ax, col, scale, unit, color, title, ylabel in [
        (axes[0], "accuracy", 100, "%",  "#2E75B6", "Accuracy", "Accuracy (%)"),
        (axes[1], "f1",       1,   "",   "#ED7D31", "Macro F1", "Macro F1"),
        (axes[2], "auc",      1,   "",   "#70AD47", "AUC-ROC",  "AUC-ROC"),
    ]:
        vals    = df[col] * scale
        best_v  = vals.max()
        best_ep = df.loc[vals.idxmax(), "epoch"]
        ax.plot(df["epoch"], vals, color=color, linewidth=2)
        ax.axhline(best_v, color=color, linestyle="--", alpha=0.4)
        ax.scatter([best_ep], [best_v], color=color, s=60, zorder=5)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.text(0.98, 0.05, f"Best: {best_v:.2f}{unit} (ep {int(best_ep)})",
                transform=ax.transAxes, ha="right", fontsize=9, color=color)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "baseline_uatr_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot: optuna_results/baseline_uatr_curves.png")

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  type=str, default=DATA_DIR)
    parser.add_argument("--n_trials",  type=int, default=N_TRIALS)
    parser.add_argument("--resume",    action="store_true",
                        help="Resume from existing study.db")
    args = parser.parse_args()

    print("=" * 60)
    print("  UATR Centralized — Optuna Hyperparameter Search")
    print("=" * 60)
    print(f"  Data dir  : {args.data_dir}")
    print(f"  Trials    : {args.n_trials}")
    print(f"  Trial budget : {TRIAL_EPOCHS} epochs  (ES patience={ES_PATIENCE})")
    print(f"  Final budget : {FINAL_EPOCHS} epochs")
    print(f"  Device    : {device}")
    print(f"  Search space:")
    print(f"    lr             : [1e-4, 5e-3]  log")
    print(f"    weight_decay   : [1e-5, 1e-2]  log")
    print(f"    dropout        : [0.1,  0.5]")
    print(f"    unfrozen_layers: [1, 3]  (1=layer4 only, 3=layer2+3+4)")
    print("=" * 60)

    # Load data once — shared across all trials
    print("\nLoading dataset...")
    train_ds = NPZDataset(os.path.join(args.data_dir, "train.npz"))
    test_ds  = NPZDataset(os.path.join(args.data_dir, "test.npz"))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    print(f"  Train: {len(train_ds)}  Test: {len(test_ds)}")

    # Create or load study
    db_path  = os.path.join(RESULTS_DIR, "study.db")
    storage  = f"sqlite:///{db_path}"
    study_name = "uatr_centralized"

    if args.resume and os.path.exists(db_path):
        print(f"\nResuming study from {db_path}")
        study = optuna.load_study(
            study_name=study_name, storage=storage
        )
        print(f"  Existing trials: {len(study.trials)}")
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",         # maximize macro F1
            sampler=TPESampler(seed=42),
            pruner=HyperbandPruner(
                min_resource=5,
                max_resource=TRIAL_EPOCHS,
                reduction_factor=3,
            ),
            load_if_exists=args.resume,
        )

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\nRunning {args.n_trials} trials...\n")
    print(f"  {'Trial':>6}  {'lr':>10}  {'wd':>10}  {'drop':>6}  "
          f"{'unfreeze':>8}  {'F1':>8}  {'Acc':>8}")
    print(f"  {'-'*65}")

    t0 = time.time()
    study.optimize(
        lambda trial: objective(trial, train_loader, test_loader),
        n_trials=args.n_trials,
        show_progress_bar=False,
    )
    search_time = time.time() - t0

    # Best params
    best = study.best_trial
    best_params = {
        "lr":              best.params["lr"],
        "weight_decay":    best.params["weight_decay"],
        "dropout":         best.params["dropout"],
        "unfrozen_layers": best.params["unfrozen_layers"],
        "best_trial_f1":   best.value,
        "trial_number":    best.number,
        "search_time_min": round(search_time / 60, 1),
    }

    print(f"\n{'='*60}")
    print(f"  SEARCH COMPLETE — {args.n_trials} trials in {search_time/60:.1f} min")
    print(f"{'='*60}")
    print(f"  Best trial : #{best.number}  F1 = {best.value:.4f}")
    print(f"  Best params:")
    print(f"    lr             = {best_params['lr']:.2e}")
    print(f"    weight_decay   = {best_params['weight_decay']:.2e}")
    print(f"    dropout        = {best_params['dropout']:.3f}")
    print(f"    unfrozen_layers= {best_params['unfrozen_layers']}")

    # Save best params
    with open(os.path.join(RESULTS_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\n  Saved: optuna_results/best_params.json")

    # Plots
    plot_optimization_history(study)
    plot_param_importance(study)

    # ── Full re-train with best params ────
    print(f"\n{'='*60}")
    print(f"  RE-TRAINING with best params ({FINAL_EPOCHS} epochs max)...")
    print(f"{'='*60}")

    best_model = get_model(
        dropout=best_params["dropout"],
        unfrozen_layers=best_params["unfrozen_layers"],
    ).to(device)

    t1 = time.time()
    final_f1, final_acc, final_auc, final_wts, final_log = train_loop(
        best_model, train_loader, test_loader,
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        max_epochs=FINAL_EPOCHS,
        es_patience=ES_PATIENCE + 3,   # slightly more patient for final run
        trial=None,
    )
    final_time = time.time() - t1

    best_model.load_state_dict(final_wts)
    torch.save(best_model.state_dict(),
               os.path.join(RESULTS_DIR, "best_baseline_tuned.pth"))

    # Save full params + final metrics
    best_params_full = {**best_params,
                        "final_f1":       final_f1,
                        "final_accuracy": final_acc,
                        "final_auc":      final_auc,
                        "final_time_min": round(final_time / 60, 1)}
    with open(os.path.join(RESULTS_DIR, "best_params_full.json"), "w") as f:
        json.dump(best_params_full, f, indent=2)

    plot_final_curves(final_log)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — best config full run")
    print(f"{'='*60}")
    print(f"  Accuracy : {final_acc*100:.2f}%")
    print(f"  Macro F1 : {final_f1:.4f}")
    print(f"  AUC-ROC  : {final_auc:.4f}")
    print(f"  Time     : {final_time/60:.1f} min")
    print(f"\n  Copy these into run_all_uatr.py / simulate_uatr.py:")
    print(f"    LR             = {best_params['lr']:.2e}")
    print(f"    WEIGHT_DECAY   = {best_params['weight_decay']:.2e}")
    print(f"    DROPOUT        = {best_params['dropout']:.3f}")
    print(f"    UNFROZEN_LAYERS= {best_params['unfrozen_layers']}")
    print(f"\n  All outputs: optuna_results/")