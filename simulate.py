"""
simulate_uatr.py — Federated Learning for UATR (ShipsEar)

Data structure expected:
  shipsear_fl/
    niid_alpha{ALPHA}_c{NUM_CLIENTS}/
      train/  0.npz, 1.npz, ... (NUM_CLIENTS-1).npz
      test/   0.npz, 1.npz, ... (NUM_CLIENTS-1).npz

Each .npz contains:
  data:    (N, 3, 224, 224) float32  — mel-spectrograms, already normalized
  targets: (N,)             int64    — class labels 0-4

Classes:
  0 = Small Working  (Fishing, Trawler, Tug)
  1 = Small Rec/Utility (Motorboat, Pilot, Sailboat)
  2 = Passenger Ferry
  3 = Large Commercial (Ocean Liner, Cargo)
  4 = Background Noise

Usage:
  python simulate_uatr.py                         # default: alpha=0.1, 10 clients
  ALPHA=1.0   NUM_CLIENTS=5  python simulate_uatr.py
  ALPHA=10.0  NUM_CLIENTS=10 python simulate_uatr.py
"""

import flwr as fl
from flwr.simulation import start_simulation
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, Context
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, EvaluateRes

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.models import resnet18, ResNet18_Weights

import numpy as np
import pandas as pd
import os, time, json
from typing import List
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)

os.environ["RAY_DEDUP_LOGS"]      = "0"
os.environ["RAY_DISABLE_METRICS"] = "1"

# ──────────────────────────────────────────
# CONFIG — override via env vars or edit here
# ──────────────────────────────────────────
DATA_ROOT    = "./shipsear_fl"
ALPHA        = float(os.environ.get("ALPHA",       "10"))
NUM_CLIENTS  = int(os.environ.get("NUM_CLIENTS",   "5"))
NUM_ROUNDS   = int(os.environ.get("NUM_ROUNDS",    "50"))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS",  "3"))
BATCH_SIZE   = 32
LR           = 1e-3
NUM_CLASSES  = 5
RESULTS_DIR  = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

RUN_NAME = f"fedavg_alpha{ALPHA}_c{NUM_CLIENTS}"

CLASS_NAMES = {
    0: "Small Working",
    1: "Small Rec/Utility",
    2: "Passenger Ferry",
    3: "Large Commercial",
    4: "Background Noise",
}

# ──────────────────────────────────────────
# Device
# ──────────────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

def progress_bar(current, total, width=30):
    filled = int(width * current / total)
    return f"[{'█'*filled}{'░'*(width-filled)}] {current}/{total}"

# ──────────────────────────────────────────
# Dataset — loads directly from .npz
# No transforms needed — data is already
# (N, 3, 224, 224) float32, normalized
# ──────────────────────────────────────────
class NPZDataset(Dataset):
    def __init__(self, npz_path):
        d             = np.load(npz_path)
        self.data     = torch.from_numpy(d["data"])      # (N, 3, 224, 224) float32
        self.targets  = torch.from_numpy(d["targets"])   # (N,) int64
        self.n        = len(self.targets)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx].long()


def get_data_dir():
    return os.path.join(DATA_ROOT, f"niid_alpha{ALPHA}_c{NUM_CLIENTS}")


def get_client_loaders(client_id):
    base = get_data_dir()
    train_ds = NPZDataset(os.path.join(base, "train", f"{client_id}.npz"))
    test_ds  = NPZDataset(os.path.join(base, "test",  f"{client_id}.npz"))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    return train_loader, test_loader


def get_global_test_loader():
    """Concatenate all clients' test sets for global evaluation."""
    base    = get_data_dir()
    all_ds  = [NPZDataset(os.path.join(base, "test", f"{i}.npz"))
               for i in range(NUM_CLIENTS)]
    return DataLoader(ConcatDataset(all_ds), batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=0)

# ──────────────────────────────────────────
# Model — ResNet-18 pretrained, 5-class head
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

_tmp      = get_model()
total_p   = sum(p.numel() for p in _tmp.parameters())
train_p   = sum(p.numel() for p in _tmp.parameters() if p.requires_grad)
del _tmp

print(f"Device          : {device}")
print(f"Run             : {RUN_NAME}")
print(f"Alpha           : {ALPHA}  |  Clients: {NUM_CLIENTS}")
print(f"Rounds          : {NUM_ROUNDS}  |  Local epochs: {LOCAL_EPOCHS}")
print(f"Trainable params: {train_p:,} / {total_p:,}  ({100*train_p/total_p:.1f}%)")

# ──────────────────────────────────────────
# Evaluate — full metrics for 5-class
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

    y   = np.array(all_labels)
    yh  = np.array(all_preds)
    yp  = np.array(all_probs)

    acc  = float(accuracy_score(y, yh))
    f1   = float(f1_score(y, yh, average="macro", zero_division=0))
    prec = float(precision_score(y, yh, average="macro", zero_division=0))
    rec  = float(recall_score(y, yh,    average="macro", zero_division=0))
    cm   = confusion_matrix(y, yh, labels=list(range(NUM_CLASSES))).tolist()

    # AUC — only if all classes present in batch
    try:
        auc = float(roc_auc_score(y, yp, multi_class="ovr", average="macro"))
    except ValueError:
        auc = 0.0   # happens when a class has no samples in this client's test set

    return acc, f1, auc, prec, rec, cm

# ──────────────────────────────────────────
# Strategy with progress + per-round logging
# ──────────────────────────────────────────
class VerboseFedAvg(FedAvg):

    def __init__(self, num_rounds, run_name, **kwargs):
        super().__init__(**kwargs)
        self.num_rounds               = num_rounds
        self.run_name                 = run_name
        self.round_times: List[float] = []
        self.log:         List[dict]  = []
        self._round_start             = 0.0

        print(f"\n{'─'*58}")
        print(f"  FL Simulation: {run_name}")
        print(f"  Clients: {NUM_CLIENTS}  |  Rounds: {num_rounds}  |  "
              f"Local epochs: {LOCAL_EPOCHS}")
        print(f"  Classes: {NUM_CLASSES} ({', '.join(CLASS_NAMES.values())})")
        print(f"{'─'*58}\n")

    def aggregate_fit(self, server_round, results, failures):
        self._round_start = time.time()
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        elapsed = time.time() - self._round_start
        self.round_times.append(elapsed)

        accs, f1s, aucs = [], [], []
        for _, r in results:
            m = r.metrics
            if "accuracy" in m: accs.append(m["accuracy"])
            if "f1"       in m: f1s.append(m["f1"])
            if "auc"      in m: aucs.append(m["auc"])

        avg_acc = float(np.mean(accs)) * 100 if accs else 0.0
        avg_f1  = float(np.mean(f1s))         if f1s  else 0.0
        avg_auc = float(np.mean(aucs))         if aucs else 0.0

        row = {
            "run": self.run_name, "round": server_round,
            "alpha": ALPHA, "num_clients": NUM_CLIENTS,
            "accuracy": avg_acc / 100, "f1": avg_f1,
            "auc": avg_auc, "time": elapsed,
        }
        self.log.append(row)

        bar   = progress_bar(server_round, self.num_rounds)
        eta   = np.mean(self.round_times) * (self.num_rounds - server_round)
        prev  = self.log[-2]["accuracy"] * 100 if len(self.log) > 1 else avg_acc
        delta = f"  ({avg_acc - prev:+.2f}%)"

        print(f"  Round {server_round:02d}/{self.num_rounds}  {bar}")
        print(f"           Accuracy : {avg_acc:.2f}%{delta}")
        print(f"           F1 (macro): {avg_f1:.4f}  |  AUC: {avg_auc:.4f}")
        print(f"           Time     : {elapsed:.1f}s  |  ETA: {eta:.0f}s\n")

        if server_round == self.num_rounds:
            best_acc = max(r["accuracy"] for r in self.log) * 100
            best_f1  = max(r["f1"]       for r in self.log)
            print(f"{'─'*58}")
            print(f"  Training complete!")
            print(f"  Best Accuracy : {best_acc:.2f}%")
            print(f"  Best F1       : {best_f1:.4f}")
            print(f"  Total time    : {sum(self.round_times):.0f}s")
            print(f"{'─'*58}\n")

            log_path = os.path.join(RESULTS_DIR, f"{self.run_name}_rounds.csv")
            pd.DataFrame(self.log).to_csv(log_path, index=False)
            print(f"  Round log saved to {log_path}")

        return aggregated

# ──────────────────────────────────────────
# Flower Client
# ──────────────────────────────────────────
class UATRClient(fl.client.NumPyClient):

    def __init__(self, client_id):
        self.client_id   = client_id
        self.model       = get_model().to(device)
        self.criterion   = nn.CrossEntropyLoss()
        self.optimizer   = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=LR, weight_decay=1e-4
        )
        self.train_loader, self.test_loader = get_client_loaders(client_id)
        n_train = len(self.train_loader.dataset)
        n_test  = len(self.test_loader.dataset)
        print(f"  Client {client_id}: {n_train} train  |  {n_test} test")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(),
                              [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        epochs = config.get("local_epochs", LOCAL_EPOCHS)
        for _ in range(epochs):
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
            "accuracy":  float(acc),
            "f1":        float(f1),
            "auc":       float(auc),
            "precision": float(prec),
            "recall":    float(rec),
        }


def client_fn(context: Context) -> fl.client.Client:
    client_id = int(context.node_config["partition-id"])
    return UATRClient(client_id=client_id).to_client()

# ──────────────────────────────────────────
# Run
# ──────────────────────────────────────────
if __name__ == "__main__":
    initial_parameters = ndarrays_to_parameters(
        [val.cpu().numpy() for val in get_model().state_dict().values()]
    )

    strategy = VerboseFedAvg(
        num_rounds=NUM_ROUNDS,
        run_name=RUN_NAME,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda server_round: {
            "local_epochs": LOCAL_EPOCHS,
            "round": server_round,
        },
    )

    start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )