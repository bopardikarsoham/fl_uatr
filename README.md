# 🚢 Federated Learning for Underwater Acoustic Target Recognition (UATR)

> **EECE 5643 — Simulation & Performance Evaluation | Northeastern University | Spring 2026**  
> Team 6: Soham Bopardikar · Saunak Samantray · Krishna Prasad Selvaraj

---

## Overview

This project benchmarks four federated learning algorithms on the **ShipsEar** maritime acoustic dataset across a full heterogeneity spectrum (Dirichlet α = 0.1 → 10.0), with client dropout robustness experiments. A ResNet-18 backbone with Optuna-tuned hyperparameters serves as the shared model across all FL clients.

The central question: **how much does data heterogeneity hurt FL, and which algorithm recovers best?**

**Short answer:** At moderate heterogeneity (α ≥ 0.7) all four algorithms match the centralized baseline within 2.5pp. Below α = 0.5 they diverge — and dropout paradoxically *improves* the worst configs by up to +9.13pp.

---

## Results at a Glance

| Config | FedAvg | FedProx | Per-FedAvg | Ditto |
|--------|--------|---------|------------|-------|
| α=1.0 c=5 ⭐ | **97.95%** | 97.70% | 97.23% | 97.46% |
| α=0.7 c=5 | 95.89% | 95.50% | 94.47% | 94.41% |
| α=0.7 c=10 | 95.57% | 95.23% | 94.57% | 94.50% |
| α=0.5 c=10 | 93.04% | 92.49% | 92.60% | 90.23% |
| α=0.5 c=5 | 88.84% | 88.15% | 88.20% | 88.13% |
| α=0.3 c=10 | 87.54% | 87.86% | 85.19% | **88.21%** |
| α=0.3 c=5 | 85.88% | 81.26% | **86.95%** | 83.33% |
| α=0.1 c=10 | 65.81% | 86.38% | **86.85%** | 71.73% |
| α=0.1 c=5 | 67.97% | 66.08% | **72.46%** | 70.20% |

**Centralized baseline (Optuna-tuned):** 97.98% · F1 = 0.98 · AUC = 1.00

---

## Algorithms

| Algorithm | Key Idea | Hyperparameter |
|-----------|----------|----------------|
| **FedAvg** | Weighted average of local SGD updates | — |
| **FedProx** | Adds proximal term `(μ/2)‖w − w_global‖²` to client loss | μ = 0.001 |
| **Per-FedAvg** | MAML-based: learns a global initialization that adapts in one inner step | α_inner = 5×10⁻⁴ |
| **Ditto** | Maintains a global model + per-client personalized model with proximal pull | λ = 2.0 |

All hyperparameters tuned via 5-point grid search on α=1.0 c=5 before full runs.

---

## Dataset

**ShipsEar** — underwater ship-radiated noise recordings, 5 classes:

| Class | Label |
|-------|-------|
| 0 | Small Working Vessel |
| 1 | Small Recreational / Utility |
| 2 | Passenger Ferry |
| 3 | Large Commercial |
| 4 | Background Noise |

Data partitioned using **Dirichlet(α)** distribution across clients. Lower α = more heterogeneous (non-IID). Preprocessing: spectrogram → 3-channel image → ResNet-18 input.

---

## Repository Structure

```
.
├── utils_uatr2.py           # Shared config, dataset, model, plotting — single source of truth
├── fedavg_uatr.py           # FedAvg implementation
├── fedprox_uatr.py          # FedProx (Li et al., MLSys 2020)
├── perfedavg_uatr.py        # Per-FedAvg (Fallah et al., NeurIPS 2020)
├── ditto_uatr.py            # Ditto (Li et al., ICML 2021)
├── run_dropout.sh           # All 16 dropout experiment commands
│
├── shipsear_fl/             # Partitioned data (not tracked)
│   └── niid_alpha{α}_c{c}/
│       ├── train/{0..c-1}.npz
│       └── test/{0..c-1}.npz
│
├── fedavg_results/          # Per-run CSVs + publication tables
├── fedprox_results/
├── perfedavg_results/
├── ditto_results/
│
├── fedavg_plots/            # Convergence curves + combined plots
├── fedprox_plots/
├── perfedavg_plots/
├── ditto_plots/
│
└── logs/                    # Full stdout logs per run
```

---

## Setup

```bash
pip install flwr[simulation] torch torchvision scikit-learn pandas matplotlib optuna
```

Tested on Python 3.12, Flower 1.x, PyTorch 2.x. GPU strongly recommended (Colab A100 used for full runs).

---

## Running Experiments

### Single config (quick test)

```bash
# FedAvg
python fedavg_uatr.py --alpha 1.0 --clients 5

# FedProx with tuned mu
python fedprox_uatr.py --mu 0.001 --alpha 1.0 --clients 5

# Per-FedAvg with tuned alpha_inner
python perfedavg_uatr.py --alpha_inner 5e-4 --alpha 1.0 --clients 5

# Ditto with tuned lambda
python ditto_uatr.py --lam 2.0 --alpha 1.0 --clients 5
```

### All 12 configs (full sweep)

```bash
python fedavg_uatr.py
python fedprox_uatr.py --mu 0.001
python perfedavg_uatr.py --alpha_inner 5e-4
python ditto_uatr.py --lam 2.0
```

### Hyperparameter tuning (grid search)

```bash
python fedprox_uatr.py --run_tune
python perfedavg_uatr.py --run_tune
python ditto_uatr.py --run_tune
```

### Resume interrupted runs

```bash
python fedavg_uatr.py --resume
```

### Client dropout experiments

```bash
# 10% dropout on best config (α=0.7)
python fedavg_uatr.py --alpha 0.7 --clients 5 --dropout 0.1

# 30% dropout on worst config (α=0.3)
python fedprox_uatr.py --mu 0.001 --alpha 0.3 --clients 5 --dropout 0.3

# Or run all 16 dropout experiments at once
bash run_dropout.sh
```

---

## Key Findings

### RQ1 — How close can FL get to centralized?
FedAvg at α=1.0 c=5 reaches **97.95%** — only **0.03pp** below the centralized baseline of 97.98%. The privacy cost at moderate heterogeneity is negligible.

### RQ2 — How does heterogeneity affect performance?
Three distinct regimes emerge:
- **Near-IID (α ≥ 0.7):** All four algorithms within 1–2pp of each other. 94–98%.
- **Transition (α = 0.5–0.3):** Algorithms begin to diverge. Per-FedAvg leads; FedProx suffers data starvation at α=0.3 c=5 (81.26%).
- **Extreme non-IID (α = 0.1):** FedAvg collapses −32pp. Per-FedAvg and FedProx recover 20pp+ at c=10.

### RQ3 — Does personalization help?
Per-FedAvg is the strongest at extreme non-IID — **86.85%** at α=0.1 c=10 vs FedAvg's 65.81% (+21pp). Ditto underperforms at α=0.1 due to data starvation: with too few samples per client, the personalized model overfits to noise rather than adapting the global initialization.

### RQ4 — How resilient is FL to client dropout?
**Best configs (α=0.7):** All algorithms resilient within ±1pp under 30% dropout.  
**Worst configs (α=0.3):** Dropout *improves* accuracy — FedProx gains **+9.13pp** (81.26% → 90.39%). Hypothesis: at α=0.3 some clients carry conflicting gradient directions; removing 30% of them reduces update noise and the aggregate improves. This counter-intuitive effect is specific to the transition zone.

---

## Dropout Results

| Config | Baseline | 30% Dropout | Δ |
|--------|----------|-------------|---|
| FedAvg α=0.7 c=5 | 95.89% | 96.00% | +0.11pp |
| FedProx α=0.7 c=5 | 95.50% | 96.10% | +0.60pp |
| Per-FedAvg α=0.7 c=10 | 94.57% | 93.94% | −0.63pp |
| Ditto α=0.7 c=10 | 94.50% | 95.56% | +1.06pp |
| FedAvg α=0.3 c=5 | 85.88% | 90.83% | **+4.95pp** |
| FedProx α=0.3 c=5 | 81.26% | 90.39% | **+9.13pp** ⬆ |
| Per-FedAvg α=0.3 c=10 | 85.19% | 87.86% | +2.67pp |
| Ditto α=0.3 c=5 | 83.33% | 89.45% | **+6.12pp** |

---

## Model

ResNet-18 (ImageNet pretrained) with partial fine-tuning:
- **Frozen:** layer1, conv1, bn1
- **Unfrozen:** layer2, layer3, layer4, fc
- **Head:** `Dropout(0.342) → Linear(512, 5)`

Hyperparameters tuned via **Optuna TPE** on centralized baseline:

| Parameter | Value |
|-----------|-------|
| Learning rate | 3.05 × 10⁻⁴ |
| Weight decay | 1.73 × 10⁻⁴ |
| Dropout | 0.342 |
| Unfrozen layers | 3 (layer2–4) |
| Batch size | 32 |
| Local epochs | 3 |
| FL rounds | 30 |

The same Optuna-tuned hyperparameters are used across all four FL algorithms for fair comparison.

---

## FL Configuration

```python
# Dirichlet partitioning — 12 configs
COMBINATIONS = [
    (10.0, 5), (10.0, 10),
    (1.0,  5), (1.0,  10),
    (0.7,  5), (0.7,  10),
    (0.5,  5), (0.5,  10),
    (0.3,  5), (0.3,  10),
    (0.1,  5), (0.1,  10),
]

NUM_ROUNDS   = 30
LOCAL_EPOCHS = 3
BATCH_SIZE   = 32
```

Simulation via **Flower** (`flwr[simulation]`) with Ray backend. Each client allocated `num_cpus=1, num_gpus=0.25` for parallel simulation.

---

## References

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017 — **FedAvg**
- Li et al., *Federated Optimization in Heterogeneous Networks*, MLSys 2020 — **FedProx**
- Fallah et al., *Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach*, NeurIPS 2020 — **Per-FedAvg**
- Li et al., *Ditto: Fair and Robust Federated Learning Through Personalization*, ICML 2021 — **Ditto**
- Santos-Dominguez et al., *ShipsEar: An underwater vessel noise database*, Applied Acoustics 2016 — **Dataset**

---

## Citation

If you use this code or results, please cite:

```bibtex
@misc{uatr_fl_2026,
  title  = {Federated Learning for Underwater Acoustic Target Recognition},
  author = {Bopardikar, Soham and Samantray, Saunak and Selvaraj, Krishna Prasad},
  year   = {2026},
  note   = {EECE 5643, Northeastern University}
}
```

---

*Built with [Flower](https://flower.ai) · [PyTorch](https://pytorch.org) · [ShipsEar](http://atlanttic.uvigo.es/underwaternoise/)*
