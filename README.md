# Pathwise Bayesian Model Averaging (PBMA)

> PAC-Bayesian Aggregation over Continuous Training Trajectories

---

## Overview

Modern deep generative models are trained via stochastic optimisation, producing a **continuous trajectory of parameters**. Standard practice discards this trajectory, selecting only the final or best checkpoint.

**PBMA** treats the training trajectory as a **continuous ensemble**, aggregating checkpoints in probability space with formal PAC-Bayesian guarantees — rather than collapsing training into a single point estimate.

---

## Core Idea

Instead of picking the final/best model, PBMA defines a **weighted mixture over checkpoints**:

$$p_Q(x) = \sum_j w_j \, p_{\theta_j}(x), \qquad w_j \propto \exp(-\eta \hat{L}_j)$$

Checkpoints with lower validation loss receive higher weight via **Gibbs weighting**, producing a more stable and expressive model with no additional training cost.

---

## Theoretical Guarantees

| Theorem | Result |
|---|---|
| Oracle Bound | PBMA performs nearly as well as the best single checkpoint |
| Jensen Improvement | Mixture strictly improves over average log-loss |
| Variational Optimality | Gibbs weights are the exact minimiser of empirical risk + KL |
| PAC-Bayes Bound | Generalisation guarantee depending on KL(Q\|\|P) and sample size |

---

## Project Structure

```
pbma/
  configs/          # YAML configs for each experiment
  datasets/         # Dataset loading and preprocessing
  models/           # VAE, RealNVP, Glow architectures
  training/         # Trainer, checkpoint manager, EMA
  pbma/             # Core PBMA: weighting, aggregation, metrics
  scripts/          # Train, evaluate, sample scripts
  utils/            # Seeding, logging, plotting
```

---

## Experimental Plan

| Phase | Dataset | Model | Goal |
|---|---|---|---|
| 1 | MNIST | VAE | Proof of concept |
| 2 | MNIST | RealNVP | Exact likelihood test |
| 3 | CIFAR-10 | RealNVP / Glow | Publishable benchmark |

### Baselines
- Last checkpoint
- Best validation checkpoint
- EMA (Exponential Moving Average)
- Uniform checkpoint mixture
- PBMA-Gibbs *(ours)*
- PBMA-Top-k Gibbs *(ours)*

### Metrics
- Test NLL / Bits per dim
- FID
- Seed-to-seed variance

---

## Setup

```bash
# Clone the repo
git clone https://github.com/benjaminsw/PBMA.git
cd PBMA

# Create and activate virtual environment
python3 -m venv PBMA-ENV
source PBMA-ENV/bin/activate

# Install dependencies
pip install -r pbma/requirements.txt
```

---

## Quick Start

```bash
# Train a model with checkpoint saving
python scripts/train.py --config configs/mnist_realnvp.yaml

# Validate all checkpoints
python scripts/validate_checkpoints.py

# Run PBMA evaluation
python scripts/run_pbma_eval.py --eta 10 --top_k 10

# Sample from PBMA mixture
python scripts/sample_pbma.py
```

---

## PBMA Weighting Example

```python
import torch

# val_losses: list of validation NLL per checkpoint
losses = torch.tensor(val_losses)
shifted = losses - losses.min()
weights = torch.softmax(-eta * shifted, dim=0)

# Compute mixture log-prob in log-space
log_probs = torch.stack([model.log_prob(x) for model in models], dim=0)  # [K, B]
log_w = torch.log(weights).unsqueeze(1)
log_mix = torch.logsumexp(log_w + log_probs, dim=0)  # [B]
nll = -log_mix.mean()
```

---

## Expected Results

- PBMA ≤ best checkpoint NLL + small PAC-Bayes penalty
- Consistent improvement over EMA
- Reduced variance across random seeds
- No additional training cost

---

## References

- McAllester (1999) — PAC-Bayes bounds
- Dinh et al. (2017) — RealNVP
- Kingma & Welling (2014) — VAE
- Ho et al. (2020) — Diffusion models

---

## License

MIT
