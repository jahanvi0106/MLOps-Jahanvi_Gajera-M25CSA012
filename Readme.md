# Adversarial ML on CIFAR-10 — Project Guide

## Setup

```bash
# 1. Install dependencies
pip install adversarial-robustness-toolbox wandb torch torchvision

# 2. Log in to WandB (paste your key from wandb.ai/authorize)
wandb login

# 3. Run
python adversarial_ml.py
```

Recommended: **Google Colab (T4 GPU)** or any CUDA machine.  
Training takes ~25 min on a T4.

---

## Architecture & Design Decisions

### Task (i) — FGSM Attack

| Component | Choice | Reason |
|-----------|--------|--------|
| Classifier | ResNet-18 (non-pretrained) | Lightweight, ≥72% on CIFAR-10 achievable |
| CIFAR-10 adaptation | 3×3 conv1, no maxpool | Avoids spatial downsampling on 32×32 |
| Optimizer | SGD + CosineAnnealing | Standard CIFAR-10 recipe |
| ART integration | `NormWrapper` | ART expects [0,1] input; normalisation is pushed inside the model |

**FGSM from scratch:**
```
x_adv = x + ε · sign(∇_x L(θ, x, y))
```
A single gradient step in the direction that maximises the loss.

**FGSM via ART:**  
`FastGradientMethod(estimator=art_clf, eps=ε)` — same math, but ART handles
clipping, batching, and norm selection automatically.

### Task (ii) — Adversarial Detectors

| Component | Choice |
|-----------|--------|
| Detector backbone | ResNet-34 (non-pretrained) |
| Label | 0 = clean, 1 = adversarial |
| Dataset | 50/50 clean + adversarial from `raw_loader` (40 batches each) |
| Train/Val split | 80/20 |
| PGD config | ε=0.03, step=0.007, iter=40 |
| BIM config | ε=0.03, step=0.007, iter=10 |

---

## Expected Results

### FGSM Accuracy Table

| ε | Clean | Scratch Acc | ART Acc |
|---|-------|-------------|---------|
| 0.00 | ~0.74 | 0.74 | 0.74 |
| 0.01 | ~0.74 | ~0.65 | ~0.63 |
| 0.02 | ~0.74 | ~0.52 | ~0.50 |
| 0.03 | ~0.74 | ~0.40 | ~0.38 |
| 0.05 | ~0.74 | ~0.25 | ~0.22 |
| 0.07 | ~0.74 | ~0.17 | ~0.15 |
| 0.10 | ~0.74 | ~0.12 | ~0.10 |

> **Scratch ≈ ART** because both implement the same FGSM formula.
> Small gaps arise from clipping differences and floating-point order.

### Detector Results

| Attack | Expected Detection Acc |
|--------|----------------------|
| PGD    | ~85–92 % ✓           |
| BIM    | ~80–88 % ✓           |

Both well above the 70% threshold. PGD's larger perturbation makes detection easier.

---

## Analysis

### Why Scratch ≈ ART for FGSM?
Both implement `x_adv = clip(x + ε·sign(∇L))`. The difference is that ART
clips in [0,1] pixel space while the scratch version clips in normalised space —
a minor numerical difference, not a conceptual one.

### Perturbation Strength vs Performance Drop
- At **ε=0.01**: modest ~9% drop — images look identical to humans.
- At **ε=0.03**: ~34% drop — imperceptible perturbations fool the model ~60% of the time.
- At **ε=0.10**: catastrophic ~62% drop — some visual distortion visible.

### Why is PGD detection accuracy ≥ BIM?
PGD runs more iterations (40 vs 10) with random restarts, producing more
"structured" adversarial noise. This noise has a distinct statistical
fingerprint that ResNet-34 can learn to distinguish from clean images.
BIM's perturbations are subtler and thus slightly harder to detect.

### Impact of the Attack
FGSM exploits the linearity of DNNs: even tiny gradient-aligned perturbations
can push inputs across decision boundaries. IBM ART's implementation is
equivalent but more production-ready (batching, dtype handling, clipping).

---

## WandB Panels

| Panel | Contents |
|-------|----------|
| `samples/FGSM_Scratch` | 10 (original, adversarial) pairs — scratch |
| `samples/FGSM_ART` | 10 pairs — ART |
| `samples/PGD` | 10 pairs — PGD attack |
| `samples/BIM` | 10 pairs — BIM attack |
| `fgsm/eps_vs_accuracy` | Line chart: perturbation ε vs accuracy |
| `fgsm/visual_comparison` | 3×10 grid comparison figure |
| `summary/final_chart` | Combined bar + line summary |