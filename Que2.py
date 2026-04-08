#!/usr/bin/env python3
"""
=============================================================================
Adversarial ML on CIFAR-10  —  Full Solution
=============================================================================
(i)  FGSM Attack: From Scratch vs IBM ART  (ResNet-18)
(ii) Adversarial Detection using ResNet-34 (PGD / BIM via ART)
     + WandB logging of 10 samples per attack type

SETUP (run once):
    pip install adversarial-robustness-toolbox wandb torch torchvision
    wandb login          # paste your API key from wandb.ai/authorize

Run:
    python adversarial_ml.py
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os, copy, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
warnings.filterwarnings("ignore")

import wandb
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (
    FastGradientMethod  as ART_FGSM,
    ProjectedGradientDescent as ART_PGD,
    BasicIterativeMethod     as ART_BIM,
)

# ─────────────────────────────────────────────────────────────────────────────
# Global Config
# ─────────────────────────────────────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 128
EPOCHS_CLF    = 10          # ResNet-18 classifier
EPOCHS_DET    = 20          # ResNet-34 binary detector
LR            = 0.1
FGSM_EPS_LIST = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
PGD_EPS       = 0.03
BIM_EPS       = 0.03
SEED          = 42
NUM_CLASSES   = 10
CLASSES       = ("plane","car","bird","cat","deer","dog","frog","horse","ship","truck")
CIFAR_MEAN    = np.array([0.4914, 0.4822, 0.4465])
CIFAR_STD     = np.array([0.2023, 0.1994, 0.2010])

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"[INFO] Device: {DEVICE}")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADERS
# ═════════════════════════════════════════════════════════════════════════════
def get_normalised_loaders():
    """Standard normalised loaders for ResNet-18 training/eval."""
    mean, std = tuple(CIFAR_MEAN), tuple(CIFAR_STD)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_ld, test_ld

def get_raw_loader():
    """[0,1] loader — needed by ART which expects unscaled pixel values."""
    ds = torchvision.datasets.CIFAR10("./data", train=False, download=False,
                                      transform=transforms.ToTensor())
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  MODEL DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════
def make_resnet18(num_out=NUM_CLASSES):
    """Non-pretrained ResNet-18 adapted for CIFAR-10 (32×32)."""
    m = models.resnet18(weights=None)
    m.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(512, num_out)
    return m.to(DEVICE)

def make_resnet34(num_out=2):
    """Non-pretrained ResNet-34 for binary clean/adv detection."""
    m = models.resnet34(weights=None)
    m.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(512, num_out)
    return m.to(DEVICE)

# Wrapper that applies CIFAR normalisation internally (for ART)
class NormWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        mn = torch.tensor(CIFAR_MEAN, dtype=torch.float32).view(1,3,1,1)
        st = torch.tensor(CIFAR_STD,  dtype=torch.float32).view(1,3,1,1)
        self.register_buffer("mn", mn)
        self.register_buffer("st", st)

    def forward(self, x):
        return self.base((x.to(self.mn.device) - self.mn) / self.st)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  TRAINING & EVAL HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, scheduler=None):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += out.argmax(1).eq(y).sum().item()
        total      += x.size(0)
    if scheduler:
        scheduler.step()
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += model(x).argmax(1).eq(y).sum().item()
        total   += x.size(0)
    return correct / total


# ═════════════════════════════════════════════════════════════════════════════
# 4.  (i-A) TRAIN ResNet-18  ≥ 72 % on CIFAR-10
# ═════════════════════════════════════════════════════════════════════════════
def train_classifier(train_ld, test_ld, run):
    print("\n" + "="*60)
    print("TASK i-A: Training ResNet-18 on clean CIFAR-10")
    print("="*60)
    model     = make_resnet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_CLF)
    best_acc  = 0.0

    for ep in range(1, EPOCHS_CLF + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, optimizer, criterion, scheduler)
        te_acc          = evaluate(model, test_ld)
        run.log({"clf/epoch":     ep,
                 "clf/train_loss": tr_loss,
                 "clf/train_acc":  tr_acc,
                 "clf/test_acc":   te_acc})
        print(f"  Epoch {ep:2d}/{EPOCHS_CLF} | loss={tr_loss:.4f} | "
              f"train={tr_acc:.4f} | test={te_acc:.4f}")
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), "best_resnet18.pth")

    model.load_state_dict(torch.load("best_resnet18.pth", map_location=DEVICE))
    run.summary["clean_accuracy"] = best_acc
    print(f"\n  ✓ Best test accuracy: {best_acc:.4f}")
    assert best_acc >= 0.72, f"  ✗ Accuracy {best_acc:.4f} < 0.72 — train longer!"
    return model, best_acc


# ═════════════════════════════════════════════════════════════════════════════
# 5.  (i-B) FGSM from scratch
# ═════════════════════════════════════════════════════════════════════════════
def fgsm_scratch(model, x, y, eps):
    """
    FGSM: x_adv = x + ε · sign(∇_x L(θ, x, y))
    Works on normalised tensors.
    """
    model.eval()
    x_in = x.clone().detach().to(DEVICE).requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x_in), y.to(DEVICE))
    loss.backward()
    return (x_in + eps * x_in.grad.sign()).detach()


def eval_fgsm_scratch(model, loader, eps, n_batches=20):
    # NOTE: do NOT use @torch.no_grad() here — fgsm_scratch needs gradients
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x_adv = fgsm_scratch(model, x, y, eps)   # requires grad internally
        with torch.no_grad():                      # no_grad only for inference
            correct += model(x_adv).argmax(1).eq(y.to(DEVICE)).sum().item()
        total += x.size(0)
    return correct / total


# ═════════════════════════════════════════════════════════════════════════════
# 6.  ART wrapper factory
# ═════════════════════════════════════════════════════════════════════════════
def build_art_classifier(wrapped_model):
    """
    wrapped_model: NormWrapper around base ResNet-18.
    ART receives [0,1] images; normalisation happens inside the model.
    """
    return PyTorchClassifier(
        model       = wrapped_model,
        loss        = nn.CrossEntropyLoss(),
        optimizer   = optim.SGD(wrapped_model.parameters(), lr=0.01),
        input_shape = (3, 32, 32),
        nb_classes  = NUM_CLASSES,
        clip_values = (0.0, 1.0),
        device_type = "gpu" if torch.cuda.is_available() else "cpu",
    )


# ═════════════════════════════════════════════════════════════════════════════
# 7.  (i-C) FGSM via ART
# ═════════════════════════════════════════════════════════════════════════════
def eval_fgsm_art(art_clf, raw_loader, eps, n_batches=20):
    attack  = ART_FGSM(estimator=art_clf, eps=eps)
    correct = total = 0
    for i, (x, y) in enumerate(raw_loader):
        if i >= n_batches:
            break
        x_adv  = attack.generate(x=x.numpy())
        preds  = art_clf.predict(x_adv).argmax(1)
        correct += (preds == y.numpy()).sum()
        total   += x.size(0)
    return correct / total


# ═════════════════════════════════════════════════════════════════════════════
# 8.  Image helpers
# ═════════════════════════════════════════════════════════════════════════════
def denorm(tensor):
    """Normalised tensor → numpy HWC [0,1]."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * CIFAR_STD + CIFAR_MEAN, 0, 1)

def raw2np(tensor):
    """Raw [0,1] tensor → numpy HWC."""
    return np.clip(tensor.cpu().numpy().transpose(1, 2, 0), 0, 1)


# ═════════════════════════════════════════════════════════════════════════════
# 9.  (i-D) Visual comparison figure  (Original / Scratch / ART)
# ═════════════════════════════════════════════════════════════════════════════
def save_comparison_figure(x_norm, x_adv_scratch, x_adv_art_np,
                            labels, preds_orig, preds_scratch, preds_art,
                            eps, path):
    n = min(10, x_norm.shape[0])
    fig, axes = plt.subplots(3, n, figsize=(2.2*n, 7))

    row_labels = [
        "Original",
        f"FGSM Scratch\n(ε={eps})",
        f"FGSM ART\n(ε={eps})",
    ]

    for col in range(n):
        imgs  = [denorm(x_norm[col]),
                 denorm(x_adv_scratch[col]),
                 np.clip(x_adv_art_np[col].transpose(1,2,0), 0, 1)]
        preds = [preds_orig[col], preds_scratch[col], preds_art[col]]
        for row in range(3):
            ax = axes[row, col]
            ax.imshow(imgs[row])
            color = "green" if preds[row] == labels[col] else "red"
            ax.set_title(CLASSES[preds[row]], fontsize=7, color=color, fontweight="bold")
            ax.axis("off")

    for row, lbl in enumerate(row_labels):
        axes[row, 0].set_ylabel(lbl, fontsize=9, rotation=0,
                                labelpad=95, va="center")

    plt.suptitle(f"FGSM Comparison  (ε={eps})  —  green=correct  red=wrong",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 10. WandB: log 10 sample pairs (clean + adversarial)
# ═════════════════════════════════════════════════════════════════════════════
def log_10_samples(run, panel_name,
                   orig_imgs, adv_imgs, labels,
                   preds_orig, preds_adv,
                   orig_is_normalised=True, adv_is_raw=True):
    """
    orig_imgs : torch tensor (N,3,32,32)
    adv_imgs  : torch tensor (N,3,32,32)  raw [0,1]  OR  normalised
    """
    panel = []
    for i in range(min(10, len(orig_imgs))):
        orig_np = denorm(orig_imgs[i]) if orig_is_normalised else raw2np(orig_imgs[i])
        adv_np  = raw2np(adv_imgs[i])  if adv_is_raw        else denorm(adv_imgs[i])

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.2, 1.8))
        a1.imshow(orig_np); a1.axis("off")
        a1.set_title(f"Clean\n{CLASSES[labels[i]]}", fontsize=7,
                     color="green" if preds_orig[i]==labels[i] else "red")
        a2.imshow(adv_np);  a2.axis("off")
        a2.set_title(f"Adv\n{CLASSES[preds_adv[i]]}", fontsize=7,
                     color="green" if preds_adv[i]==labels[i] else "red")
        plt.tight_layout()

        # rasterise to numpy without saving to disk
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        panel.append(wandb.Image(buf, caption=f"Sample {i}: "
                                 f"{'✓' if preds_adv[i]==labels[i] else '✗'}"))

    run.log({panel_name: panel})
    print(f"  WandB panel logged: {panel_name}")


# ═════════════════════════════════════════════════════════════════════════════
# 11. (ii) Build detector dataset
# ═════════════════════════════════════════════════════════════════════════════
def build_detector_dataset(art_clf, attack, raw_loader, n_batches=40):
    """
    Generate adversarial examples, mix with clean images.
    Labels: 0 = clean, 1 = adversarial.
    """
    X_clean, X_adv = [], []
    print(f"  Generating adversarial examples ({n_batches} batches)…", end="", flush=True)
    for i, (x, _) in enumerate(raw_loader):
        if i >= n_batches:
            break
        X_clean.append(x)
        X_adv.append(torch.tensor(attack.generate(x=x.numpy()), dtype=torch.float32))
        if (i+1) % 10 == 0:
            print(f" {i+1}", end="", flush=True)
    print()

    X_clean = torch.cat(X_clean)
    X_adv   = torch.cat(X_adv)
    X       = torch.cat([X_clean, X_adv])
    Y       = torch.cat([
        torch.zeros(len(X_clean), dtype=torch.long),
        torch.ones (len(X_adv),   dtype=torch.long),
    ])
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(SEED))
    return TensorDataset(X[perm], Y[perm])


def train_detector(model, train_ds, val_ds, run, tag):
    print(f"\n  Training detector [{tag}]…")
    tr_ld = DataLoader(train_ds, batch_size=64, shuffle=True)
    va_ld = DataLoader(val_ds,   batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_DET)
    best_val  = 0.0

    for ep in range(1, EPOCHS_DET + 1):
        tr_loss, tr_acc = train_one_epoch(model, tr_ld, optimizer, criterion, scheduler)
        va_acc          = evaluate(model, va_ld)
        run.log({f"det_{tag}/epoch":     ep,
                 f"det_{tag}/train_loss": tr_loss,
                 f"det_{tag}/train_acc":  tr_acc,
                 f"det_{tag}/val_acc":    va_acc})
        print(f"    Ep {ep:2d}/{EPOCHS_DET} | loss={tr_loss:.4f} | "
              f"train={tr_acc:.4f} | val={va_acc:.4f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), f"best_detector_{tag}.pth")

    model.load_state_dict(torch.load(f"best_detector_{tag}.pth", map_location=DEVICE))
    run.summary[f"detector_{tag}_val_acc"] = best_val
    status = "✓ PASS (≥70%)" if best_val >= 0.70 else "✗ FAIL (<70%)"
    print(f"  Best val accuracy [{tag}]: {best_val:.4f}  {status}")
    return model, best_val


# ═════════════════════════════════════════════════════════════════════════════
# 12. MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    # ── WandB init ────────────────────────────────────────────────────────────
    run = wandb.init(
        project="adversarial-ml-cifar10",
        name="fgsm-pgd-bim-resnet",
        config={
            "epochs_clf":     EPOCHS_CLF,
            "epochs_det":     EPOCHS_DET,
            "batch_size":     BATCH_SIZE,
            "fgsm_eps_list":  FGSM_EPS_LIST,
            "pgd_eps":        PGD_EPS,
            "bim_eps":        BIM_EPS,
            "seed":           SEED,
            "device":         str(DEVICE),
        },
    )

    train_ld, test_ld = get_normalised_loaders()
    raw_ld            = get_raw_loader()

    # ── (i-A) Train ResNet-18 ─────────────────────────────────────────────────
    clf, clean_acc = train_classifier(train_ld, test_ld, run)
    clf.eval()

    # # ── Build ART wrapper (normalisation inside model) ───────────────────────
    wrapped_clf = NormWrapper(clf).to(DEVICE)
    art_clf     = build_art_classifier(wrapped_clf)

    # ── (i-B & i-C) Sweep over ε ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("TASK i-B/C: FGSM Scratch vs ART — ε sweep")
    print("="*60)
    eps_list, scratch_accs, art_accs = [], [], []

    for eps in FGSM_EPS_LIST:
        acc_s = eval_fgsm_scratch(clf,     test_ld, eps, n_batches=20)
        acc_a = eval_fgsm_art    (art_clf, raw_ld,  eps, n_batches=20)
        print(f"  ε={eps:.2f}  Scratch={acc_s:.4f}  ART={acc_a:.4f}")
        run.log({"fgsm/eps": eps, "fgsm/scratch_acc": acc_s, "fgsm/art_acc": acc_a})
        eps_list.append(eps); scratch_accs.append(acc_s); art_accs.append(acc_a)

    # Perturbation vs accuracy plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(clean_acc, color="black", lw=2, ls="--",
               label=f"Clean ({clean_acc:.3f})")
    ax.plot(eps_list, scratch_accs, "o-r", lw=2, label="FGSM Scratch")
    ax.plot(eps_list, art_accs,     "s-b", lw=2, label="FGSM ART")
    ax.fill_between(eps_list, scratch_accs, art_accs,
                    alpha=0.12, color="purple")
    ax.set_xlabel("Perturbation ε", fontsize=12)
    ax.set_ylabel("Test Accuracy",  fontsize=12)
    ax.set_title("FGSM: Perturbation Strength vs Classification Accuracy", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fgsm_eps_curve.png", dpi=130)
    run.log({"fgsm/eps_vs_accuracy": wandb.Image("fgsm_eps_curve.png")})
    plt.close()

    # ── (i-D) Visual comparison at ε=0.03 ────────────────────────────────────
    eps_vis = 0.03
    print(f"\n  Generating visual comparison at ε={eps_vis}…")
    x_norm10, y10 = next(iter(test_ld))
    x_norm10, y10 = x_norm10[:10], y10[:10]
    x_adv_s       = fgsm_scratch(clf, x_norm10, y10, eps_vis)

    x_raw10, _    = next(iter(raw_ld))
    x_raw10       = x_raw10[:10]
    art_fgsm_vis  = ART_FGSM(estimator=art_clf, eps=eps_vis)
    x_adv_art_np  = art_fgsm_vis.generate(x=x_raw10.numpy())

    with torch.no_grad():
        po = clf(x_norm10.to(DEVICE)).argmax(1).cpu().numpy()
        ps = clf(x_adv_s.to(DEVICE)).argmax(1).cpu().numpy()
    pa = art_clf.predict(x_adv_art_np).argmax(1)

    save_comparison_figure(x_norm10, x_adv_s, x_adv_art_np,
                           y10.numpy(), po, ps, pa,
                           eps_vis, "fgsm_visual_comparison.png")
    run.log({"fgsm/visual_comparison": wandb.Image("fgsm_visual_comparison.png")})

    # ── WandB: 10 samples — FGSM Scratch ─────────────────────────────────────
    log_10_samples(run, "samples/FGSM_Scratch",
                   x_norm10, x_adv_s, y10.numpy(), po, ps,
                   orig_is_normalised=True, adv_is_raw=False)

    # ── WandB: 10 samples — FGSM ART ─────────────────────────────────────────
    log_10_samples(run, "samples/FGSM_ART",
                   x_norm10, torch.tensor(x_adv_art_np), y10.numpy(), po, pa,
                   orig_is_normalised=True, adv_is_raw=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TASK (ii) — Adversarial Detectors (ResNet-34)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("TASK ii: Adversarial Detectors (ResNet-34)")
    print("="*60)

    attack_cfgs = {
        "PGD": ART_PGD(estimator=art_clf, eps=PGD_EPS, eps_step=0.007,
                       max_iter=40, targeted=False, verbose=False),
        "BIM": ART_BIM(estimator=art_clf, eps=BIM_EPS, eps_step=0.007,
                       max_iter=10, verbose=False),
    }

    det_results = {}
    for tag, attack in attack_cfgs.items():
        print(f"\n── Attack: {tag} ──────────────────────")
        full_ds   = build_detector_dataset(art_clf, attack, raw_ld, n_batches=40)
        n_val     = len(full_ds) // 5
        n_tr      = len(full_ds) - n_val
        tr_ds, va_ds = random_split(full_ds, [n_tr, n_val],
                                    generator=torch.Generator().manual_seed(SEED))
        det_model  = make_resnet34()
        det_model, best_val = train_detector(det_model, tr_ds, va_ds, run, tag)
        det_results[tag] = best_val

        # WandB: 10 samples for this attack
        x_raw10b, y10b = next(iter(raw_ld))
        x_raw10b = x_raw10b[:10]; y10b = y10b[:10]
        x_adv10  = torch.tensor(attack.generate(x=x_raw10b.numpy()))
        pc       = art_clf.predict(x_raw10b.numpy()).argmax(1)
        pa_det   = art_clf.predict(x_adv10.numpy()).argmax(1)
        log_10_samples(run, f"samples/{tag}",
                       x_raw10b, x_adv10, y10b.numpy(), pc, pa_det,
                       orig_is_normalised=False, adv_is_raw=True)

    # ═════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(f"\n  ResNet-18 clean accuracy : {clean_acc:.4f}")
    print(f"\n  FGSM Attack Results:")
    print(f"  {'ε':>6}  {'Scratch Acc':>12}  {'Drop':>8}  "
          f"{'ART Acc':>10}  {'Drop':>8}")
    print("  " + "-"*58)
    for eps, sa, aa in zip(eps_list, scratch_accs, art_accs):
        print(f"  {eps:6.2f}  {sa:12.4f}  {clean_acc-sa:8.4f}  "
              f"{aa:10.4f}  {clean_acc-aa:8.4f}")

    print(f"\n  Adversarial Detector Results (ResNet-34):")
    print(f"  {'Attack':>6}  {'Val Accuracy':>14}  {'Status':>12}")
    print("  " + "-"*38)
    for tag, acc in det_results.items():
        status = "✓ PASS" if acc >= 0.70 else "✗ FAIL"
        print(f"  {tag:>6}  {acc:14.4f}  {status:>12}")

    # Summary bar chart
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: ε curve
    axes[0].axhline(clean_acc, color="black", lw=2, ls="--",
                    label=f"Clean ({clean_acc:.3f})")
    axes[0].plot(eps_list, scratch_accs, "o-r", lw=2, ms=7, label="FGSM Scratch")
    axes[0].plot(eps_list, art_accs,     "s-b", lw=2, ms=7, label="FGSM ART")
    axes[0].fill_between(eps_list, scratch_accs, art_accs,
                         alpha=0.12, color="purple", label="Gap")
    axes[0].set_xlabel("ε (perturbation strength)", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title("FGSM: Accuracy vs Perturbation Strength", fontsize=12)
    axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3)

    # Right: Detector bar chart
    tags   = list(det_results.keys())
    accs   = [det_results[t] for t in tags]
    colors = ["#4a90d9" if a >= 0.70 else "#e74c3c" for a in accs]
    bars   = axes[1].bar(tags, accs, color=colors, width=0.4,
                         edgecolor="black", linewidth=1.2)
    axes[1].axhline(0.70, color="red", lw=2, ls="--", label="70% threshold")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Attack", fontsize=12)
    axes[1].set_ylabel("Detection Accuracy", fontsize=12)
    axes[1].set_title("Adversarial Detector Performance (ResNet-34)", fontsize=12)
    for bar, acc in zip(bars, accs):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     acc + 0.015, f"{acc:.3f}",
                     ha="center", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("final_summary.png", dpi=130)
    run.log({"summary/final_chart": wandb.Image("final_summary.png")})
    plt.close()
    print("\n  Saved: final_summary.png")
    print(f"\n  WandB run URL: {run.url}")

    run.finish()
    print("\n✓ All tasks complete.")


if __name__ == "__main__":
    main()