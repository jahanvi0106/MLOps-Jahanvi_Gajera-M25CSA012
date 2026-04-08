# Assignment 5 - LoRA & Adversarial Robustness (ART)

## 👩‍🎓 Student Details

**Name:** Jahanvi Gajera
**Roll No:** M25CSA012

---

## 📌 Overview

This assignment explores:

* **Q1:** Efficient fine-tuning of Vision Transformers using **LoRA (Low-Rank Adaptation)**
* **Q2:** **Adversarial attacks and detection** using IBM Adversarial Robustness Toolbox (ART)

---

# 🔷 Q1: ViT + LoRA on CIFAR-100

## 📊 Objective

* Fine-tune a pre-trained ViT-S model on CIFAR-100
* Compare **full fine-tuning vs LoRA-based fine-tuning**
* Analyze effect of **Rank (r)** and **Alpha (α)**

---

## ⚙️ Experimental Setup

* Model: ViT-S (pretrained on ImageNet)
* Dataset: CIFAR-100
* LoRA applied on attention layers (Q, K, V)
* Hyperparameters:

  * Rank: 2, 4, 8
  * Alpha: 2, 4, 8
  * Dropout: 0.1

---

## 📈 Key Results

### 🔹 Baseline (No LoRA)

* Test Accuracy: **36.31%**
* Trainable Params: **38K**

### 🔹 Best LoRA Configuration

* Rank = 8, Alpha = 8
* Test Accuracy: **84.41% (Best)**
* Trainable Params: **259K**

---

## 🧠 Observations

* LoRA significantly improves performance over full fine-tuning
* Increasing rank & alpha improves representation capacity
* LoRA achieves **>2× performance boost** with fewer trainable parameters

---

# 🔷 Q2: Adversarial Attacks using IBM ART

## ⚙️ Setup

* Dataset: CIFAR-10
* Model: ResNet-18 (trained from scratch)
* Baseline Accuracy: **73.4%**

---

## ⚡ Part (i): FGSM Attack

### 📌 Method

* FGSM implemented:

  * From scratch (PyTorch)
  * Using IBM ART

### 📊 Results

| ε    | Clean | FGSM (Scratch) | FGSM (ART) |
| ---- | ----- | -------------- | ---------- |
| 0.01 | 73.4% | 61.2%          | 60.9%      |
| 0.03 | 73.4% | 38.7%          | 38.2%      |
| 0.05 | 73.4% | 24.3%          | 23.8%      |
| 0.10 | 73.4% | 11.8%          | 11.5%      |

---

### 🔍 Observations

* Small perturbations drastically reduce accuracy
* ART and scratch implementations produce similar results
* Near-linear drop in performance with increasing ε

---

## 🛡️ Part (ii): Adversarial Detection

### 📌 Models

* Detector A: PGD-based adversarial detection
* Detector B: BIM-based adversarial detection
* Architecture: ResNet-34 (Binary classifier)

---

### 📊 Detection Results

| Attack | Accuracy | Precision | Recall | F1   |
| ------ | -------- | --------- | ------ | ---- |
| PGD    | 78.3%    | 0.81      | 0.75   | 0.78 |
| BIM    | 74.6%    | 0.77      | 0.72   | 0.74 |

---

### 🔍 Observations

* Both detectors exceed required **≥70% accuracy**
* PGD attacks are more detectable than BIM
* Detection performance depends on attack type

---

# 📊 WandB Logging

Includes:

* Training & validation curves
* FGSM attack comparisons
* PGD & BIM adversarial samples
* Detector performance graphs
* Confusion matrices


## 🔗 Links

* WandB:
  Question1 : https://wandb.ai/m25csa012-iit-jodhpur/dlops-assignment5
  Question2 : https://wandb.ai/m25csa012-iit-jodhpur/adversarial-ml-cifar10/runs/ipxe6om0?nw=nwuserm25csa012

---

## 📌 Notes

* All experiments implemented using **PyTorch**
* Best model weights included
* Code follows assignment guidelines

---
