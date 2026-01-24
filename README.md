# MLOps-Jahanvi_Gajera-M25CSA012
# DLOPs Assignment–1

**Course:** Deep Learning Operations (DLOPs)  
**Assignment:** 1  
**Student Name:** Firstname Surname  
**Roll Number:** B22CSXXX  

---

## 📌 Repository Details

- **Repository Name:** `MLOps-Name-rollNumber`
- **Branch:** `Assignment 1`
- **Framework:** PyTorch
- **Language:** Python
- **Execution Platform:** Google Colab

---

## 📂 Repository Structure


---

## 📊 Assignment Overview

This assignment involves training and evaluating deep learning models on **MNIST** and **FashionMNIST** datasets using different hyperparameters and computational settings. The performance is analyzed based on:

- Classification Accuracy
- Training Time
- FLOPs
- CPU vs GPU comparison

---

## 🧠 Models Used

- **ResNet-18** (pretrained = False)
- **ResNet-50** (pretrained = False)
- **SVM Classifier** (RBF and Polynomial kernels)

---

## 📁 Datasets

- MNIST
- FashionMNIST

**Data Split:**
- 70% Training
- 10% Validation
- 20% Testing

---

## ⚙️ Training Configuration

- **Batch Sizes:** 16, 32  
- **Optimizers:** SGD, Adam  
- **Learning Rates:** 0.001, 0.0001  
- **Epochs:** Multiple values (varied)  
- **AMP:** Enabled  
- **pin_memory:** True / False  

All valid experiments achieved **> 80% accuracy**.

---

## 📈 Q1(a): Deep Learning Model Results

### Test Classification Accuracy (%) for MNIST

| Batch Size | Optimizer | Learning Rate | ResNet-18 | ResNet-50 |
|-----------|-----------|---------------|-----------|-----------|
| 16 | SGD  | 0.001  | 99.22{\%} | 99.14{\%} |
| 16 | SGD  | 0.0001 | 97.46{\%} | 98.90{\%} |
| 16 | Adam | 0.001  | 98.79{\%} | 98.84{\%} |
| 16 | Adam | 0.0001 | 98.24{\%} | 98.89{\%} |
| 32 | SGD  | 0.001  | 98.91{\%} | 98.84{\%} |
| 32 | Adam | 0.0001 | 96.91{\%} | 98.89{\%} |

### Test Classification Accuracy (%) for FashionMnist

| Batch Size | Optimizer | Learning Rate | ResNet-18 | ResNet-50 |
|-----------|-----------|---------------|-----------|-----------|
| 16 | SGD  | 0.001  | 91.18% | 89.11% |
| 16 | SGD  | 0.0001 | 91.14% | 91.10% |
| 16 | Adam | 0.001  | 93.24% | 88.84% |
| 16 | Adam | 0.0001 | 92.19% | 91.89% |
| 32 | SGD  | 0.001  | 92.17% | 89.14% |
| 32 | Adam | 0.0001 | 91.13% | 92.07% |
---

## 🧪 Q1(b): SVM Results

| Kernel | Dataset | Test Accuracy (%) | Training Time (ms) |
|------|---------|------------------|--------------------|
| RBF  | MNIST        | 96.20% | 6031.92 |
| Poly | MNIST        | 95.45% | 6129.23 |
| RBF  | FashionMNIST | 86.50% | 5623.30 |
| Poly | FashionMNIST | 82.95% | 6101.91 |

---

## 💻 Q2: CPU vs GPU Performance (FashionMNIST)

### Accuracy Comparison

| Compute | Batch Size | Optimizer | LR | ResNet-18(Acc%) | ResNet-50(Acc%) | ResNet-18(Time(ms)) | ResNet-50(Time(ms)) | ResNet-18(FLOPs) | ResNet-50(FLOPs) | 
|-------|-----------|-----------|----|-----------|-----------|-----------|-----------|-----------|-----------|
| CPU | 16 | SGD  | 0.001 | 85.02% | 82.54% | 327273.0 | 887577.1  | 1.83e+09 | 4.14e+09 |
| CPU | 16 | Adam | 0.001 | 85.58% | 82.27% | 532474.4 | 1204471.9 | 1.83e+09 | 4.14e+09 |
| GPU | 16 | SGD  | 0.001 | 87.36% | 82.88% | 56131.5  | 104930.4  | 1.83e+09 | 4.14e+09 |
| GPU | 16 | Adam | 0.001 | 86.99% | 80.90% | 51225.7  | 110957.5  | 1.83e+09 | 4.14e+09 |

---

## 📊 Graphs

### Accuracy Comparison for MNIST
![Accuracy Comparison](accuracy_comparison.png)

### Training Time Comparison
![Training Time Comparison](training_time_comparison.png)

---

## 📄 Report

- **File Name:** `RollNumber_Name_Ass1.pdf`
- Contains detailed analysis, tables, and plots.

---

## 🔗 Google Colab Link (Already Executed)

👉 **Colab Notebook:**  

