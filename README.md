# This repository is for the under review manuscript "Heavy-tailed update arises from information-driven self-organization in non-equilibrium learning"


# 📂 Folder Overview: `scripts/`

This document describes the role and contents of each folder within the `scripts/` directory. Use it as a reference to understand where each analysis module, dataset, and model training script resides.

---

## 📁 interval/

**Purpose**:  
Analyzes how internal neural representations evolve over time during training. It computes and visualizes *interval distances* across different layers or steps.

**Contents**:
- `.npy` files storing binned distance metrics (e.g., `xbins`, `nums`)
- `plot_interval.ipynb`: Notebook to visualize training trajectory shifts

---

## 📁 updates/

**Purpose**:  
Quantifies changes in model weights (update magnitudes) between training steps. This can reveal trends in learning rate effects and training dynamics.

**Contents**:
- `.npy` files capturing update magnitudes
- `plot_update.ipynb`: Generates comparative plots

---

## 📁 perturbation/

**Purpose**:  
Measures model robustness by adding perturbations to the weights and tracking how loss variance responds.

**Contents**:
- `.npy` files recording perturbed loss behavior
- `plot_perturbation.ipynb`: Analysis of perturbation robustness

---

## 📁 train/

**Purpose**:  
Includes training scripts for various models and datasets.

**Contents**:
- `MNIST-MLP.py`: Trains MLP on MNIST
- `CiFar-CNN.py`: Trains CNN on CIFAR-10
- `charac-transformer.py`: Trains a character-level Transformer
- `update_tracker_*.py`: Monitors updates during training

---

## 📁 validation/

**Purpose**:  
Performs validation analysis including mutual information (MI), gradient comparison, and metric submission.

**Contents**:
- `mi_gradient.py`: Computes MI and PCA-based gradient direction.
- `plot_MIbeta.ipynb`: Visualizes MI and PCA-based gradient direction treads.
- `.npy` and `.csv`: Precomputed MI or prediction result arrays

---
