# This repository is for the under review manuscript "Heavy-tailed update arises from information-driven self-organization in non-equilibrium learning"


This repository contains training scripts, weight update tracking, perturbation analysis, and information-theoretic validation tools for studying non-equilibrium learning behavior in neural networks.

## 📁 Directory Structure

- `train/`: Scripts to train various models (MLP, CNN, Transformer) on MNIST, CIFAR and 26-character prediction datasets.
- `updates/`: Scripts and `.npy` data files tracking weight updates across training steps.
- `interval/`: Interval-based data files capturing weight update characteristics.
- `perturbation/`: Perturbation response analysis via loss surface variance.
- `validation/`: Mutual information and gradient validation analysis.

## 🚀 Quick Start

### 1. Environment Setup

Create and activate your Python environment:

    conda create -n netdynamics python=3.8
    conda activate netdynamics

Install required packages:

    pip install numpy matplotlib pandas scipy scikit-learn torch torchvision wandb

### 2. Train Models

Run the following scripts to train models:

- Train MLP on MNIST:

      python train/MNIST-MLP.py

- Train CNN on CIFAR-10:

      python train/CiFar-CNN.py

- Train Transformer on character dataset:

      python train/charac-transformer.py

### 3. Track Weight Updates

Use the following scripts to record weight updates:

- For CNN:

      python train/update_tracker_cnn.py

- For 2-layer MLP:

      python train/update_tracker_2linear.py

### 4. Analyze Weight Update Distribution

Open and run the Jupyter notebook:

    jupyter notebook updates/plot_update.ipynb

### 5. Perturbation Analysis

Visualize loss response to perturbation in:

    jupyter notebook perturbation/plot_perturbation.ipynb

### 6. Information-Theoretic Validation

Compute MI and gradients with:

    python validation/mi_gradient.py

## 📊 Data Files

The `.npy` files store histograms:

- `*_nums.npy`: values (y-axis)
- `*_xbins.npy`: bin edges (x-axis)

Filenames encode model, optimizer, learning rate, and training window.

## 🧠 Code Purpose

This codebase enables analysis of:

- Learning dynamics through parameter updates
- Heavy-tailed update distributions
- Response to parameter perturbation
- Information-theoretic validation

---
