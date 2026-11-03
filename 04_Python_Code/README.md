# UQ-HQDGNet: Uncertainty-Guided Hybrid Quantum Domain-Generalization Network

This repository contains the complete reproducible Python and MATLAB implementation of **UQ-HQDGNet** for domain-generalizable pulmonary nodule detection and lung cancer classification.

## Directory Structure
- `config/`: Pipeline and model hyperparameters.
- `datasets/`: Dataset loader and synthetic multi-domain benchmark generator (LIDC-IDRI, LNDb, NLST).
- `preprocessing/`: HU normalization, 3D cubic resampling, lung segmentation, and patch extraction.
- `models/`: 3D Classical Hybrid Backbone, PennyLane VQC module, Multi-Task Prediction Heads.
- `quantum/`: Parameterized quantum circuits, angle encoding, circular CNOT entanglement, and Pauli-Z measurements.
- `uncertainty/`: Monte Carlo Dropout epistemic/aleatoric uncertainty quantification and ECE calibration.
- `domain_generalization/`: Gradient Reversal Layer (DANN) and Multi-Kernel Maximum Mean Discrepancy (MK-MMD).
- `training/`: Joint multi-task loss functions and training engine.
- `evaluation/`: Clinical detection (FROC, sensitivity at predetermined FPS) and classification (AUC, PR, ECE) metrics.
- `explainability/`: 3D Grad-CAM implementation.
- `visualization/`: Publication-grade plotting scripts.

## Quick Execution
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full training and evaluation benchmark
python train.py

# 3. Evaluate checkpoints and generate figures
python evaluate.py

# 4. Run single-volume inference and explainability
python inference.py
```
