"""
End-to-End Training and Benchmarking Script for UQ-HQDGNet and Baseline Models.
"""
import os
import sys
import json

curr_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(curr_dir)
workspace_root = os.path.dirname(repo_root)

for p in [curr_dir, repo_root, workspace_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from torch.utils.data import DataLoader

from hqdg_config.config import PipelineConfig
from datasets.nodule_dataset import PulmonaryNoduleDataset, generate_synthetic_benchmark_dataset
from models.uq_hqdgnet import UQHQDGNet
from training.trainer import UQHQDGTrainer

def main():
    config = PipelineConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== UQ-HQDGNet Research Benchmark Pipeline ===")
    print(f"Execution Device: {device}")
    
    # Ensure correct relative data directory
    data_dir = config.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(workspace_root, data_dir)
        config.data_dir = data_dir

    if not os.path.isabs(config.results_dir):
        config.results_dir = os.path.join(workspace_root, config.results_dir)
    if not os.path.isabs(config.checkpoints_dir):
        config.checkpoints_dir = os.path.join(workspace_root, config.checkpoints_dir)

    # 1. Dataset Generation / Loading
    splits_file = os.path.join(data_dir, "Splits", "patient_level_splits.json")
    if not os.path.exists(splits_file):
        print("Synthesizing multi-domain calibrated benchmark dataset...")
        splits_file = generate_synthetic_benchmark_dataset(data_dir, num_patients_per_domain=15, seed=config.training.seed)

    with open(splits_file, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    train_ds = PulmonaryNoduleDataset(splits['train'], is_training=True, augment=True)
    val_ds = PulmonaryNoduleDataset(splits['val'], is_training=False, augment=False)
    test_ds = PulmonaryNoduleDataset(splits['test'], is_training=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # 2. Model Initialization
    model = UQHQDGNet(
        in_channels=config.model.in_channels,
        base_filters=config.model.base_filters,
        classical_embed_dim=config.model.classical_embed_dim,
        n_qubits=config.quantum.n_qubits,
        n_quantum_layers=config.quantum.n_layers,
        num_domains=3,
        dropout_p=config.model.dropout_rate
    )

    # Fast 5-epoch training for benchmark verification
    config.training.epochs = 5
    ckpt_path = os.path.join(config.checkpoints_dir, "uq_hqdgnet_best.pt")
    trainer = UQHQDGTrainer(model, train_loader, val_loader, config, device=device)
    history = trainer.fit(save_path=ckpt_path)

    # Save training history
    history_path = os.path.join(config.results_dir, "training_history.json")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    print(f"Training completed successfully. Checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    main()
