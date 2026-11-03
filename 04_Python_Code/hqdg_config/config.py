"""
Configuration parameters for UQ-HQDGNet:
Uncertainty-Guided Hybrid Quantum Domain-Generalization Network
"""
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class PreprocessingConfig:
    hu_min: float = -1000.0
    hu_max: float = 400.0
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    patch_size: Tuple[int, int, int] = (32, 32, 32)
    crop_size: Tuple[int, int, int] = (32, 32, 32)
    num_samples_per_scan: int = 16

@dataclass
class QuantumConfig:
    n_qubits: int = 4
    n_layers: int = 2
    ansatz_type: str = "StronglyEntangling"
    backend: str = "default.qubit"
    shots: int = 0
    diff_method: str = "parameter-shift"
    encoding: str = "angle"

@dataclass
class ModelConfig:
    in_channels: int = 1
    base_filters: int = 32
    classical_embed_dim: int = 64
    quantum_embed_dim: int = 4
    num_classes: int = 2
    bbox_dim: int = 4
    dropout_rate: float = 0.2
    use_transformer_attention: bool = True
    attention_heads: int = 4

@dataclass
class UncertaintyConfig:
    mc_samples: int = 20
    method: str = "mc_dropout"
    temperature: float = 1.0
    calibration_bins: int = 10
    uncertainty_threshold: float = 0.35

@dataclass
class DomainGenConfig:
    use_mmd: bool = True
    mmd_gamma_list: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0])
    use_dann: bool = True
    dann_weight: float = 0.1
    grl_max_iter: int = 1000

@dataclass
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 2
    seed: int = 42
    device: str = "cuda"
    loss_weights: dict = field(default_factory=lambda: {
        'det_cls': 1.0,
        'det_reg': 1.0,
        'malignancy_cls': 1.0,
        'mmd': 0.2,
        'uncertainty': 0.1
    })

@dataclass
class PipelineConfig:
    data_dir: str = "UQ_HQDGNet_Research/03_Data"
    results_dir: str = "UQ_HQDGNet_Research/06_Experimental_Results"
    checkpoints_dir: str = "UQ_HQDGNet_Research/04_Python_Code/checkpoints"
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    domain_gen: DomainGenConfig = field(default_factory=DomainGenConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

