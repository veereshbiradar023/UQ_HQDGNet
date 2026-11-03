"""
UQ-HQDGNet: Uncertainty-Guided Hybrid Quantum Domain-Generalization Network.
Unites:
  - 3D Classical Hybrid Backbone
  - Variational Quantum Circuit (VQC) Module
  - Multi-Task Detection and Malignancy Classification Heads
  - Gradient Reversal Domain Alignment
  - Uncertainty Quantification Head
"""
import torch
import torch.nn as nn
from .backbone3d import ClassicalHybridBackbone3D
from quantum.vqc_module import VariationalQuantumCircuit
from domain_generalization.domain_aligner import GradientReversalLayer, MultiKernelMMD
from uncertainty.uncertainty_head import MCDropoutUncertaintyHead

class UQHQDGNet(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, classical_embed_dim=64,
                 n_qubits=4, n_quantum_layers=2, num_domains=3, dropout_p=0.25):
        super().__init__()
        
        # 1. 3D Classical Backbone
        self.backbone = ClassicalHybridBackbone3D(
            in_channels=in_channels,
            base_filters=base_filters,
            classical_embed_dim=classical_embed_dim
        )

        # 2. Variational Quantum Circuit
        self.quantum_module = VariationalQuantumCircuit(
            in_features=classical_embed_dim,
            n_qubits=n_qubits,
            n_layers=n_quantum_layers
        )

        # 3. Detection Head (Candidate Localization: presence + bbox offsets [dx, dy, dz, dr])
        self.det_head = nn.Sequential(
            nn.Linear(classical_embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)  # Presence logit
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(classical_embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 4)  # 3D bounding box regression
        )

        # 4. Malignancy Classification & Uncertainty Head
        self.uncertainty_head = MCDropoutUncertaintyHead(
            embed_dim=classical_embed_dim,
            num_classes=2,
            dropout_p=dropout_p
        )

        # 5. Domain Discriminator (GRL)
        self.grl = GradientReversalLayer(alpha=1.0)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(classical_embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, num_domains)
        )

    def forward(self, x, domain_alpha=1.0):
        # 1. Classical feature extraction
        feat_map, class_embed = self.backbone(x)

        # 2. Quantum feature learning & fusion
        fused_embed, q_expvals = self.quantum_module(class_embed)

        # 3. Nodule Detection outputs
        det_logit = self.det_head(fused_embed).squeeze(-1)
        bbox_pred = self.bbox_head(fused_embed)

        # 4. Malignancy classification outputs
        malignancy_logit = self.uncertainty_head(fused_embed)

        # 5. Domain prediction
        self.grl.alpha = domain_alpha
        domain_embed = self.grl(fused_embed)
        domain_logit = self.domain_discriminator(domain_embed)

        return {
            'det_logit': det_logit,
            'bbox_pred': bbox_pred,
            'malignancy_logit': malignancy_logit,
            'domain_logit': domain_logit,
            'fused_embed': fused_embed,
            'feat_map': feat_map,
            'q_expvals': q_expvals
        }
