"""
Uncertainty Quantification and Calibration Module.
Computes:
  1. Epistemic uncertainty via Monte Carlo Dropout (variance of predictive probabilities).
  2. Aleatoric uncertainty (Dirichlet variance / heteroscedastic entropy).
  3. Predictive entropy: H(p) = - sum p_c log p_c.
  4. Expected Calibration Error (ECE) and reliability diagrams.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MCDropoutUncertaintyHead(nn.Module):
    """
    Multi-task classification and uncertainty estimation head with active dropout.
    """
    def __init__(self, embed_dim=64, num_classes=2, dropout_p=0.25):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.cls_head(x)

    def predict_with_uncertainty(self, x, num_samples=20):
        """
        Runs T stochastic forward passes with dropout active to estimate predictive distribution.
        Returns:
          - mean_probs: [B, num_classes]
          - epistemic_unc: [B] (variance across MC passes)
          - entropy: [B] (predictive entropy)
        """
        self.train()  # Enable dropout
        mc_probs = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                mc_probs.append(probs.unsqueeze(0))

        # mc_probs: [T, B, num_classes]
        mc_probs = torch.cat(mc_probs, dim=0)
        mean_probs = torch.mean(mc_probs, dim=0)  # [B, num_classes]
        
        # Epistemic uncertainty = mean variance across classes
        epistemic_unc = torch.mean(torch.var(mc_probs, dim=0), dim=-1)  # [B]
        
        # Predictive entropy
        eps = 1e-8
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=-1)  # [B]
        
        return mean_probs, epistemic_unc, entropy

def compute_calibration_metrics(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10):
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    probs: [N, C] or [N] (probabilities of positive class)
    labels: [N] binary ground truth (0 or 1)
    """
    if probs.ndim == 2:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
    else:
        confidences = probs
        predictions = (probs >= 0.5).astype(int)
        accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    total_samples = len(labels)
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        count_in_bin = int(np.sum(in_bin))
        bin_counts.append(count_in_bin)

        if count_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            abs_diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * abs_diff
            mce = max(mce, abs_diff)
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)

    # Brier score: mean squared error of predicted probabilities
    if probs.ndim == 2:
        one_hot = np.eye(probs.shape[1])[labels]
        brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    else:
        brier = np.mean((probs - labels) ** 2)

    return {
        'ece': float(ece),
        'mce': float(mce),
        'brier': float(brier),
        'bin_accs': bin_accs,
        'bin_confs': bin_confs,
        'bin_counts': bin_counts,
        'bin_boundaries': bin_boundaries.tolist()
    }
