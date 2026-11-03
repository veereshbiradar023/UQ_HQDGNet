"""
Domain Generalization and Invariant Feature Learning Module.
Implements:
  1. Gradient Reversal Layer (GRL) for domain adversarial learning.
  2. Multi-Kernel Maximum Mean Discrepancy (MK-MMD) with RBF kernels.
"""
import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class MultiKernelMMD(nn.Module):
    """
    Computes Maximum Mean Discrepancy (MMD) across source domain feature representations
    using a linear combination of multiple Gaussian RBF kernels:
      k(x, y) = sum_m exp(- gamma_m * ||x - y||^2)
    """
    def __init__(self, gammas=None):
        super().__init__()
        if gammas is None:
            self.gammas = [0.001, 0.01, 0.1, 1.0, 10.0]
        else:
            self.gammas = gammas

    def rbf_kernel(self, x, y):
        B_x = x.size(0)
        B_y = y.size(0)
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist_sq = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        dist_sq = torch.clamp(dist_sq, min=0.0)

        kernel_val = 0.0
        for g in self.gammas:
            kernel_val = kernel_val + torch.exp(-g * dist_sq)
        return kernel_val

    def forward(self, source_features, target_features):
        """
        MMD^2 = E[k(xs, xs)] - 2 E[k(xs, xt)] + E[k(xt, xt)]
        """
        if source_features.size(0) == 0 or target_features.size(0) == 0:
            return torch.tensor(0.0, device=source_features.device)

        k_ss = self.rbf_kernel(source_features, source_features)
        k_tt = self.rbf_kernel(target_features, target_features)
        k_st = self.rbf_kernel(source_features, target_features)

        mmd_loss = k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()
        return torch.clamp(mmd_loss, min=0.0)
