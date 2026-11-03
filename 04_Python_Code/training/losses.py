"""
Multi-Task Loss Formulations for UQ-HQDGNet:
  1. Focal Loss for nodule candidate detection (presence vs background imbalance).
  2. Smooth L1 Loss for 3D bounding box regression offsets [dx, dy, dz, dr].
  3. Binary Cross-Entropy Loss for nodule malignancy risk classification.
  4. Multi-Kernel MMD Loss for cross-domain latent feature distribution alignment.
  5. Cross-Entropy Loss for domain adversarial discriminator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from domain_generalization.domain_aligner import MultiKernelMMD

class FocalLoss3D(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits: [B], targets: [B] (0 or 1)
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        w = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = -w * ((1 - pt) ** self.gamma) * torch.log(torch.clamp(pt, min=1e-8))
        return loss.mean()

class JointUQHQDGLoss(nn.Module):
    def __init__(self, w_det=1.0, w_bbox=1.0, w_mal=1.0, w_mmd=0.25, w_dom=0.1):
        super().__init__()
        self.w_det = w_det
        self.w_bbox = w_bbox
        self.w_mal = w_mal
        self.w_mmd = w_mmd
        self.w_dom = w_dom

        self.focal_loss = FocalLoss3D(alpha=0.75, gamma=2.0)
        self.bbox_loss = nn.SmoothL1Loss()
        self.mal_loss = nn.CrossEntropyLoss()
        self.mmd_loss = MultiKernelMMD()
        self.domain_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, batch):
        # 1. Detection presence loss
        l_det = self.focal_loss(outputs['det_logit'], batch['det_label'])

        # 2. Bbox regression loss (only calculated for positive nodules)
        pos_mask = (batch['det_label'] == 1)
        if pos_mask.sum() > 0:
            l_bbox = self.bbox_loss(outputs['bbox_pred'][pos_mask], batch['bbox'][pos_mask])
        else:
            l_bbox = torch.tensor(0.0, device=outputs['det_logit'].device)

        # 3. Malignancy classification loss
        l_mal = self.mal_loss(outputs['malignancy_logit'], batch['malignancy_label'])

        # 4. Domain discriminator loss
        l_dom = self.domain_loss(outputs['domain_logit'], batch['domain_id'])

        # 5. Multi-kernel MMD across domain subgroups
        domain_ids = batch['domain_id']
        dom0_mask = (domain_ids == 0)
        dom1_mask = (domain_ids == 1)
        if dom0_mask.sum() > 1 and dom1_mask.sum() > 1:
            l_mmd = self.mmd_loss(outputs['fused_embed'][dom0_mask], outputs['fused_embed'][dom1_mask])
        else:
            l_mmd = torch.tensor(0.0, device=outputs['det_logit'].device)

        total_loss = (self.w_det * l_det + 
                      self.w_bbox * l_bbox + 
                      self.w_mal * l_mal + 
                      self.w_dom * l_dom + 
                      self.w_mmd * l_mmd)

        return {
            'total_loss': total_loss,
            'l_det': l_det,
            'l_bbox': l_bbox,
            'l_mal': l_mal,
            'l_dom': l_dom,
            'l_mmd': l_mmd
        }
