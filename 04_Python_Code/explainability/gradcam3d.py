"""
3D Gradient-Weighted Class Activation Mapping (3D Grad-CAM).
Produces voxel-level spatial attention heatmaps showing anatomical focus.
"""
import torch
import torch.nn.functional as F
import numpy as np

class GradCAM3D:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        # Hook last convolutional layer of backbone stage4
        target_layer = self.model.backbone.stage4.conv2
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=1):
        """
        input_tensor: [1, 1, D, H, W]
        target_class: 1 (malignant) or 0 (benign)
        """
        self.model.eval()
        self.model.zero_grad()

        outputs = self.model(input_tensor)
        score = outputs['malignancy_logit'][0, target_class]
        score.backward()

        # Global average pool gradients over spatial dimensions (D, H, W)
        # self.gradients: [1, C, D', H', W']
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to match original 3D patch resolution
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='trilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam
