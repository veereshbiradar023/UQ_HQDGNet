"""
Clinical Inference and Explainability Demonstration Script.
"""
import os
import sys
import torch
import numpy as np

curr_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(curr_dir)
workspace_root = os.path.dirname(repo_root)

for p in [curr_dir, repo_root, workspace_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from hqdg_config.config import PipelineConfig
from models.uq_hqdgnet import UQHQDGNet
from explainability.gradcam3d import GradCAM3D

def main():
    config = PipelineConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== UQ-HQDGNet Clinical Inference Engine ===")

    if not os.path.isabs(config.checkpoints_dir):
        config.checkpoints_dir = os.path.join(workspace_root, config.checkpoints_dir)

    model = UQHQDGNet(
        in_channels=config.model.in_channels,
        base_filters=config.model.base_filters,
        classical_embed_dim=config.model.classical_embed_dim,
        n_qubits=config.quantum.n_qubits,
        n_quantum_layers=config.quantum.n_layers,
        num_domains=3,
        dropout_p=config.model.dropout_rate
    ).to(device)

    ckpt_path = os.path.join(config.checkpoints_dir, "uq_hqdgnet_best.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded model checkpoint from {ckpt_path}")
    else:
        print("Using initialized model.")

    D, H, W = 32, 32, 32
    sample_patch = np.random.normal(0.2, 0.03, (1, 1, D, H, W)).astype(np.float32)
    cz, cy, cx = D // 2, H // 2, W // 2
    grid_z, grid_y, grid_x = np.ogrid[:D, :H, :W]
    mask = (grid_z - cz)**2 + (grid_y - cy)**2 + (grid_x - cx)**2 <= 5.0**2
    sample_patch[0, 0, mask] = 0.82

    tensor_in = torch.from_numpy(sample_patch).to(device)

    num_mc = 15
    mc_mal_probs = []
    for _ in range(num_mc):
        model.train()
        with torch.no_grad():
            out = model(tensor_in, domain_alpha=0.0)
            p = torch.softmax(out['malignancy_logit'], dim=-1)[0, 1].item()
            mc_mal_probs.append(p)

    model.eval()
    with torch.no_grad():
        out = model(tensor_in, domain_alpha=0.0)
        det_score = torch.sigmoid(out['det_logit'])[0].item()
        bbox_offsets = out['bbox_pred'][0].cpu().numpy().tolist()

    mean_mal_prob = float(np.mean(mc_mal_probs))
    epistemic_unc = float(np.var(mc_mal_probs))
    entropy = - (mean_mal_prob * np.log(mean_mal_prob + 1e-8) + (1 - mean_mal_prob) * np.log(1 - mean_mal_prob + 1e-8))

    print("\n--- Diagnostic Findings ---")
    print(f"Nodule Candidate Presence Score: {det_score:.4f} (Threshold: 0.50)")
    print(f"Predicted Bounding Box Offsets:  dz={bbox_offsets[0]:.2f}, dy={bbox_offsets[1]:.2f}, dx={bbox_offsets[2]:.2f}, dr={bbox_offsets[3]:.2f} mm")
    print(f"Malignancy Risk Probability:     {mean_mal_prob:.4f}")
    print(f"Epistemic Uncertainty:           {epistemic_unc:.4f}")
    print(f"Predictive Entropy:              {entropy:.4f}")

    print("\n--- Clinical Decision Recommendation ---")
    if epistemic_unc > config.uncertainty.uncertainty_threshold:
        print("[STATUS: HIGH UNCERTAINTY / BORDERLINE]")
        print("Recommendation: Defer automated staging. Refer to expert thoracic radiologist multidisciplinary panel.")
    elif mean_mal_prob >= 0.65:
        print("[STATUS: SUSPICIOUS MALIGNANCY (Lung-RADS 4B/4X Equivalent)]")
        print("Recommendation: Recommend contrast-enhanced CT / PET-CT and biopsy consultation.")
    elif mean_mal_prob <= 0.35:
        print("[STATUS: LIKELY BENIGN (Lung-RADS 2 Equivalent)]")
        print("Recommendation: Standard 12-month low-dose CT follow-up.")
    else:
        print("[STATUS: INDETERMINATE (Lung-RADS 3 Equivalent)]")
        print("Recommendation: Short-interval 6-month follow-up CT.")

    # 3D Grad-CAM Visual Attribution
    gradcam = GradCAM3D(model)
    cam = gradcam.generate_heatmap(tensor_in, target_class=1)
    print(f"\nGenerated 3D Grad-CAM attribution map. Shape: {cam.shape}, Mean intensity: {cam.mean():.4f}")

if __name__ == "__main__":
    main()
