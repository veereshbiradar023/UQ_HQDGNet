"""
Publication-Quality Figure Generation:
Generates ROC Curves, Precision-Recall Curves, FROC Detection Curves,
Calibration Reliability Diagrams, Confusion Matrices, and Risk-Coverage Curves.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

def plot_roc_curves(eval_results_dict, save_path):
    plt.figure(figsize=(7, 6))
    for name, res in eval_results_dict.items():
        fpr = res['roc_curve']['fpr']
        tpr = res['roc_curve']['tpr']
        auc = res['roc_auc']
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve plot to {save_path}")

def plot_pr_curves(eval_results_dict, save_path):
    plt.figure(figsize=(7, 6))
    for name, res in eval_results_dict.items():
        prec = res['pr_curve']['precision']
        rec = res['pr_curve']['recall']
        pr_auc = res['pr_auc']
        plt.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.title('Precision-Recall (PR) Curves')
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved PR curve plot to {save_path}")

def plot_froc_curve(froc_dict, save_path):
    plt.figure(figsize=(7, 6))
    for name, data in froc_dict.items():
        fps = data['fps']
        sens = data['sensitivities']
        score = data['mean_froc_score']
        # Sort for smooth line
        idx = np.argsort(fps)
        plt.plot(np.array(fps)[idx], np.array(sens)[idx], lw=2, label=f"{name} (Mean FROC = {score:.3f})")

    plt.xscale('log')
    plt.xlim([0.1, 10.0])
    plt.ylim([0.4, 1.02])
    plt.xlabel('False Positives per Scan (log scale)')
    plt.ylabel('Nodule Detection Sensitivity')
    plt.title('Free-Response ROC (FROC) Curve')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved FROC curve to {save_path}")

def plot_calibration_diagram(calib_details, save_path, model_name="UQ-HQDGNet"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    bin_accs = calib_details['bin_accs']
    bin_confs = calib_details['bin_confs']
    bin_counts = calib_details['bin_counts']
    ece = calib_details['ece']
    brier = calib_details['brier']

    # Reliability line
    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect Calibration')
    ax1.plot(bin_confs, bin_accs, 's-', color='#1f77b4', lw=2, markersize=6,
             label=f"{model_name} (ECE = {ece:.3f}, Brier = {brier:.3f})")
    ax1.set_ylabel('Empirical Accuracy')
    ax1.set_title(f'Calibration Reliability Diagram ({model_name})')
    ax1.legend(loc='upper left', frameon=True)
    ax1.set_ylim([-0.05, 1.05])

    # Confidence distribution histogram
    bin_centers = [(calib_details['bin_boundaries'][i] + calib_details['bin_boundaries'][i+1])/2 for i in range(len(bin_counts))]
    ax2.bar(bin_centers, bin_counts, width=0.08, color='#aec7e8', edgecolor='#1f77b4', alpha=0.8)
    ax2.set_xlabel('Predicted Confidence')
    ax2.set_ylabel('Count')
    ax2.set_xlim([0.0, 1.0])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved calibration diagram to {save_path}")

def plot_confusion_matrix(cm, save_path, class_names=['Benign', 'Malignant']):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True Label', xlabel='Predicted Label',
           title='Confusion Matrix')

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

def plot_risk_coverage(risk_coverage_dict, save_path):
    plt.figure(figsize=(7, 6))
    for name, data in risk_coverage_dict.items():
        cov = data['coverage']
        risk = data['risk']
        plt.plot(cov, risk, 'o-', lw=2, label=f"{name}")

    plt.xlabel('Coverage (Proportion of Evaluated Predictions)')
    plt.ylabel('Selective Risk (Classification Error Rate)')
    plt.title('Selective Prediction Risk-Coverage Trade-Off')
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved risk-coverage plot to {save_path}")
