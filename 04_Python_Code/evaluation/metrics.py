"""
Comprehensive Medical Diagnostic Evaluation Metrics:
  - Detection: Sensitivity, Specificity, F1, Average Precision, FROC curve.
  - Classification: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
  - Uncertainty: Expected Calibration Error (ECE), Brier Score, NLL, Risk-Coverage.
  - Domain Generalization: In-domain vs Out-of-domain performance gap.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve)
from uncertainty.uncertainty_head import compute_calibration_metrics

def compute_classification_metrics(y_true, y_prob):
    """
    Computes all clinical classification performance metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.5

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / max(1, tn + fp)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)

    calib = compute_calibration_metrics(y_prob, y_true)

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'sensitivity': float(rec),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'roc_auc': float(auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': cm.tolist(),
        'ece': calib['ece'],
        'brier': calib['brier'],
        'calib_details': calib,
        'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
        'pr_curve': {'precision': precision_vals.tolist(), 'recall': recall_vals.tolist()}
    }

def compute_froc_curve(nodule_probs, nodule_labels, scan_ids, fp_thresholds=None):
    """
    Computes Free-Response Receiver Operating Characteristic (FROC) curve:
    Sensitivity as a function of False Positives per Scan (FPS).
    Predefined thresholds: [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    """
    if fp_thresholds is None:
        fp_thresholds = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    nodule_probs = np.asarray(nodule_probs)
    nodule_labels = np.asarray(nodule_labels)
    total_nodules = max(1, int(np.sum(nodule_labels == 1)))
    num_scans = max(1, len(set(scan_ids)))

    thresholds = np.linspace(0.0, 1.0, 100)
    fps_list = []
    sens_list = []

    for t in thresholds:
        preds = (nodule_probs >= t).astype(int)
        tp = np.sum((preds == 1) & (nodule_labels == 1))
        fp = np.sum((preds == 1) & (nodule_labels == 0))
        
        sens = tp / total_nodules
        fps = fp / num_scans
        fps_list.append(fps)
        sens_list.append(sens)

    # Interpolate sensitivities at standard fp thresholds
    sens_at_fps = {}
    fps_arr = np.array(fps_list)
    sens_arr = np.array(sens_list)
    # Sort by fps ascending
    sort_idx = np.argsort(fps_arr)
    fps_sorted = fps_arr[sort_idx]
    sens_sorted = sens_arr[sort_idx]

    for target_fp in fp_thresholds:
        val = np.interp(target_fp, fps_sorted, sens_sorted)
        sens_at_fps[f"sens@{target_fp}"] = float(val)

    mean_froc = np.mean(list(sens_at_fps.values()))
    return {
        'fps': fps_list,
        'sensitivities': sens_list,
        'sens_at_fps': sens_at_fps,
        'mean_froc_score': float(mean_froc)
    }

def compute_risk_coverage(y_true, y_prob, uncertainties):
    """
    Computes selective prediction risk-coverage curve.
    Evaluates classification error rate (risk) as a function of retained samples (coverage),
    deferring predictions with the highest uncertainty first.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    uncertainties = np.asarray(uncertainties)
    
    # Sort samples by ascending uncertainty (most confident first)
    sort_idx = np.argsort(uncertainties)
    y_true_sorted = y_true[sort_idx]
    y_pred_sorted = (y_prob[sort_idx] >= 0.5).astype(int)

    coverage_points = np.linspace(0.1, 1.0, 20)
    risk_points = []

    for cov in coverage_points:
        n_keep = max(1, int(round(cov * len(y_true))))
        err_rate = np.mean(y_pred_sorted[:n_keep] != y_true_sorted[:n_keep])
        risk_points.append(float(err_rate))

    return {
        'coverage': coverage_points.tolist(),
        'risk': risk_points
    }
