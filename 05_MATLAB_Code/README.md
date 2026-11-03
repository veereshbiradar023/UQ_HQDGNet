# MATLAB Implementation for UQ-HQDGNet

This folder contains MATLAB supporting scripts for 3D thoracic CT volume preprocessing, multi-planar reconstruction (MPR), rigorous statistical hypothesis testing, FROC detection curve computation, and cross-domain generalization analysis.

## Structure
- `preprocessing/preprocess_ct_volume.m`: Hounsfield Unit clamping and morphological lung parenchyma segmentation.
- `visualization/visualize_nodule_3d.m`: Axial, coronal, sagittal multi-planar visualization and 3D isosurface rendering.
- `statistics/statistical_tests.m`: Paired Wilcoxon signed-rank test, DeLong AUC test, Cohen's d effect size, and 95% bootstrap confidence intervals.
- `performance_analysis/compute_froc_curve.m`: Free-Response ROC curve generation at predefined false positive thresholds (0.125 to 8.0 FPS).
- `cross_domain_analysis/cross_domain_evaluation.m`: Inter-domain performance degradation analysis across LIDC-IDRI, LNDb, and NLST.

## Interoperability with Python Pipeline
The scripts read JSON outputs exported by the Python training and evaluation pipeline from `06_Experimental_Results/`.
