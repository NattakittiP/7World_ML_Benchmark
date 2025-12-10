# A Multi-World Synthetic Benchmark for Evaluating RSCE in Clinical Machine Learning

This repository contains the full code and results for the paper:

> **A Multi-World Synthetic Benchmark for Evaluating RSCE in Clinical Machine Learning**  
> *(IEEE Access submission)*

The project introduces a synthetic, multi-world benchmarking framework for
evaluating clinical machine learning models along four integrity dimensions:

- **R — Reliability** (clean-world discrimination)  
- **S — Robustness** (performance under realistic perturbations)  
- **C — Calibration Stability** (drift in calibration across worlds)  
- **E — Explainability Stability** (stability of SHAP-based attributions)

All experiments are run on fully synthetic, clinically-inspired data
(no real patient-level data are included in this repository).

---

## Project Overview

Modern clinical ML models are often evaluated only on a single, clean test set
with metrics like AUROC or AUPRC. In practice, however, deployed models must
survive:

- hospital and population shifts,  
- missing and noisy laboratory measurements,  
- corrupted surrogate markers,  
- nonlinear distortions and label noise, and  
- changing explanation patterns when we inspect them with SHAP or other tools.

This repository provides:

- A **multi-world synthetic benchmark** with seven worlds  
  \(one clean reference world \(W_A\) and six perturbed worlds \(W_B\)–\(W_G\)\),
- A **20-model “model zoo”** covering linear, tree-based, boosting, kernel,
  neural, and Bayesian families,
- A **composite integrity index (RSCE)** that combines:
  - **R**eliability
  - **S**robustness
  - **C**alibration stability
  - **E**xplainability stability,
- **Ablation studies** (no-WBV, no-calibration) and  
- **SHAP-based explainability stability analysis** across worlds.

The goal is not to replace real-world external validation, but to provide a
transparent, controllable testbed for stress-testing clinical ML pipelines.

---

## Repository Structure

A suggested structure for this repository is:

```text
.
├── README.md
├── Python Code
│   ├── ablation_no_calibration.py
│   ├── ablation_no_WBV.py
│   ├── analyze_RSEC_plots.py
│   ├── compute_E_All_score.py
│   ├── compute_E_score.py
│   ├── compute_R_S_Cscores.py
│   ├── compute_R_S_scores.py
│   ├── config.py
│   ├── preprocess.py
│   ├── rsce_centroid_plot.py
│   ├── rsce_cluster_map.py
│   ├── rsce_error_bars.py
│   ├── rsce_family_radar.py
│   ├── RSCE_plot.py
│   ├── RSCE_sensitivity.py
│   ├── rsce_violinplot.py
│   ├── RSCE_Worldrank.py
│   ├── SHAP_Family.py
│   ├── SHAP_family_hierarchical.py
│   ├── shap_visualizations.py
│   ├── WorldA_Ideal.py
│   ├── WorldB_hospital_skew.py
│   ├── WorldC_full_standalone.py
│   ├── WorldD_distribution_shift.py
│   ├── WorldE_wbv_errors.py
│   ├── WorldF_strong_nonlinear.py
│   ├── worldG_label_noise.py
│   ├── WorldG_plots.py
│   ├── nhanes_merge_Dataset.py
│   ├── nhanes_World_metrics_RSCE_fullscore.py
│   ├── Merge_MIMIC-IV_and_Preparing.py
│   ├── RSCE_MIMIC-IV.py
│   └── RSCE_MIMIC-IV_Plot.py
├── NHaNES_Real_Data_Set
│   ├── AUQ_L.xpt
│   ├── BIOPRO_L.xpt
│   ├── BMX_L.xpt
│   ├── CBC_L.xpt
│   ├── DEMO_L.xpt
│   ├── FASTQX_L.xpt
│   ├── GLU_L.xpt
│   ├── HDL_L.XPT
│   ├── TCHOL_L.xpt
│   └── TRIGLY_L.xpt
├── Result_Figure
│   ├── bar_C.png
│   ├── bar_E.png
│   ├── bar_R.png
│   ├── bar_RSEC_mean.png
│   ├── bar_S.png
│   ├── ExtraTrees_A_vs_F_mean_abs_shap.png
│   ├── ExtraTrees_WorldA_bar.png
│   ├── ExtraTrees_WorldA_beeswarm.png
│   ├── ExtraTrees_WorldA_dependence_BMI.png
│   ├── ExtraTrees_WorldA_dependence_TG0h.png
│   ├── ExtraTrees_WorldA_dependence_WBV.png
│   ├── ExtraTrees_WorldF_bar.png
│   ├── ExtraTrees_WorldF_beeswarm.png
│   ├── ExtraTrees_WorldF_dependence_BMI.png
│   ├── ExtraTrees_WorldF_dependence_TG0h.png
│   ├── ExtraTrees_WorldF_dependence_WBV.png
│   ├── G1_auc_vs_noise.png
│   ├── G1_brier_vs_noise.png
│   ├── G2_auc_bar.png
│   ├── G2_brier_bar.png
│   ├── GradientBoosting_A_vs_F_mean_abs_shap.png
│   ├── GradientBoosting_WorldA_bar.png
│   ├── GradientBoosting_WorldA_beeswarm.png
│   ├── GradientBoosting_WorldA_dependence_BMI.png
│   ├── GradientBoosting_WorldA_dependence_TG0h.png
│   ├── GradientBoosting_WorldA_dependence_WBV.png
│   ├── GradientBoosting_WorldF_bar.png
│   ├── GradientBoosting_WorldF_beeswarm.png
│   ├── GradientBoosting_WorldF_dependence_BMI.png
│   ├── GradientBoosting_WorldF_dependence_TG0h.png
│   ├── GradientBoosting_WorldF_dependence_WBV.png
│   ├── heatmap_R_S_C_E.png
│   ├── Logistic_L1_A_vs_F_mean_abs_shap.png
│   ├── Logistic_L1_WorldA_bar.png
│   ├── Logistic_L1_WorldA_beeswarm.png
│   ├── Logistic_L1_WorldA_dependence_BMI.png
│   ├── Logistic_L1_WorldA_dependence_TG0h.png
│   ├── Logistic_L1_WorldA_dependence_WBV.png
│   ├── Logistic_L1_WorldF_bar.png
│   ├── Logistic_L1_WorldF_beeswarm.png
│   ├── Logistic_L1_WorldF_dependence_BMI.png
│   ├── Logistic_L1_WorldF_dependence_TG0h.png
│   ├── Logistic_L1_WorldF_dependence_WBV.png
│   ├── MLP_A_vs_F_mean_abs_shap.png
│   ├── MLP_WorldA_bar.png
│   ├── MLP_WorldA_beeswarm.png
│   ├── MLP_WorldA_dependence_BMI.png
│   ├── MLP_WorldA_dependence_TG0h.png
│   ├── MLP_WorldA_dependence_WBV.png
│   ├── MLP_WorldF_bar.png
│   ├── MLP_WorldF_beeswarm.png
│   ├── MLP_WorldF_dependence_BMI.png
│   ├── MLP_WorldF_dependence_TG0h.png
│   ├── MLP_WorldF_dependence_WBV.png
│   ├── radar_R_S_C_E_all_models.png
│   ├── RandomForest_A_vs_F_mean_abs_shap.png
│   ├── RandomForest_WorldA_bar.png
│   ├── RandomForest_WorldA_beeswarm.png
│   ├── RandomForest_WorldA_dependence_BMI.png
│   ├── RandomForest_WorldA_dependence_TG0h.png
│   ├── RandomForest_WorldA_dependence_WBV.png
│   ├── RandomForest_WorldF_bar.png
│   ├── RandomForest_WorldF_beeswarm.png
│   ├── RandomForest_WorldF_dependence_BMI.png
│   ├── RandomForest_WorldF_dependence_TG0h.png
│   ├── RandomForest_WorldF_dependence_WBV.png
│   ├── RSCE_3D_cube.png
│   ├── RSCE_bar_with_errorbars.png
│   ├── RSCE_barplot.png
│   ├── RSCE_cluster_map.png
│   ├── RSCE_family_bar.png
│   ├── RSCE_family_centroid_scatter.png
│   ├── RSCE_family_radar.png
│   ├── RSCE_heatmap.png
│   ├── RSCE_radar.png
│   ├── RSCE_sensitivity.png
│   ├── RSCE_violin.png
│   ├── scatter_3D_R_S_E.png
│   ├── scatter_C_vs_E.png
│   ├── scatter_R_vs_E.png
│   ├── scatter_R_vs_S.png
│   ├── scatter_S_vs_E.png
│   ├── SHAP_family_Bayes_top5.png
│   ├── SHAP_family_Boosting_top5.png
│   ├── SHAP_family_hierarchical_map.png
│   ├── SHAP_Family_importance.png
│   ├── SHAP_family_Kernel_SVM_top5.png
│   ├── SHAP_family_Linear_models_top5.png
│   ├── SHAP_family_Neural_network_top5.png
│   ├── SHAP_family_Tree-based_bagging_top5.png
│   ├── SVC_RBF_A_vs_F_mean_abs_shap.png
│   ├── SVC_RBF_WorldA_bar.png
│   ├── SVC_RBF_WorldA_beeswarm.png
│   ├── SVC_RBF_WorldA_dependence_BMI.png
│   ├── SVC_RBF_WorldA_dependence_TG0h.png
│   ├── SVC_RBF_WorldA_dependence_WBV.png
│   ├── SVC_RBF_WorldF_bar.png
│   ├── SVC_RBF_WorldF_beeswarm.png
│   ├── SVC_RBF_WorldF_dependence_BMI.png
│   ├── SVC_RBF_WorldF_dependence_TG0h.png
│   ├── SVC_RBF_WorldF_dependence_WBV.png
│   ├── world_rank_spearman_heatmap.png
│   ├── world_rank_spearman_vs_WorldA.png
│   ├── XGBoost_A_vs_F_mean_abs_shap.png
│   ├── XGBoost_WorldA_bar.png
│   ├── XGBoost_WorldA_beeswarm.png
│   ├── XGBoost_WorldA_dependence_BMI.png
│   ├── XGBoost_WorldA_dependence_TG0h.png
│   ├── XGBoost_WorldA_dependence_WBV.png
│   ├── XGBoost_WorldF_bar.png
│   ├── XGBoost_WorldF_beeswarm.png
│   ├── XGBoost_WorldF_dependence_BMI.png
│   ├── XGBoost_WorldF_dependence_TG0h.png
│   ├── XGBoost_WorldF_dependence_WBV.png
│   ├── sensitivity_plot.png
│   ├── MIMIC-IV_Result.png
│   ├── nhanes_ECE_result.png
│   ├── nhanes_Logloss_result.png
│   ├── nhanes_AUROC_result.png
│   └── nhanes_Brier_result.png
├── Result_Text
│   ├── Ablation_no_calibration.txt
│   ├── Ablation_no_WBV.txt
│   ├── E scores.txt
│   ├── E scores_All.txt
│   ├── Family centroids.txt
│   ├── Merged & RSCE-computed table.txt
│   ├── R & S scores.txt
│   ├── R_S_C scores.txt
│   ├── World A result.txt
│   ├── World B result.txt
│   ├── World C results.txt
│   ├── World D results.txt
│   ├── World E results.txt
│   ├── World F result.txt
│   ├── World G result.txt
│   └── World rank matrix.txt
├── Result_npz, Json, Joblib
│   ├── summary.json
│   ├── preprocessor.joblib
│   ├── test_arrays.npz
│   └── train_arrays.npz
└── Result_CSV
    ├── ablation_no_calibration_results.csv
    ├── ablation_no_WBV_results.csv
    ├── Baseline.csv
    ├── RSCE_errorbars.csv
    ├── RSCE_family_centroids.csv
    ├── scores_E.csv
    ├── scores_E_all.csv
    ├── Scores_R_S.csv
    ├── scores_R_S_C.csv
    ├── shap_family_importance.csv
    ├── SHAP_family_matrix.csv
    ├── shap_feature_importance_by_model.csv
    ├── world_rank_spearman.csv
    ├── world_rankings.csv
    ├── worldA_Ideal.csv
    ├── worldB_hospital_skew_results.csv
    ├── worldC_noise_missing_outliers_results.csv
    ├── worldD_distribution_shift_results.csv
    ├── worldE_wbv_error_results.csv
    ├── worldF_strong_nonlinear_results.csv
    ├── worldG_label_noise_results.csv
    ├── nhanes_rsce_dataset.csv
    ├── nhanes_rsce_dataset_clean.csv
    ├── nhanes_E_components.csv
    ├── nhanes_E_scores.csv
    ├── nhanes_RSCE_full_scores.csv
    ├── nhanes_world_metrics.csv
    ├── mortality_E_components.csv
    ├── mortality_E_scores.csv
    ├── mortality_RSCE_full_scores.csv
    ├── mortality_world_metrics.csv
    ├── merged_lab_admission_patient.csv
    ├── clear_merged_lab_admission_patient.csv
    ├── analytic_dataset_mortality_all_admissions.csv
    ├── MIMIC-IV_admissions_RealDataset.csv
    ├── MIMIC-IV_labevents_RealDataset.csv
    └── MIMIC-IV_patients_RealDataset.csv



   
