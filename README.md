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
├── requirements.txt
├── config/
│   ├── worlds_config.yaml        # Parameters for synthetic world generation
│   ├── models_config.yaml        # Model zoo + hyperparameter grids
│   └── rsce_config.yaml          # RSCE weights, bootstrap settings, etc.
├── data/
│   ├── world_A.csv               # Reference world
│   ├── world_B.csv               # Hospital shift
│   ├── world_C.csv               # Missingness (MCAR+MAR)
│   ├── world_D.csv               # Noise
│   ├── world_E.csv               # Surrogate corruption
│   ├── world_F.csv               # Nonlinear distortion
│   └── world_G.csv               # Label noise
├── src/
│   ├── generate_worlds.py        # Synthetic generator + perturbation engine
│   ├── train_models.py           # Train model zoo across worlds
│   ├── compute_metrics.py        # Compute AUROC/AUPRC/Brier/log-loss/ECE
│   ├── compute_rsce.py           # Compute R/S/C/E + RSCE
│   ├── explainability_shap.py    # SHAP + explainability stability (E)
│   ├── ablation_no_wbv.py        # Remove WBV and re-evaluate
│   ├── ablation_no_calibration.py# Evaluate raw vs calibrated models
│   └── utils/                    # Shared utility functions
├── results/
│   ├── scores_R_S_C.csv          # Per-model R,S,C components
│   ├── scores_E_all.csv          # Per-model E components
│   ├── RSCE_family_centroids.csv # Family-level centroids (R,S,C,E,RSCE)
│   ├── RSCE_errorbars.csv        # Bootstrap CIs for RSCE
│   ├── ablation_no_WBV_results.csv
│   ├── ablation_no_calibration_results.csv
│   ├── shap_family_importance.csv
│   ├── SHAP_family_matrix.csv
│   ├── shap_feature_importance_by_model.csv
│   └── world_rankings.csv        # Rankings of models across worlds
└── figures/
    ├── RSCE_barplot.png
    ├── RSCE_heatmap.png
    ├── RSCE_family_bar.png
    ├── RSCE_family_radar.png
    ├── RSCE_violin.png
    ├── RSCE_sensitivity.png


    ├── world_rank_spearman_heatmap.png
    ├── world_rank_spearman_vs_WorldA.png
    ├── SHAP_family_hierarchical_map.png
    ├── SHAP_Family_importance.png
    └── RSCE_bar_with_errorbars.png# 7World_ML_Benchmark
