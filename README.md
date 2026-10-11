# PhysioTransformer

**Sport-Aware Transformer for Non-Invasive Lactate Threshold Estimation Across Multiple Endurance Disciplines**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![Status: Under Review](https://img.shields.io/badge/Status-Under%20Review%20at%20Comp%20Biol%20Med-yellow.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📄 Paper & Preprint

> **Eskandarisani, M., & Daryanoosh, F.**  
> *Can AI Predict Lactate Thresholds Without Blood Samples? Deep Learning Succeeds Within Sports but Fails Across Movement Modalities.*  
> **Status:** Under Review at *Computers in Biology and Medicine*.  
> **Preprint DOI:** [10.21203/rs.3.rs-10811779/v1](https://doi.org/10.21203/rs.3.rs-10811779/v1)[cite: 48]

---

## 🧠 Overview

PhysioTransformer is an attention-based deep learning framework engineered for non-invasive post-test estimation of the aerobic lactate threshold ($HR_{2\text{mmol}\cdot\text{L}^{-1}}$) using dynamic physiological time-series and static anthropometric features.

**Key Architectural Features:**
- **Learnable Sport Embeddings:** Encodes categorical movement modalities (running, cycling, rowing, kayaking) into dense vector representations.
- **Transformer Encoder:** 4-layer multi-head self-attention backbone ($h = 8$, $d_{\text{model}} = 128$) with sinusoidal positional encodings.
- **Attention Pooling & Dual Heads:** Global attention pooling feeding multi-task heads for stage-wise curve reconstruction and soft-argmax ordinal regression.
- **Strict Subject-Grouped CV:** Evaluated using a subject-grouped 5-fold cross-validation protocol to eliminate data leakage across athlete folds.

**Main Results (Subject-Grouped 5-Fold CV, $N = 819$):**

| Metric | Value |
|--------|-------|
| Overall MAE | $5.11 \pm 0.53\ \text{bpm}$ |
| Overall RMSE | $7.33 \pm 1.55\ \text{bpm}$ |
| Point Estimation $R^2$ | $0.609 \pm 0.099$ |
| Curve Trajectory $R^2$ | $0.864 \pm 0.005$ |
| Bland-Altman Bias | $+0.49\ \text{bpm}$ (95% LoA: $[-17.82,\ +18.80]\ \text{bpm}$) |

---

## 📁 Repository Structure

```text
PhysioTransformer/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── models/
│   ├── model_full.py              # Full PhysioTransformer (main model)
│   ├── ablation_no_sport_emb.py   # Ablation: no sport embedding
│   └── ablation_sport_heads.py    # Ablation: per-sport LT heads
│
├── training/
│   ├── train.py                   # Main 5-fold training loop
│   ├── loso_validation.py         # Leave-one-sport-out validation
│   └── loso_no_kayak.py           # Sensitivity analysis (kayak excluded)
│
├── analysis/
│   ├── analysis_and_figures.py    # Generate all publication figures
│   ├── baseline_comparison.py     # LR / RF / XGB / LSTM baselines
│   ├── post_hoc_analysis.py       # Uncertainty calibration + Wilcoxon tests
│   └── data_quality_filter.py     # Quality control pipeline
│
├── results/
│   ├── loso_results.json
│   ├── baseline_results.json
│   ├── loso_no_kayak_results.json
│   └── figures/
│
└── data/
    └── README.md                  # Instructions to download from Zenodo
