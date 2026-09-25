
# PhysioTransformer

**Sport-Aware Transformer for Non-Invasive Lactate Threshold Estimation Across Multiple Endurance Disciplines**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![Status: Under Review](https://img.shields.io/badge/Status-Under%20Review-yellow.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📄 Paper & Preprint

> **Eskandarisani M, Daryanoosh F.**  
> *Deep learning accurately estimates the 2 mmol·L⁻¹ lactate threshold from heart rate and power data across endurance sports: A cross-validation and transferability study.*  
> **Status:** Under Review at *PLOS ONE*  
> **Preprint DOI:** [10.21203/rs.3.rs-10811779/v1](https://doi.org/10.21203/rs.3.rs-10811779/v1)

---

## 🧠 Overview

PhysioTransformer is a sport-conditioned transformer encoder that estimates the
lactate threshold (LT) non-invasively from heart rate and performance features
extracted during incremental exercise testing.

**Key features:**
- Sport embedding (running / cycling / rowing / kayak)
- 4-layer transformer encoder with 8 attention heads
- Learned attention pooling for physiologically interpretable stage weighting
- Ordinal bin regression (soft-argmax expected value calculation) for continuous LT estimation
- Subject-wise GroupKFold cross-validation (no data leakage)

**Main results (5-fold CV, n = 823):**

| Metric | Value |
|--------|-------|
| LT MAE | 5.56 ± 0.64 bpm |
| 95% CI | [5.08, 6.10] bpm |
| ICC(2,1) | 0.662 |
| Bland-Altman Bias | +0.49 bpm |
| LoA | [−17.82, +18.80] bpm |

---

## 📁 Repository Structure


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

---

## 📦 Dataset

The dataset is **not** included in this repository.
It is publicly available from Zenodo:

> Mooney R, Quinlan LR, Corrión G, Clarke G, Knapp T, O'Laighin G.
> *Physiological graded incremental exercise testing database (v2).*
> Zenodo. 2024.  
> **DOI: [10.5281/zenodo.10841412](https://doi.org/10.5281/zenodo.10841412)**

**Download instructions:**
1. Go to the Zenodo link above
2. Download all `.xlsx` files
3. Place them in the `data/` folder

The database was collected at the Human Performance Laboratory,
School of Medicine, Trinity College Dublin, Ireland.

---

## ⚙️ Installation

```bash
git clone [https://github.com/mehrnaz-ai/PhysioTransformer.git](https://github.com/mehrnaz-ai/PhysioTransformer.git)
cd PhysioTransformer
pip install -r requirements.txt

Requirements: Python 3.11, PyTorch 2.0, CUDA (optional but recommended)
🚀 Reproducing Results
Step 1 — Data Quality Filtering
python analysis/data_quality_filter.py data/

Step 2 — Train Full Model (5-fold CV)
python models/model_full.py

Step 3 — Run Ablations
python models/ablation_no_sport_emb.py
python models/ablation_sport_heads.py

Step 4 — LOSO Validation
python training/loso_validation.py

Step 5 — Sensitivity Analysis (Kayak Exclusion)
python training/loso_no_kayak.py

Step 6 — Baseline Comparison
python analysis/baseline_comparison.py

Step 7 — Generate Figures and Post-hoc Analysis
python analysis/analysis_and_figures.py
python analysis/post_hoc_analysis.py

📊 Key Results
Within-Distribution Performance (5-Fold CV, N = 823)
| Sport | N | MAE (bpm) | RMSE (bpm) | Bias (bpm) |
|---|---|---|---|---|
| Running | 215 | 4.85 | 9.69 | +0.13 |
| Cycling | 284 | 5.86 | 8.31 | +1.00 |
| Rowing | 204 | 4.96 | 6.52 | −0.04 |
| Kayak | 120 | 10.01 | 18.01 | +1.77 |
| Overall | 823 | 5.56 ± 0.64 | 8.96 ± 2.68 | +0.49 |
Baseline Comparison
| Model | MAE (bpm) | R² |
|---|---|---|
| Linear Regression | 8.27 ± 0.98 | 0.014 |
| Random Forest | 7.73 ± 0.61 | 0.279 |
| XGBoost | 7.90 ± 0.72 | 0.256 |
| LSTM | 5.64 ± 0.62 | 0.504 |
| PhysioTransformer | 5.56 ± 0.64 | 0.516 |
📋 Citation
If you use PhysioTransformer or reference this work in your research, please cite the preprint:
@article{eskandarisani2026physiotransformer,
  title   = {Deep learning accurately estimates the 2 mmol·L⁻¹ lactate threshold 
             from heart rate and power data across endurance sports: 
             A cross-validation and transferability study},
  author  = {Eskandarisani, Mehrnaz and Daryanoosh, Farhad},
  journal = {Research Square (Preprint)},
  year    = {2026},
  doi     = {10.21203/rs.3.rs-10811779/v1}
}

Also cite the original dataset:
@dataset{mooney2024physio,
  author    = {Mooney, Ronan and Quinlan, Leo R. and Corrión, Gonzalo 
               and Clarke, Gerard and Knapp, Thomas and O'Laighin, Gearóid},
  title     = {Physiological graded incremental exercise testing database (v2)},
  year      = {2024},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.10841412}
}

📬 Contact
Corresponding author:
Mehrnaz Eskandarisani
mehrnazeskandarisani1@gmail.com
📜 License
This project is licensed under the MIT License — see LICENSE for details.
