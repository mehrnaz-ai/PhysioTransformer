# PhysioTransformer

**Sport-Aware Transformer for Non-Invasive Lactate Threshold Estimation Across Multiple Endurance Disciplines**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📄 Paper

> **EskandariSani M, Daryanoosh F.**

> DOI: [to be added after acceptance]

---

## 🧠 Overview

PhysioTransformer is a sport-conditioned transformer encoder that estimates the
lactate threshold (LT) non-invasively from heart rate and performance features
extracted during incremental exercise testing.

**Key features:**
- Sport embedding (running / cycling / rowing / kayak)
- 4-layer transformer encoder with 8 attention heads
- Learned attention pooling for physiologically interpretable stage weighting
- Ordinal bin regression for LT estimation
- Subject-wise GroupKFold cross-validation (no data leakage)

**Main results (5-fold CV, n = 825):**

| Metric | Value |
|--------|-------|
| LT MAE | 5.56 ± 0.64 bpm |
| 95% CI | [5.08, 6.10] bpm |
| ICC(2,1) | 0.662 |
| Bland-Altman Bias | +0.49 bpm |
| LoA | [−17.82, +18.80] bpm |

---

## 📁 Repository Structure

```
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
│   ├── analysis_and_figures.py    # Generate all 7 publication figures
│   ├── baseline_comparison.py     # LR / RF / XGB / LSTM baselines
│   ├── post_hoc_analysis.py       # Uncertainty calibration + Wilcoxon tests
│   └── data_quality_filter.py     # Quality control pipeline
│
├── results/
│   ├── loso_results.json
│   ├── baseline_results.json
│   ├── loso_no_kayak_results.json
│   └── figures/
│       ├── fig1_ablation.pdf
│       ├── fig2_pred_vs_true.pdf
│       ├── fig3_bland_altman.pdf
│       ├── fig4_error_by_sport.pdf
│       ├── fig5_lactate_curves.pdf
│       ├── fig6_attention_weights.pdf
│       └── fig7_uncertainty_calibration.pdf
│
└── data/
    └── README.md                  # Instructions to download from Zenodo
```

---

## 📦 Dataset

The dataset is **not** included in this repository.
It is publicly available from Zenodo:

> Mooney R, Quinlan LR, Corrión G, Clarke G, Knapp T, O'Laighin G.
> *Physiological graded incremental exercise testing database.*
> Zenodo. 2022.
> **DOI: [10.5281/zenodo.7693648](https://doi.org/10.5281/zenodo.7693648)**

**Download instructions:**
1. Go to the Zenodo link above
2. Download all `.xlsx` files
3. Place them in the `data/` folder

The database was collected at the Human Performance Laboratory,
School of Medicine, Trinity College Dublin, Ireland.

---

## ⚙️ Installation

```bash
git clone https://github.com/mehrnaz-ai/PhysioTransformer.git
cd PhysioTransformer
pip install -r requirements.txt
```

**Requirements:** Python 3.11, PyTorch 2.0, CUDA (optional but recommended)

---

## 🚀 Reproducing Results

### Step 1 — Data Quality Filtering
```bash
python analysis/data_quality_filter.py data/
```

### Step 2 — Train Full Model (5-fold CV)
```bash
python models/model_full.py
# Outputs: checkpoint_full_fold{1-5}.pt
# Time: ~3 hours on GPU
```

### Step 3 — Run Ablations
```bash
python models/ablation_no_sport_emb.py
python models/ablation_sport_heads.py
```

### Step 4 — LOSO Validation
```bash
python training/loso_validation.py
# Outputs: loso_results.json
```

### Step 5 — Sensitivity Analysis (Kayak Exclusion)
```bash
python training/loso_no_kayak.py
# Outputs: loso_no_kayak_results.json
```

### Step 6 — Baseline Comparison
```bash
python analysis/baseline_comparison.py
# Outputs: baseline_results.json
```

### Step 7 — Generate All Figures
```bash
python analysis/analysis_and_figures.py
# Outputs: results/figures/fig1–fig7.pdf
```

### Step 8 — Post-hoc Analysis
```bash
python analysis/post_hoc_analysis.py
# Outputs: uncertainty calibration, Wilcoxon tests, outlier report
```

---

## 📊 Key Results

### Within-Distribution Performance (5-Fold CV)

| Sport | N | MAE (bpm) | RMSE (bpm) | Bias (bpm) |
|-------|---|-----------|-----------|-----------|
| Running | 281 | 4.85 | 9.69 | +0.13 |
| Cycling | 280 | 5.86 | 8.31 | +1.00 |
| Rowing | 210 | 4.96 | 6.52 | −0.04 |
| Kayak | 54 | 10.01 | 18.01 | +1.77 |
| **Overall** | **825** | **5.56 ± 0.64** | **8.96 ± 2.68** | **+0.49** |

### Baseline Comparison

| Model | MAE (bpm) | R² |
|-------|-----------|-----|
| Linear Regression | 8.27 ± 0.98 | 0.014 |
| Random Forest | 7.73 ± 0.61 | 0.279 |
| XGBoost | 7.90 ± 0.72 | 0.256 |
| LSTM | 5.64 ± 0.62 | 0.504 |
| **PhysioTransformer** | **5.56 ± 0.64** | **0.516** |

### Cross-Sport Generalization (LOSO)

| Held-out Sport | Full LOSO MAE | Without Kayak MAE | Δ |
|----------------|--------------|------------------|---|
| Running | 8.95 bpm | 7.12 bpm | −1.83 |
| Cycling | 7.61 bpm | 6.89 bpm | −0.72 |
| Rowing | 7.59 bpm | 7.21 bpm | −0.38 |
| Kayak | 11.27 bpm | N/A | — |

---

## 🏗️ Model Architecture

```
Input (T × 14 features)
    │
    ├── Sport Embedding (5 → 16 dim)
    │       concatenated to input → (T × 30)
    │
    ├── Linear Projection → d_model = 128
    │
    ├── Sinusoidal Positional Encoding
    │
    ├── Transformer Encoder
    │       4 layers × 8 heads
    │       feedforward dim = 1024
    │       dropout = 0.15, GELU
    │
    ├── Shared Representation Head
    │       Linear(128→128) → LayerNorm → GELU → Linear(128→64) → GELU
    │
    ├── Attention Pooling → h_global ∈ ℝ⁶⁴
    │
    ├── Curve Head → per-stage lactate reconstruction
    └── LT Head   → 10 ordinal bins [0.40, 1.00] × HRmax
```

**Training:**
- Loss: L = L_curve(masked MSE) + 0.5 × L_LT(Smooth L1)
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)
- Scheduler: OneCycleLR (max_lr=1e-3, 51 epochs)
- Batch size: 32 | Seeds: PyTorch 42, NumPy 42

---

## 🧪 Ablation Study

| Configuration | MAE (bpm) | ΔMAE | p-value |
|--------------|-----------|------|---------|
| Full Model | 5.56 ± 0.64 | — | — |
| − Sport Embedding | 5.69 ± 0.68 | +0.13 | 0.031* |
| − Shared LT Head | 5.89 ± 0.70 | +0.33 | 0.031* |

*Wilcoxon signed-rank test, paired per-fold (n=5)*

---

## 📋 Citation

If you use PhysioTransformer in your research, please cite:

```bibtex
@article{eskandari2025physiotransformer,
  title   = {PhysioTransformer: Sport-Aware Transformer Modeling for 
             Non-Invasive Lactate Threshold Estimation Across Multiple 
             Endurance Disciplines},
  author  = {Eskandarisani, Mehrnaz and Daryanoosh, Farhad},
  journal = {[Journal Name]},
  year    = {2025},
  doi     = {[to be added]}
}
```

Also cite the original dataset:

```bibtex
@dataset{mooney2022physio,
  author    = {Mooney, Ronan and Quinlan, Leo R. and Corrión, Gonzalo 
               and Clarke, Gerard and Knapp, Thomas and O'Laighin, Gearóid},
  title     = {Physiological graded incremental exercise testing database},
  year      = {2022},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.7693648}
}
```

---

## 📬 Contact

**Corresponding author:**
Mehrnaz Eskandarisani
Mehrnazeskandarisani1@gmail.com

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

The dataset is licensed separately by its original authors (see Zenodo link).
