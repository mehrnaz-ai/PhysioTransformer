"""
analysis_and_figures.py — Complete Analysis and Publication Figures

Requires (same directory):
  model_full.py / ablation_no_sport_emb.py / ablation_sport_heads.py
  checkpoint_full_fold{1-5}.pt
  checkpoint_no_emb_fold{1-5}.pt
  checkpoint_sport_heads_fold{1-5}.pt
  data/  (original .xlsx files)

Outputs → figures/
  fig1_ablation.pdf
  fig2_pred_vs_true.pdf
  fig3_bland_altman.pdf
  fig4_error_by_sport.pdf
  fig5_lactate_curves.pdf
  fig6_attention_weights.pdf
  fig7_uncertainty_calibration.pdf

Cache:
  predictions_full.pkl  — re-run figures instantly without re-loading checkpoints
"""

import os
import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ── imports from training files ────────────────────────────────────────────────
sys.path.insert(0, ".")
from model_full import (
    LactateDataset, collate_fn,
    PhysioTransformerFull,
    SPORT_TO_IDX, IDX_TO_SPORT,
)
from ablation_no_sport_emb import PhysioTransformerNoEmb
from ablation_sport_heads   import PhysioTransformerSportHeads

# ── config ─────────────────────────────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
SEED      = 42
OUT_DIR   = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── known ablation results (from training output) ──────────────────────────────
ABLATION = {
    "Full Model": {
        "mae": 5.56, "mae_std": 0.64,
        "rmse": 8.96, "rmse_std": 2.68,
        "curve_r2": 0.856, "curve_r2_std": 0.008,
        "lt_r2": 0.516, "lt_r2_std": 0.139,
    },
    "No Sport\nEmbedding": {
        "mae": 5.69, "mae_std": 0.68,
        "rmse": 9.26, "rmse_std": 2.69,
        "curve_r2": 0.841, "curve_r2_std": 0.008,
        "lt_r2": 0.483, "lt_r2_std": 0.144,
    },
    "Sport-Specific\nLT Heads": {
        "mae": 5.89, "mae_std": 0.70,
        "rmse": 9.31, "rmse_std": 2.60,
        "curve_r2": 0.848, "curve_r2_std": 0.008,
        "lt_r2": 0.480, "lt_r2_std": 0.121,
    },
}

# ── colour palette (Okabe-Ito, colour-blind safe) ─────────────────────────────
SPORT_COLOR = {
    "running": "#E69F00",
    "cycling": "#56B4E9",
    "rowing":  "#009E73",
    "kayak":   "#CC79A7",
    "unknown": "#999999",
}
MODEL_COLOR = {
    "Full Model":             "#2166AC",
    "No Sport\nEmbedding":   "#999999",
    "Sport-Specific\nLT Heads": "#999999",
}

# ── publication matplotlib style ──────────────────────────────────────────────
def set_pub_style():
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          10,
        "axes.labelsize":     11,
        "axes.titlesize":     11,
        "axes.titleweight":   "bold",
        "legend.fontsize":    9,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          False,
    })

set_pub_style()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PREDICTION COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_predictions(model, test_loader, fold):
    """Run inference on one fold's test set, return list of sample dicts."""
    model.eval()
    rows = []

    for X, y, L, lt, hrmax, sport, athlete_ids in test_loader:
        X_dev     = X.to(DEVICE)
        sport_dev = sport.to(DEVICE)
        hrmax_dev = hrmax.to(DEVICE)

        curve, lt_pred, logvar = model(X_dev, L, sport_dev)
        std = torch.exp(0.5 * logvar.clamp(-5.0, 2.0)).squeeze(-1)

        lt_true_bpm = (lt.to(DEVICE) * hrmax_dev).cpu().numpy()
        lt_pred_bpm = (lt_pred       * hrmax_dev).cpu().numpy()
        lt_std_bpm  = (std           * hrmax_dev).cpu().numpy()

        for i in range(X.size(0)):
            Li = L[i].item()
            c_pred = torch.expm1(curve[i, :Li, 0].clamp(-3.0, 4.0)).cpu().numpy()
            c_true = torch.expm1(y[i,   :Li, 0].clamp(-3.0, 4.0)).cpu().numpy()

            rows.append({
                "athlete_id":    athlete_ids[i],
                "sport":         IDX_TO_SPORT[sport[i].item()],
                "fold":          fold,
                "lt_true_bpm":   float(lt_true_bpm[i]),
                "lt_pred_bpm":   float(lt_pred_bpm[i]),
                "lt_std_bpm":    float(lt_std_bpm[i]),
                "hrmax":         float(hrmax[i].item()),
                "error_bpm":     float(lt_pred_bpm[i] - lt_true_bpm[i]),
                "abs_error_bpm": float(abs(lt_pred_bpm[i] - lt_true_bpm[i])),
                "curve_pred":    c_pred,
                "curve_true":    c_true,
                "seq_len":       Li,
                # raw x features (cpu) for attention extraction later
                "X_raw":         X[i, :Li].numpy(),
                "sport_idx":     int(sport[i].item()),
            })
    return rows


def load_all_folds(model_class, ckpt_pattern, dataset, groups):
    """Load each fold's best checkpoint, collect all test predictions."""
    gkf  = GroupKFold(n_splits=5)
    all_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(np.arange(len(dataset)), groups=groups)
    ):
        ckpt_path = ckpt_pattern.format(fold + 1)
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: {ckpt_path} not found — skipping fold {fold+1}")
            continue

        test_loader = DataLoader(
            Subset(dataset, test_idx), batch_size=32,
            shuffle=False, collate_fn=collate_fn,
        )
        model = model_class(dataset[0][0].shape[1]).to(DEVICE)
        ckpt  = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])

        rows = collect_predictions(model, test_loader, fold + 1)
        all_rows.extend(rows)
        print(f"  Fold {fold+1}: {len(rows)} test samples loaded "
              f"(best epoch={ckpt.get('epoch','?')}, "
              f"val MAE={ckpt.get('val_mae',float('nan')):.2f})")

    return all_rows


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — STATISTICAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(values, B=2000, alpha=0.05, seed=42):
    """Bootstrap CI for the mean of `values`."""
    rng  = np.random.default_rng(seed)
    boot = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(B)
    ])
    return (np.mean(values),
            np.percentile(boot, 100 * alpha / 2),
            np.percentile(boot, 100 * (1 - alpha / 2)))


def icc21(y_true, y_pred):
    """ICC(2,1) — two-way mixed, absolute agreement, single measures."""
    n       = len(y_true)
    ratings = np.column_stack([y_true, y_pred])
    gm      = ratings.mean()
    msb     = 2 * ((ratings.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msw     = ((ratings - ratings.mean(axis=1, keepdims=True)) ** 2).sum() / n
    return (msb - msw) / (msb + msw)


def calibration_coverage(lt_true, lt_pred, lt_std):
    """Fraction of true values inside ±1σ and ±2σ prediction intervals."""
    cov1 = np.mean(np.abs(lt_true - lt_pred) < 1.0 * lt_std)
    cov2 = np.mean(np.abs(lt_true - lt_pred) < 2.0 * lt_std)
    return cov1, cov2


def print_full_stats(df):
    lt_true = df["lt_true_bpm"].values
    lt_pred = df["lt_pred_bpm"].values
    lt_std  = df["lt_std_bpm"].values
    errors  = df["error_bpm"].values
    ae      = df["abs_error_bpm"].values

    mae_mean, mae_lo, mae_hi = bootstrap_ci(ae)
    bias    = errors.mean()
    sd_diff = errors.std()
    loa_lo  = bias - 1.96 * sd_diff
    loa_hi  = bias + 1.96 * sd_diff
    icc     = icc21(lt_true, lt_pred)
    cov1, cov2 = calibration_coverage(lt_true, lt_pred, lt_std)
    unc_r, unc_p = stats.pearsonr(lt_std, ae)

    print(f"\n{'='*60}")
    print("FULL MODEL — COMPLETE RESULTS")
    print(f"{'='*60}")
    print(f"N samples          :  {len(df)}")
    print(f"LT MAE             :  {mae_mean:.2f} bpm  "
          f"(95% CI: {mae_lo:.2f}–{mae_hi:.2f})")
    print(f"LT RMSE            :  {np.sqrt((errors**2).mean()):.2f} bpm")
    print(f"Bias               :  {bias:+.2f} bpm")
    print(f"Limits of Agreement:  [{loa_lo:.2f}, {loa_hi:.2f}] bpm")
    print(f"ICC(2,1)           :  {icc:.3f}")
    print(f"Uncertainty r      :  {unc_r:.3f}  (p={unc_p:.4f})")
    print(f"Coverage at 1σ     :  {cov1:.1%}  (expected 68.3%)")
    print(f"Coverage at 2σ     :  {cov2:.1%}  (expected 95.4%)")

    print(f"\n{'─'*60}")
    print("BY SPORT")
    print(f"{'─'*60}")
    for sp in ["running", "cycling", "rowing", "kayak"]:
        sub = df[df["sport"] == sp]
        if len(sub) == 0:
            continue
        sp_ae = sub["abs_error_bpm"].values
        sp_e  = sub["error_bpm"].values
        print(f"  {sp:<10}  N={len(sub):3d}  "
              f"MAE={sp_ae.mean():.2f}  "
              f"RMSE={np.sqrt((sp_e**2).mean()):.2f}  "
              f"Bias={sp_e.mean():+.2f}")

    print(f"\n{'─'*60}")
    print("ABLATION SUMMARY")
    print(f"{'─'*60}")
    header = f"{'Model':<28} {'MAE':>6} {'±SD':>5}  {'ΔMAE':>6}  {'Curve R²':>9}  {'LT R²':>7}"
    print(header)
    print("─" * len(header))
    ref_mae = ABLATION["Full Model"]["mae"]
    for name, r in ABLATION.items():
        delta = f"+{r['mae']-ref_mae:.2f}" if r["mae"] != ref_mae else "—"
        label = name.replace("\n", " ")
        print(f"  {label:<26} {r['mae']:>6.2f} ±{r['mae_std']:.2f}  "
              f"{delta:>6}  "
              f"{r['curve_r2']:>9.3f}  "
              f"{r['lt_r2']:>7.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FIGURE 1: ABLATION COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def fig_ablation():
    models  = list(ABLATION.keys())
    n       = len(models)
    x       = np.arange(n)
    colors  = ["#2166AC", "#AAAAAA", "#AAAAAA"]   # highlight full model

    metrics = [
        ("mae",       "mae_std",       "LT MAE (bpm)",   True),
        ("curve_r2",  "curve_r2_std",  "Curve R²",        False),
        ("lt_r2",     "lt_r2_std",     "LT R²",           False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    for ax, (key, std_key, ylabel, lower_better) in zip(axes, metrics):
        vals = np.array([ABLATION[m][key]     for m in models])
        errs = np.array([ABLATION[m][std_key] for m in models])

        bars = ax.bar(x, vals, yerr=errs, color=colors,
                      capsize=5, error_kw={"lw": 1.5, "capthick": 1.5},
                      zorder=3, edgecolor="white", linewidth=0.5)

        # value labels on top of each bar
        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + e + (0.04 if not lower_better else 0.12),
                    f"{v:.3f}" if not lower_better else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

        arrow = "↓ better" if lower_better else "↑ better"
        ax.set_title(f"{ylabel}\n({arrow})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("Sport-Specific\n", "Sport-\nSpecific\n")
                            for m in models], fontsize=8.5)
        ax.set_ylabel(ylabel)

        # add delta annotations for non-full models
        ref = vals[0]
        for i in range(1, n):
            delta = vals[i] - ref
            sign  = "+" if delta > 0 else ""
            ax.text(i, 0.02, f"Δ {sign}{delta:.2f}",
                    ha="center", va="bottom", fontsize=7.5,
                    color="#CC0000", transform=ax.get_xaxis_transform())

    axes[0].set_ylim(0, max(ABLATION[m]["mae"] for m in models) * 1.25)

    fig.suptitle("Ablation Study — Component Contribution",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "fig1_ablation.pdf"
    plt.savefig(out)
    plt.close()
    print(f"✓  fig1_ablation.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FIGURE 2: PREDICTED vs TRUE SCATTER
# ══════════════════════════════════════════════════════════════════════════════

def fig_pred_vs_true(df):
    fig, ax = plt.subplots(figsize=(6, 6))

    sports = [s for s in ["running", "cycling", "rowing", "kayak"]
              if s in df["sport"].values]

    handles = []
    for sp in sports:
        sub  = df[df["sport"] == sp]
        sc   = ax.scatter(sub["lt_true_bpm"], sub["lt_pred_bpm"],
                          c=SPORT_COLOR[sp], alpha=0.55, s=22,
                          edgecolors="none", zorder=3, label=sp.capitalize())
        handles.append(mpatches.Patch(color=SPORT_COLOR[sp],
                                      label=f"{sp.capitalize()} (n={len(sub)})"))

    # identity line
    lo = min(df["lt_true_bpm"].min(), df["lt_pred_bpm"].min()) - 2
    hi = max(df["lt_true_bpm"].max(), df["lt_pred_bpm"].max()) + 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect prediction")
    ax.set_xlim(lo, hi);  ax.set_ylim(lo, hi)

    # stats box
    ae    = df["abs_error_bpm"].values
    r2    = 1 - ((df["lt_pred_bpm"] - df["lt_true_bpm"])**2).sum() / \
                ((df["lt_true_bpm"] - df["lt_true_bpm"].mean())**2).sum()
    icc   = icc21(df["lt_true_bpm"].values, df["lt_pred_bpm"].values)
    txt   = f"MAE  = {ae.mean():.2f} bpm\nR²   = {r2:.3f}\nICC  = {icc:.3f}\nn    = {len(df)}"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    ax.set_xlabel("True LT (bpm)");  ax.set_ylabel("Predicted LT (bpm)")
    ax.set_title("Predicted vs. True Lactate Threshold", fontweight="bold")
    ax.legend(handles=handles, fontsize=9, framealpha=0.9)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_pred_vs_true.pdf")
    plt.close()
    print(f"✓  fig2_pred_vs_true.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FIGURE 3: BLAND-ALTMAN
# ══════════════════════════════════════════════════════════════════════════════

def fig_bland_altman(df):
    fig, ax = plt.subplots(figsize=(7, 5))

    mean_val = (df["lt_pred_bpm"] + df["lt_true_bpm"]) / 2
    diff_val =  df["lt_pred_bpm"] - df["lt_true_bpm"]
    bias     = diff_val.mean()
    sd       = diff_val.std()
    loa_lo   = bias - 1.96 * sd
    loa_hi   = bias + 1.96 * sd

    sports = [s for s in ["running", "cycling", "rowing", "kayak"]
              if s in df["sport"].values]
    handles = []
    for sp in sports:
        sub = df[df["sport"] == sp]
        ax.scatter(
            (sub["lt_pred_bpm"] + sub["lt_true_bpm"]) / 2,
            sub["lt_pred_bpm"] - sub["lt_true_bpm"],
            c=SPORT_COLOR[sp], alpha=0.55, s=22, edgecolors="none", zorder=3,
        )
        handles.append(mpatches.Patch(color=SPORT_COLOR[sp],
                                      label=f"{sp.capitalize()} (n={len(sub)})"))

    x_range = np.array([mean_val.min() - 2, mean_val.max() + 2])
    ax.axhline(bias,   color="#CC0000", lw=1.8, ls="--",
               label=f"Bias = {bias:+.2f} bpm", zorder=4)
    ax.axhline(loa_hi, color="#E69F00", lw=1.4, ls=":",
               label=f"+1.96 SD = {loa_hi:.2f} bpm", zorder=4)
    ax.axhline(loa_lo, color="#E69F00", lw=1.4, ls=":",
               label=f"−1.96 SD = {loa_lo:.2f} bpm", zorder=4)
    ax.axhline(0, color="black", lw=0.6, ls="-", alpha=0.3)

    # shaded LoA band
    ax.fill_between(x_range, loa_lo, loa_hi, alpha=0.06,
                    color="#E69F00", zorder=2)

    ax.set_xlabel("Mean of Predicted and True LT (bpm)")
    ax.set_ylabel("Predicted − True LT (bpm)")
    ax.set_title("Bland–Altman Plot: LT Estimation", fontweight="bold")
    ax.set_xlim(x_range)

    all_handles = handles + [
        plt.Line2D([0], [0], color="#CC0000", lw=1.8, ls="--",
                   label=f"Bias = {bias:+.2f} bpm"),
        plt.Line2D([0], [0], color="#E69F00", lw=1.4, ls=":",
                   label=f"95% LoA: [{loa_lo:.1f}, {loa_hi:.1f}] bpm"),
    ]
    ax.legend(handles=all_handles, fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig3_bland_altman.pdf")
    plt.close()
    print(f"✓  fig3_bland_altman.pdf")
    return bias, sd, loa_lo, loa_hi


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FIGURE 4: ERROR BY SPORT
# ══════════════════════════════════════════════════════════════════════════════

def fig_error_by_sport(df):
    sports  = [s for s in ["running", "cycling", "rowing", "kayak"]
               if s in df["sport"].values]
    fig, ax = plt.subplots(figsize=(7, 5))

    data_by_sport = [df[df["sport"] == sp]["abs_error_bpm"].values
                     for sp in sports]

    bp = ax.boxplot(data_by_sport, patch_artist=True, notch=False,
                    widths=0.45, showfliers=False,
                    medianprops={"color": "black", "lw": 2},
                    whiskerprops={"lw": 1.2},
                    capprops={"lw": 1.2})

    for patch, sp in zip(bp["boxes"], sports):
        patch.set_facecolor(SPORT_COLOR[sp])
        patch.set_alpha(0.7)

    # overlay individual points with jitter
    rng = np.random.default_rng(42)
    for i, (sp, data) in enumerate(zip(sports, data_by_sport), start=1):
        jitter = rng.uniform(-0.15, 0.15, len(data))
        ax.scatter(np.full(len(data), i) + jitter, data,
                   color=SPORT_COLOR[sp], alpha=0.35, s=12,
                   edgecolors="none", zorder=3)
        # N annotation
        ax.text(i, -0.8, f"n={len(data)}", ha="center",
                fontsize=8.5, color="gray",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(range(1, len(sports) + 1))
    ax.set_xticklabels([s.capitalize() for s in sports])
    ax.set_ylabel("|Predicted − True| LT (bpm)")
    ax.set_title("Absolute LT Error by Sport", fontweight="bold")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_error_by_sport.pdf")
    plt.close()
    print(f"✓  fig4_error_by_sport.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — FIGURE 5: REPRESENTATIVE LACTATE CURVES
# ══════════════════════════════════════════════════════════════════════════════

def fig_lactate_curves(rows, df):
    """4-panel: best case, worst case, one per rare sport, highest uncertainty."""
    ae_vals  = np.array([r["abs_error_bpm"] for r in rows])
    std_vals = np.array([r["lt_std_bpm"]    for r in rows])
    med_ae   = float(np.median(ae_vals))

    best     = rows[int(np.argmin(ae_vals))]
    worst    = rows[int(np.argmax(ae_vals))]

    # one non-running example with error < median
    mid_row = None
    for sp in ["rowing", "cycling", "kayak"]:
        cands = [r for r in rows
                 if r["sport"] == sp and r["abs_error_bpm"] < med_ae]
        if cands:
            mid_row = min(cands, key=lambda r: r["abs_error_bpm"])
            break
    if mid_row is None:
        mid_row = rows[len(rows) // 2]

    high_unc = rows[int(np.argmax(std_vals))]

    examples = [
        (best,     "Best case"),
        (worst,    "Worst case"),
        (mid_row,  "Cross-sport transfer"),
        (high_unc, "Highest uncertainty"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, (row, label) in zip(axes, examples):
        c_pred = np.asarray(row["curve_pred"], dtype=float)
        c_true = np.asarray(row["curve_true"], dtype=float)
        n      = len(c_pred)
        x      = np.arange(1, n + 1)
        col    = SPORT_COLOR.get(str(row["sport"]), "#2166AC")

        ax.plot(x, c_true, "k-", lw=2.0, label="True lactate",      zorder=4)
        ax.plot(x, c_pred, "--", lw=2.0, color=col,
                label="Predicted lactate", zorder=4)

        # map LT bpm to approximate stage index
        hrmax      = float(row["hrmax"])
        lt_t_stage = (float(row["lt_true_bpm"]) / hrmax) * n
        lt_p_stage = (float(row["lt_pred_bpm"]) / hrmax) * n
        lw_band    = (float(row["lt_std_bpm"])  / hrmax) * n

        ax.axvline(lt_t_stage, color="black", lw=1.4, ls="--", alpha=0.7,
                   label=f"True LT ({row['lt_true_bpm']:.0f} bpm)")
        ax.axvline(lt_p_stage, color=col, lw=1.4, ls=":", alpha=0.9,
                   label=f"Pred LT ({row['lt_pred_bpm']:.0f} bpm, "
                         f"±{row['lt_std_bpm']:.1f})")
        ax.axvspan(lt_p_stage - lw_band, lt_p_stage + lw_band,
                   alpha=0.10, color=col)

        err_str = f"Error: {float(row['error_bpm']):+.1f} bpm"
        ax.set_title(
            f"{label}  |  {str(row['sport']).capitalize()}  |  {err_str}",
            fontsize=9.5, fontweight="bold",
        )
        ax.set_xlabel("Test stage")
        ax.set_ylabel("Lactate (mmol/L)")
        ax.legend(fontsize=7.5, loc="upper left")

    fig.suptitle("Representative Lactate Curve Predictions",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig5_lactate_curves.pdf")
    plt.close()
    print(f"✓  fig5_lactate_curves.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FIGURE 6: ATTENTION WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_pooling_weights(model, X_np, L, sport_idx):
    """Extract attention pooling weights from PhysioTransformerFull."""
    X    = torch.tensor(X_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    L_t  = torch.tensor([L])
    sp   = torch.tensor([sport_idx]).to(DEVICE)

    B, T, _ = X.shape
    emb  = model.sport_emb(sp).unsqueeze(1).expand(-1, T, -1)
    x    = torch.cat([X, emb], dim=-1)
    z    = model.pos(model.inp(x))
    mask = torch.arange(T, device=DEVICE)[None, :] >= L_t[:, None].to(DEVICE)
    h    = model.encoder(z, src_key_padding_mask=mask)
    h    = model.repr(h)
    score = model.attn_pool(h).masked_fill(mask.unsqueeze(-1), -1e9)
    w     = torch.softmax(score, dim=1)     # (1, T, 1)
    return w[0, :L, 0].cpu().numpy()        # (L,)


def fig_attention_weights(df, dataset, groups):
    """4-panel: attention pooling weights vs. lactate for 4 athletes."""
    # Load fold-1 checkpoint only (for illustration)
    ckpt_path = "checkpoint_full_fold1.pt"
    if not os.path.exists(ckpt_path):
        print(f"  SKIP fig6: {ckpt_path} not found")
        return

    model = PhysioTransformerFull(dataset[0][0].shape[1]).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # pick 4 diverse athletes from the predictions
    sports_order = ["running", "cycling", "rowing", "kayak"]
    picks = []
    for sp in sports_order:
        sub = df[df["sport"] == sp]
        if len(sub) == 0:
            continue
        # prefer median error
        med = sub["abs_error_bpm"].median()
        row = sub.iloc[(sub["abs_error_bpm"] - med).abs().argsort().iloc[0]]
        picks.append(row)
    if len(picks) < 2:
        print("  SKIP fig6: not enough sport diversity in predictions")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, row in zip(axes, picks[:4]):
        attn = extract_pooling_weights(
            model, np.array(row["X_raw"]),
            row["seq_len"], row["sport_idx"]
        )
        lac  = np.array(row["curve_true"])
        x    = np.arange(1, len(lac) + 1)
        col  = SPORT_COLOR.get(row["sport"], "#2166AC")

        # attention bars (left axis)
        ax.bar(x, attn * 100, color=col, alpha=0.45, label="Attn weight (%)")
        ax.set_ylabel("Attention weight (%)", color=col)
        ax.tick_params(axis="y", labelcolor=col)

        # lactate line (right axis)
        ax2 = ax.twinx()
        ax2.plot(x, lac, "k-", lw=2.0, label="Lactate (mmol/L)", zorder=5)
        ax2.set_ylabel("Lactate (mmol/L)")
        ax2.spines["top"].set_visible(False)

        # mark LT stage
        lt_stage = (row["lt_true_bpm"] / row["hrmax"]) * len(lac)
        ax.axvline(lt_stage, color="black", lw=1.5, ls="--",
                   alpha=0.6, label="True LT")

        # peak attention annotation
        peak_stage = np.argmax(attn) + 1
        ax.annotate(f"Peak\nstage {peak_stage}",
                    xy=(peak_stage, attn.max() * 100),
                    xytext=(peak_stage + 0.8, attn.max() * 100 * 0.85),
                    fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.8))

        ax.set_xlabel("Test stage")
        ax.set_title(
            f"{row['sport'].capitalize()}  |  "
            f"Pred {row['lt_pred_bpm']:.0f} bpm  "
            f"True {row['lt_true_bpm']:.0f} bpm",
            fontsize=9.5, fontweight="bold"
        )
        ax.set_xlim(0.5, len(lac) + 0.5)

    # hide unused panels
    for ax in axes[len(picks):]:
        ax.set_visible(False)

    fig.suptitle(
        "Attention Pooling Weights vs. Lactate Curve\n"
        "(Higher weight = model relies more on that test stage for LT prediction)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig6_attention_weights.pdf")
    plt.close()
    print(f"✓  fig6_attention_weights.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — FIGURE 7: UNCERTAINTY CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def fig_uncertainty_calibration(df):
    lt_std = df["lt_std_bpm"].values
    ae     = df["abs_error_bpm"].values
    sports = df["sport"].values

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # — Panel A: scatter std vs |error| —
    ax = axes[0]
    for sp in ["running", "cycling", "rowing", "kayak"]:
        mask = sports == sp
        if mask.sum() == 0:
            continue
        ax.scatter(lt_std[mask], ae[mask], c=SPORT_COLOR[sp],
                   alpha=0.45, s=18, edgecolors="none", label=sp.capitalize())

    lim = max(lt_std.max(), ae.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="Perfect calibration")
    r, p = stats.pearsonr(lt_std, ae)
    ax.text(0.05, 0.93,
            f"Pearson r = {r:.3f}\np = {p:.4f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
    ax.set_xlabel("Predicted uncertainty σ (bpm)")
    ax.set_ylabel("|Predicted − True| LT (bpm)")
    ax.set_title("(a) Uncertainty vs. Absolute Error", fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.9)

    # — Panel B: coverage bars —
    ax = axes[1]
    n_sigmas   = [1, 2]
    coverages  = [np.mean(ae < k * lt_std) * 100 for k in n_sigmas]
    expected   = [68.3, 95.4]
    xpos       = np.array([0, 1])
    w          = 0.3

    bars_obs = ax.bar(xpos - w / 2, coverages, width=w,
                      label="Observed", color="#2166AC", alpha=0.8)
    bars_exp = ax.bar(xpos + w / 2, expected,  width=w,
                      label="Expected (Gaussian)", color="#AAAAAA", alpha=0.8)

    for bar, v in zip(list(bars_obs) + list(bars_exp),
                      coverages + expected):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                f"{v:.1f}%", ha="center", fontsize=9)

    ax.set_xticks(xpos)
    ax.set_xticklabels(["±1σ interval", "±2σ interval"])
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(0, 115)
    ax.set_title("(b) Prediction Interval Coverage", fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle("Uncertainty Calibration Analysis",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig7_uncertainty_calibration.pdf")
    plt.close()
    print(f"✓  fig7_uncertainty_calibration.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    CACHE = "predictions_full.pkl"

    # ── 1. Load or generate predictions ───────────────────────────────────────
    if os.path.exists(CACHE):
        print(f"Loading cached predictions from {CACHE} …")
        with open(CACHE, "rb") as f:
            rows = pickle.load(f)
        print(f"  {len(rows)} samples loaded from cache")
    else:
        print("Loading dataset …")
        ds     = LactateDataset("data")
        groups = [ds[i][-1] for i in range(len(ds))]

        print("\nCollecting full-model predictions across all 5 folds …")
        rows = load_all_folds(
            PhysioTransformerFull,
            "checkpoint_full_fold{}.pt",
            ds, groups,
        )
        with open(CACHE, "wb") as f:
            pickle.dump(rows, f)
        print(f"  Cached {len(rows)} predictions → {CACHE}")

    # ── 2. Build DataFrame ────────────────────────────────────────────────────
    # Separate curve arrays from scalar columns for DataFrame compatibility
    scalar_cols = [
        "athlete_id", "sport", "fold",
        "lt_true_bpm", "lt_pred_bpm", "lt_std_bpm",
        "hrmax", "error_bpm", "abs_error_bpm", "seq_len", "sport_idx",
    ]
    df = pd.DataFrame([{k: r[k] for k in scalar_cols} for r in rows])

    # Keep curve arrays and raw X in the original list for figure functions
    # that need them (fig5, fig6)

    # ── 3. Print full statistics ──────────────────────────────────────────────
    print_full_stats(df)

    # ── 4. Generate all figures ───────────────────────────────────────────────
    print(f"\nGenerating figures → {OUT_DIR}/\n")

    # Fig 1 uses hardcoded ABLATION dict — no predictions needed
    fig_ablation()

    fig_pred_vs_true(df)

    bias, sd, loa_lo, loa_hi = fig_bland_altman(df)
    print(f"     Bias={bias:+.2f} bpm  LoA=[{loa_lo:.2f}, {loa_hi:.2f}]")

    fig_error_by_sport(df)

    # Add curve arrays back for fig 5
    for i, row in enumerate(rows):
        df.at[i, "curve_pred"] = row["curve_pred"]
        df.at[i, "curve_true"] = row["curve_true"]
        df.at[i, "X_raw"]      = row["X_raw"]

    fig_lactate_curves(df)

    # Fig 6 requires dataset for GroupKFold reconstruction
    if not os.path.exists(CACHE):
        # dataset already loaded above
        fig_attention_weights(df, ds, groups)
    else:
        # need to reload dataset for attention extraction
        print("  Reloading dataset for attention weight extraction …")
        ds     = LactateDataset("data")
        groups = [ds[i][-1] for i in range(len(ds))]
        fig_attention_weights(df, ds, groups)

    fig_uncertainty_calibration(df)

    print(f"\n{'='*60}")
    print(f"All figures saved to {OUT_DIR}/")
    print(f"{'='*60}")
