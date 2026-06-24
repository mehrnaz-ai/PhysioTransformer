"""
post_hoc_analysis.py — Uncertainty Calibration, Attention Validation, Statistical Tests

1. Fix uncertainty via temperature scaling
2. Validate attention peaks near LT inflection
3. Wilcoxon significance tests on ablations
4. Investigate the 101 bpm outlier
"""

import os
import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from model_full import PhysioTransformerFull, IDX_TO_SPORT, DEVICE, LactateDataset
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset
from analysis_and_figures import collate_fn

# ══════════════════════════════════════════════════════════════════════════════
# 1. TEMPERATURE SCALING FOR UNCERTAINTY
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_temperature(lt_true, lt_pred, lt_logvar, T_range=(0.5, 3.0), n_steps=20):
    """Find temperature that minimizes NLL on validation set."""
    temps = np.linspace(T_range[0], T_range[1], n_steps)
    nlls = []

    for T in temps:
        var_scaled = torch.exp(lt_logvar.clamp(-5.0, 2.0)) * (T ** 2)
        nll = (
            0.5 * torch.log(var_scaled + 1e-6)
            + (lt_pred - lt_true) ** 2 / (2.0 * var_scaled + 1e-6)
        ).mean().item()
        nlls.append(nll)

    best_T = temps[np.argmin(nlls)]
    return best_T, np.array(nlls)


def calibrate_uncertainty(lt_std_raw, temperature):
    """Apply temperature scaling to raw uncertainty."""
    return lt_std_raw * temperature


def check_calibration_before_after(df, rows, temperature):
    """Compute calibration metrics before and after temperature scaling."""
    ae  = df["abs_error_bpm"].values
    std_raw = df["lt_std_bpm"].values
    std_cal = std_raw * temperature

    results = {
        "temperature": temperature,
        "before": {
            "correlation": float(stats.pearsonr(std_raw, ae)[0]),
            "p_value": float(stats.pearsonr(std_raw, ae)[1]),
            "coverage_1sigma": float(np.mean(ae < 1.0 * std_raw)),
            "coverage_2sigma": float(np.mean(ae < 2.0 * std_raw)),
        },
        "after": {
            "correlation": float(stats.pearsonr(std_cal, ae)[0]),
            "p_value": float(stats.pearsonr(std_cal, ae)[1]),
            "coverage_1sigma": float(np.mean(ae < 1.0 * std_cal)),
            "coverage_2sigma": float(np.mean(ae < 2.0 * std_cal)),
        },
    }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 2. ATTENTION PEAK VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_attention_peak(model, X_raw, sport_idx):
    """Extract the stage where attention is highest."""
    X = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    L = X.shape[1]
    sp = torch.tensor([sport_idx]).to(DEVICE)

    B, T, _ = X.shape
    emb = model.sport_emb(sp).unsqueeze(1).expand(-1, T, -1)
    x = torch.cat([X, emb], dim=-1)
    z = model.pos(model.inp(x))
    mask = torch.arange(T, device=DEVICE)[None, :] >= L
    h = model.encoder(z, src_key_padding_mask=mask)
    h = model.repr(h)
    score = model.attn_pool(h).masked_fill(mask.unsqueeze(-1), -1e9)
    w = torch.softmax(score, dim=1)

    peak_idx = torch.argmax(w[0, :L, 0]).item()
    return peak_idx, w[0, :L, 0].cpu().numpy()


def validate_attention_peaks(rows, ckpt_path):
    """Check: for each sample, is attention peak within ±2 stages of LT?"""
    model = PhysioTransformerFull(14).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    results = []
    for row in rows:
        X_raw = np.array(row["X_raw"])
        seq_len = row["seq_len"]
        hrmax = row["hrmax"]
        lt_true_bpm = row["lt_true_bpm"]
        sport_idx = row["sport_idx"]

        # Infer approximate LT stage from normalized LT
        lt_norm = lt_true_bpm / hrmax
        lt_stage_approx = lt_norm * seq_len

        # Get attention peak
        peak_idx, weights = extract_attention_peak(model, X_raw, sport_idx)

        within_2 = abs(peak_idx - lt_stage_approx) <= 2
        results.append({
            "lt_stage": float(lt_stage_approx),
            "attn_peak": float(peak_idx),
            "delta": float(abs(peak_idx - lt_stage_approx)),
            "within_2stages": bool(within_2),
            "sport": row["sport"],
        })

    df_attn = pd.DataFrame(results)
    coverage = df_attn["within_2stages"].mean()
    mean_delta = df_attn["delta"].mean()
    median_delta = df_attn["delta"].median()

    return coverage, mean_delta, median_delta, df_attn


# ══════════════════════════════════════════════════════════════════════════════
# 3. WILCOXON SIGNED-RANK TESTS ON ABLATIONS
# ══════════════════════════════════════════════════════════════════════════════

def wilcoxon_ablation_test(full_maes, ablation_maes, name="Ablation"):
    """Test if ablation significantly increases MAE (per-fold paired test)."""
    stat, p = stats.wilcoxon(full_maes, ablation_maes, alternative="less")

    # Effect size: rank-biserial correlation
    n = len(full_maes)
    r = 1.0 - (2.0 * stat) / (n * (n + 1))

    # Mean difference
    delta = np.mean(ablation_maes - full_maes)

    return {
        "ablation": name,
        "w_statistic": float(stat),
        "p_value": float(p),
        "effect_size_r": float(r),
        "mean_delta_mae": float(delta),
        "significant": p < 0.05,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. INVESTIGATE 101 BPM OUTLIER
# ══════════════════════════════════════════════════════════════════════════════

def find_outliers(df, rows, threshold_ae=30):
    """Identify high-error samples."""
    outliers = df[df["abs_error_bpm"] > threshold_ae].copy()
    outliers = outliers.sort_values("abs_error_bpm", ascending=False)

    print(f"\n{'='*70}")
    print(f"OUTLIER ANALYSIS (|error| > {threshold_ae} bpm)")
    print(f"{'='*70}")
    print(f"Found {len(outliers)} outliers out of {len(df)} samples "
          f"({len(outliers)/len(df)*100:.1f}%)\n")

    for idx, row in outliers.head(10).iterrows():
        row_data = rows[idx]
        print(f"Rank {outliers.index.get_loc(idx)+1}:")
        print(f"  Sport:        {row['sport'].capitalize()}")
        print(f"  Athlete:      {row['athlete_id']}")
        print(f"  Fold:         {row['fold']}")
        print(f"  True LT:      {row['lt_true_bpm']:.0f} bpm")
        print(f"  Pred LT:      {row['lt_pred_bpm']:.0f} bpm")
        print(f"  Error:        {row['error_bpm']:+.1f} bpm ({row['error_bpm']/row['lt_true_bpm']*100:+.0f}%)")
        print(f"  Uncertainty:  ±{row['lt_std_bpm']:.1f} bpm")
        print(f"  HRmax:        {row['hrmax']:.0f}")
        print(f"  Test stages:  {row['seq_len']}")
        curve_true = np.array(row_data["curve_true"])
        print(f"  Lactate peak: {curve_true.max():.2f} mmol/L")
        print()

    return outliers


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("Loading cached predictions …")
    with open("predictions_full.pkl", "rb") as f:
        rows = pickle.load(f)

    scalar_cols = [
        "athlete_id", "sport", "fold",
        "lt_true_bpm", "lt_pred_bpm", "lt_std_bpm",
        "hrmax", "error_bpm", "abs_error_bpm", "seq_len", "sport_idx",
    ]
    df = pd.DataFrame([{k: r[k] for k in scalar_cols} for r in rows])

    # ────────────────────────────────────────────────────────────────────────
    # 1. UNCERTAINTY CALIBRATION FIX
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("1. TEMPERATURE SCALING FOR UNCERTAINTY CALIBRATION")
    print("="*70)

    # Find optimal temperature
    lt_true_t = torch.tensor(df["lt_true_bpm"].values, dtype=torch.float32)
    lt_pred_t = torch.tensor(df["lt_pred_bpm"].values, dtype=torch.float32)
    lt_std_t = torch.tensor(df["lt_std_bpm"].values, dtype=torch.float32)
    lt_logvar_t = torch.log(lt_std_t ** 2).unsqueeze(-1)

    best_T, nll_curve = find_optimal_temperature(lt_true_t, lt_pred_t, lt_logvar_t)
    print(f"\nOptimal temperature: T = {best_T:.3f}")

    # Check calibration before/after
    calibration_results = check_calibration_before_after(df, rows, best_T)

    print("\n" + "─" * 70)
    print("BEFORE Temperature Scaling:")
    print("─" * 70)
    print(f"  Correlation (std vs |error|):  r = {calibration_results['before']['correlation']:.3f}  "
          f"(p = {calibration_results['before']['p_value']:.4f})")
    print(f"  Coverage at ±1σ:               {calibration_results['before']['coverage_1sigma']*100:.1f}%  "
          f"(expected 68.3%)")
    print(f"  Coverage at ±2σ:               {calibration_results['before']['coverage_2sigma']*100:.1f}%  "
          f"(expected 95.4%)")

    print("\n" + "─" * 70)
    print(f"AFTER Temperature Scaling (T = {best_T:.3f}):")
    print("─" * 70)
    print(f"  Correlation (std vs |error|):  r = {calibration_results['after']['correlation']:.3f}  "
          f"(p = {calibration_results['after']['p_value']:.4f})")
    print(f"  Coverage at ±1σ:               {calibration_results['after']['coverage_1sigma']*100:.1f}%  "
          f"(expected 68.3%)")
    print(f"  Coverage at ±2σ:               {calibration_results['after']['coverage_2sigma']*100:.1f}%  "
          f"(expected 95.4%)")

    print(f"\n✓ Recommendation: Apply T={best_T:.3f} or REMOVE uncertainty claims from paper")

    # ────────────────────────────────────────────────────────────────────────
    # 2. ATTENTION PEAK VALIDATION
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("2. ATTENTION PEAK VALIDATION")
    print("="*70)

    if os.path.exists("checkpoint_full_fold1.pt"):
        coverage, mean_delta, median_delta, df_attn = validate_attention_peaks(
            rows, "checkpoint_full_fold1.pt"
        )
        print(f"\nAttention peak validation (fold 1 only):")
        print(f"  Coverage: {coverage*100:.1f}% of samples have attention peak "
              f"within ±2 stages of true LT")
        print(f"  Mean delta:   {mean_delta:.2f} stages")
        print(f"  Median delta: {median_delta:.2f} stages")

        print(f"\nBy sport:")
        for sp in ["running", "cycling", "rowing", "kayak"]:
            sub = df_attn[df_attn["sport"] == sp]
            if len(sub) > 0:
                cov = sub["within_2stages"].mean()
                print(f"  {sp:<10}: {cov*100:.1f}% (n={len(sub)})")
    else:
        print("  Checkpoint not found, skipping")

    # ────────────────────────────────────────────────────────────────────────
    # 3. WILCOXON SIGNIFICANCE TESTS
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("3. WILCOXON SIGNED-RANK TESTS — ABLATION SIGNIFICANCE")
    print("="*70)

    # Load ablation fold results (hardcoded from training output)
    # In a full pipeline, these would come from saved fold MAEs
    full_fold_maes = np.array([4.73, 6.70, 5.41, 5.55, 5.42])
    noem_fold_maes = np.array([4.89, 6.87, 5.52, 5.65, 5.54])
    heads_fold_maes = np.array([5.03, 7.06, 5.73, 5.81, 5.76])

    test_noem = wilcoxon_ablation_test(full_fold_maes, noem_fold_maes,
                                       name="No Sport Embedding")
    test_heads = wilcoxon_ablation_test(full_fold_maes, heads_fold_maes,
                                        name="Sport-Specific LT Heads")

    print("\n" + "─" * 70)
    print("No Sport Embedding:")
    print("─" * 70)
    print(f"  W = {test_noem['w_statistic']:.1f}")
    print(f"  p = {test_noem['p_value']:.4f}  {'***' if test_noem['p_value'] < 0.001 else '**' if test_noem['p_value'] < 0.01 else '*' if test_noem['p_value'] < 0.05 else 'ns'}")
    print(f"  Effect size (r): {test_noem['effect_size_r']:.3f}")
    print(f"  Δ MAE: {test_noem['mean_delta_mae']:+.2f} bpm")
    if test_noem["significant"]:
        print(f"  → Sport embedding SIGNIFICANTLY contributes to accuracy")
    else:
        print(f"  → Effect not statistically significant")

    print("\n" + "─" * 70)
    print("Sport-Specific LT Heads:")
    print("─" * 70)
    print(f"  W = {test_heads['w_statistic']:.1f}")
    print(f"  p = {test_heads['p_value']:.4f}  {'***' if test_heads['p_value'] < 0.001 else '**' if test_heads['p_value'] < 0.01 else '*' if test_heads['p_value'] < 0.05 else 'ns'}")
    print(f"  Effect size (r): {test_heads['effect_size_r']:.3f}")
    print(f"  Δ MAE: {test_heads['mean_delta_mae']:+.2f} bpm")
    if test_heads["significant"]:
        print(f"  → Shared head SIGNIFICANTLY outperforms sport-specific heads")
    else:
        print(f"  → Effect not statistically significant")

    # ────────────────────────────────────────────────────────────────────────
    # 4. OUTLIER INVESTIGATION
    # ────────────────────────────────────────────────────────────────────────
    outliers = find_outliers(df, rows, threshold_ae=30)

    print(f"\n{'='*70}")
    print("OUTLIER INSIGHTS")
    print(f"{'='*70}")
    worst_sport = outliers["sport"].value_counts()
    print(f"\nOutliers by sport:")
    for sp, count in worst_sport.items():
        pct = count / len(df[df["sport"] == sp]) * 100
        print(f"  {sp:<10}: {count} outliers ({pct:.1f}% of {sp} tests)")

    # Save all results
    print(f"\n{'='*70}")
    all_results = {
        "uncertainty_calibration": calibration_results,
        "ablation_tests": {
            "no_embedding": test_noem,
            "sport_heads": test_heads,
        },
        "outliers": outliers[["sport", "athlete_id", "fold", "lt_true_bpm", "lt_pred_bpm",
                              "error_bpm", "abs_error_bpm"]].to_dict(orient="records")[:10],
    }

    import json
    with open("post_hoc_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print("Results saved to post_hoc_results.json")
