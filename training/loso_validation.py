"""
loso_validation.py — Leave-One-Sport-Out Cross-Sport Generalization

Train on 3 sports, test on held-out sport. Repeat for each sport.
Critical for validating "multi-sport" claims.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from model_full import (
    LactateDataset, collate_fn, PhysioTransformerFull,
    DEVICE, SEED, N_EPOCHS, BATCH_SIZE, LR, MAX_LR, WEIGHT_DECAY,
    loss_fn, IDX_TO_SPORT
)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# LOSO EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_loso(model, loader):
    """Evaluate on LOSO test set."""
    model.eval()
    lt_p, lt_t = [], []

    for X, y, L, lt, hrmax, sport, _ in loader:
        X = X.to(DEVICE)
        sport = sport.to(DEVICE)
        hrmax_dev = hrmax.to(DEVICE)

        curve, lt_pred, logvar = model(X, L, sport)

        lt_true_bpm = (lt.to(DEVICE) * hrmax_dev).view(-1)
        lt_pred_bpm = (lt_pred * hrmax_dev).view(-1)

        ok_lt = torch.isfinite(lt_pred_bpm) & torch.isfinite(lt_true_bpm)
        lt_p.append(lt_pred_bpm[ok_lt].cpu())
        lt_t.append(lt_true_bpm[ok_lt].cpu())

    lt_p = torch.cat(lt_p)
    lt_t = torch.cat(lt_t)

    mae  = (lt_p - lt_t).abs().mean().item()
    rmse = ((lt_p - lt_t) ** 2).mean().sqrt().item()

    ss_res = ((lt_t - lt_p) ** 2).sum()
    ss_tot = ((lt_t - lt_t.mean()) ** 2).sum()
    r2     = (1.0 - ss_res / (ss_tot + 1e-8)).item()

    return mae, rmse, r2, len(lt_p)


def run_loso(ds, groups):
    """Run LOSO for each sport."""
    results = {}
    sports_list = ["running", "cycling", "rowing", "kayak"]

    for held_out_sport in sports_list:
        print(f"\n{'='*70}")
        print(f"LOSO: Hold-out {held_out_sport.upper()}")
        print(f"{'='*70}")

        # Get indices for held-out sport and others
        held_out_idx = [
            i for i in range(len(ds))
            if IDX_TO_SPORT.get(int(ds[i][4]), "unknown") == held_out_sport
        ]
        train_idx = [
            i for i in range(len(ds))
            if i not in held_out_idx
        ]

        if len(held_out_idx) == 0:
            print(f"  SKIP: No samples for {held_out_sport}")
            continue

        print(f"  Train N = {len(train_idx)}")
        print(f"  Test  N = {len(held_out_idx)}")

        train_loader = DataLoader(
            Subset(ds, train_idx), batch_size=BATCH_SIZE,
            shuffle=True, collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            Subset(ds, held_out_idx), batch_size=BATCH_SIZE,
            shuffle=False, collate_fn=collate_fn,
        )

        model = PhysioTransformerFull(ds[0][0].shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(),
                                lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=MAX_LR, epochs=N_EPOCHS,
            steps_per_epoch=len(train_loader),
        )

        best_val_mae = float("inf")
        best_epoch = 0

        for epoch in range(N_EPOCHS):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X, y, L, lt, hrmax, sport, _ in train_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                lt = lt.to(DEVICE)
                sport = sport.to(DEVICE)

                opt.zero_grad()
                pred, lt_pred, logvar = model(X, L, sport)
                loss = loss_fn(pred, y, lt_pred, lt, logvar, L)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()
                epoch_loss += loss.item()
                n_batches += 1

            mae, _, _, _ = evaluate_loso(model, test_loader)
            if mae < best_val_mae:
                best_val_mae = mae
                best_epoch = epoch

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch:02d}: Test MAE {mae:.2f} bpm")

        # Final evaluation
        mae, rmse, r2, n = evaluate_loso(model, test_loader)

        results[held_out_sport] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "n": n,
            "best_epoch": best_epoch,
        }

        print(f"\n  Final: MAE={mae:.2f} bpm, RMSE={rmse:.2f} bpm, R²={r2:.3f}")

    return results


if __name__ == "__main__":
    print("Loading dataset …")
    ds = LactateDataset("data")
    groups = [ds[i][-1] for i in range(len(ds))]

    print("Running LOSO validation …")
    loso_results = run_loso(ds, groups)

    print(f"\n{'='*70}")
    print("LOSO RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Sport':<15} {'N':<8} {'MAE (bpm)':<12} {'RMSE (bpm)':<12} {'R²':<10}")
    print("─" * 70)

    in_dist_mae = 5.56  # from full model 5-fold CV
    for sport in ["running", "cycling", "rowing", "kayak"]:
        if sport in loso_results:
            r = loso_results[sport]
            delta = f"+{r['mae'] - in_dist_mae:.2f}" if r['mae'] > in_dist_mae else f"{r['mae'] - in_dist_mae:.2f}"
            print(f"  {sport:<13} {r['n']:<8} {r['mae']:>10.2f}  {r['rmse']:>10.2f}  {r['r2']:>8.3f}  "
                  f"({delta} vs in-dist)")

    print(f"\n  In-distribution (5-fold CV): MAE = {in_dist_mae:.2f} bpm")
    print(f"\n{'='*70}")

    # Save results
    import json
    with open("loso_results.json", "w") as f:
        json.dump(loso_results, f, indent=2)
    print("Results saved to loso_results.json")
