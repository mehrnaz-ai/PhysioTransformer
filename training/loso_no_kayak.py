

import os, sys, json, warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from loso_validation import (
    LactateDataset, collate_fn, PhysioTransformerFull,
    DEVICE, SEED, N_EPOCHS, BATCH_SIZE, LR, MAX_LR, WEIGHT_DECAY,
    loss_fn, evaluate_loso, IDX_TO_SPORT
)

torch.manual_seed(SEED)
np.random.seed(SEED)


try:
    with open("loso_results.json") as f:
        prev_loso = json.load(f)
except FileNotFoundError:
    prev_loso = {
        "running": {"mae": 8.95, "rmse": 11.99, "r2": 0.082, "n": 281},
        "cycling": {"mae": 7.61, "rmse": 10.50, "r2": 0.236, "n": 281},
        "rowing":  {"mae": 7.59, "rmse":  9.42, "r2": 0.243, "n": 209},
    }


if __name__ == "__main__":

    print("Loading dataset …")
    ds = LactateDataset("data")
    input_dim = ds[0][0].shape[1]

    
    kayak_sport_id = 3  
    no_kayak_idx = [
        i for i in range(len(ds))
        if int(ds[i][4]) != kayak_sport_id
    ]

    print(f"\nDataset with kayak:    {len(ds)} samples")
    print(f"Dataset without kayak: {len(no_kayak_idx)} samples")
    print(f"Kayak removed:         {len(ds) - len(no_kayak_idx)} samples\n")

    results_no_kayak = {}
    sports_to_test = ["running", "cycling", "rowing"]

    for held_out_sport in sports_to_test:
        print(f"\n{'='*70}")
        print(f"LOSO (no kayak): Hold-out {held_out_sport.upper()}")
        print(f"{'='*70}")

        sport_id_map = {"running": 0, "cycling": 1, "rowing": 2}
        held_id = sport_id_map[held_out_sport]

        # Split: held-out vs train (both without kayak)
        held_idx  = [i for i in no_kayak_idx if int(ds[i][4]) == held_id]
        train_idx = [i for i in no_kayak_idx if int(ds[i][4]) != held_id]

        print(f"  Train N = {len(train_idx)} (2 sports, no kayak)")
        print(f"  Test  N = {len(held_idx)}")

        train_loader = DataLoader(
            Subset(ds, train_idx), batch_size=BATCH_SIZE,
            shuffle=True, collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            Subset(ds, held_idx), batch_size=BATCH_SIZE,
            shuffle=False, collate_fn=collate_fn,
        )

        model = PhysioTransformerFull(input_dim).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=MAX_LR, epochs=N_EPOCHS,
            steps_per_epoch=len(train_loader),
        )

        best_mae = float("inf")

        for epoch in range(N_EPOCHS):
            model.train()
            for X, y, L, lt, hrmax, sport, _ in train_loader:
                X, y, lt, sport = X.to(DEVICE), y.to(DEVICE), lt.to(DEVICE), sport.to(DEVICE)
                opt.zero_grad()
                pred, lt_pred, logvar = model(X, L, sport)
                loss = loss_fn(pred, y, lt_pred, lt, logvar, L)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    scheduler.step()

            mae, rmse, r2 = evaluate_loso(model, test_loader)
            best_mae = min(best_mae, mae)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch:02d}: MAE {mae:.2f} bpm")

        results_no_kayak[held_out_sport] = {
            "mae": mae, "rmse": rmse, "r2": r2, "n": len(held_idx)
        }
        print(f"\n  Final: MAE={mae:.2f} bpm, RMSE={rmse:.2f} bpm, R²={r2:.3f}")

    
    print(f"\n{'='*70}")
    print("SENSITIVITY ANALYSIS: LOSO With vs Without Kayak in Training")
    print(f"{'='*70}\n")
    print(f"{'Sport':<12} {'Full LOSO MAE':>15} {'No-Kayak MAE':>14} {'Δ MAE':>10} {'Interpretation'}")
    print("─" * 75)

    for sport in sports_to_test:
        full_mae = prev_loso.get(sport, {}).get("mae", float("nan"))
        nk_mae   = results_no_kayak[sport]["mae"]
        delta    = nk_mae - full_mae
        if delta < -0.3:
            interp = "IMPROVED (kayak hurt transfer)"
        elif delta > 0.3:
            interp = "WORSE (kayak helped transfer)"
        else:
            interp = "NO CHANGE (structural shift)"
        print(f"  {sport:<10} {full_mae:>14.2f}   {nk_mae:>13.2f}   {delta:>+9.2f}   {interp}")

    print(f"\nIn-distribution (5-fold CV): 5.56 bpm (reference)\n")


    print(f"{'='*70}")
    print("SUGGESTED PAPER TEXT (add to Section 4.3 or Limitations):")
    print(f"{'='*70}\n")

    all_improved = all(
        results_no_kayak[s]["mae"] < prev_loso.get(s,{}).get("mae", 99) - 0.3
        for s in sports_to_test
    )
    none_changed = all(
        abs(results_no_kayak[s]["mae"] - prev_loso.get(s,{}).get("mae", 99)) < 0.3
        for s in sports_to_test
    )

    if all_improved:
        print(
            "A sensitivity analysis excluding kayak from training (retaining only running,\n"
            "cycling, and rowing) improved LOSO MAE for all three aerobic-endurance sports\n"
            "(running: {:.2f}→{:.2f} bpm; cycling: {:.2f}→{:.2f} bpm; rowing: {:.2f}→{:.2f} bpm),\n"
            "suggesting that the small, domain-distant kayak sample introduced noise into\n"
            "the shared encoder. Cross-sport transfer among aerobic-endurance disciplines\n"
            "may be substantially more feasible with balanced and physiologically similar\n"
            "training data.".format(
                prev_loso["running"]["mae"], results_no_kayak["running"]["mae"],
                prev_loso["cycling"]["mae"], results_no_kayak["cycling"]["mae"],
                prev_loso["rowing"]["mae"],  results_no_kayak["rowing"]["mae"],
            )
        )
    elif none_changed:
        print(
            "A sensitivity analysis excluding kayak from training (n=3 sports) produced\n"
            "comparable LOSO results for running, cycling, and rowing (ΔMAE < 0.3 bpm\n"
            "for all sports), indicating that domain shift reflects structural\n"
            "physiological differences between sports rather than sample imbalance."
        )
    else:
        for sport in sports_to_test:
            full = prev_loso.get(sport, {}).get("mae", float("nan"))
            nk   = results_no_kayak[sport]["mae"]
            print(f"  {sport}: {full:.2f} → {nk:.2f} bpm (Δ={nk-full:+.2f})")

    
    output = {
        "note": "LOSO sensitivity analysis — kayak excluded from training",
        "in_distribution_mae": 5.56,
        "full_loso": prev_loso,
        "no_kayak_loso": results_no_kayak,
    }
    with open("loso_no_kayak_results.json", "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved → loso_no_kayak_results.json")
