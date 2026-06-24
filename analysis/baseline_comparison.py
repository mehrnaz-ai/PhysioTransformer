"""
baseline_comparison.py — Classical and Recurrent Baselines

Train LR, RF, XGBoost, LSTM on the same 5-fold GroupKFold splits.
Compare to PhysioTransformerFull.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from model_full import (
    LactateDataset, collate_fn, PhysioTransformerFull,
    DEVICE, SEED, N_EPOCHS, BATCH_SIZE, LR, MAX_LR, WEIGHT_DECAY,
    loss_fn, IDX_TO_SPORT,
)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION FOR CLASSICAL MODELS
# ══════════════════════════════════════════════════════════════════════════════

def extract_global_features(X_seq, L):
    """
    Convert variable-length sequence to fixed-length feature vector.
    X_seq: (T, 14)
    L: int, valid sequence length
    Returns: (56,) feature vector
    """
    X_valid = X_seq[:L]  # Trim to valid length
    feats = []

    for feat_idx in range(X_seq.shape[-1]):  # 14 features
        col = X_valid[:, feat_idx]
        feats.extend([
            col.mean(),
            col.std() if len(col) > 1 else 0.0,
            col.max(),
            col[-1] - col[0],  # delta
        ])
    return np.array(feats, dtype=np.float32)


def prepare_classical_data(ds, train_idx, test_idx):
    """Prepare feature matrices for sklearn models."""
    X_train, y_train = [], []
    X_test, y_test = [], []

    for i in train_idx:
        X, y, lt, hrmax, sport, _ = ds[i]
        L = X.shape[0]
        X_feat = extract_global_features(X.numpy(), L)
        X_train.append(X_feat)
        y_train.append(float(lt) * float(hrmax))  # LT in bpm

    for i in test_idx:
        X, y, lt, hrmax, sport, _ = ds[i]
        L = X.shape[0]
        X_feat = extract_global_features(X.numpy(), L)
        X_test.append(X_feat)
        y_test.append(float(lt) * float(hrmax))

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


# ══════════════════════════════════════════════════════════════════════════════
# LSTM BASELINE
# ══════════════════════════════════════════════════════════════════════════════

class LSTMBaseline(torch.nn.Module):
    def __init__(self, input_dim, hidden=128, n_layers=2, dropout=0.15):
        super().__init__()
        self.sport_emb = torch.nn.Embedding(5, 16)
        self.lstm = torch.nn.LSTM(
            input_dim + 16, hidden, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0
        )
        self.curve_head = torch.nn.Linear(hidden, 1)
        self.attn = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1),
        )
        self.lt_bins = torch.nn.Sequential(
            torch.nn.Linear(hidden, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 10),
        )
        self.register_buffer(
            "lt_bin_centers",
            torch.linspace(0.4, 1.0, 10),
        )
        self.logvar = torch.nn.Sequential(
            torch.nn.Linear(hidden, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x, L, sport):
        B, T, _ = x.shape
        emb = self.sport_emb(sport).unsqueeze(1).repeat(1, T, 1)
        x = torch.cat([x, emb], dim=-1)

        out, _ = self.lstm(x)
        curve = self.curve_head(out)

        mask = torch.arange(T, device=x.device)[None, :] >= L[:, None]
        score = self.attn(out).masked_fill(mask.unsqueeze(-1), -1e9)
        w = torch.softmax(score, dim=1)
        h_global = (w * out).sum(dim=1)

        lt_logits = self.lt_bins(h_global)
        lt_prob = torch.softmax(lt_logits, dim=-1)
        lt_pred = (lt_prob * self.lt_bin_centers).sum(dim=-1)
        logvar = self.logvar(h_global).clamp(-5.0, 2.0)

        return curve, lt_pred, logvar


@torch.no_grad()
def evaluate_lstm(model, loader):
    model.eval()
    lt_p, lt_t = [], []

    for X, y, L, lt, hrmax, sport, _ in loader:
        X, sport = X.to(DEVICE), sport.to(DEVICE)
        hrmax_dev = hrmax.to(DEVICE)

        curve, lt_pred, logvar = model(X, L, sport)

        lt_true_bpm = (lt.to(DEVICE) * hrmax_dev).view(-1)
        lt_pred_bpm = (lt_pred * hrmax_dev).view(-1)

        ok_lt = torch.isfinite(lt_pred_bpm) & torch.isfinite(lt_true_bpm)
        lt_p.append(lt_pred_bpm[ok_lt].cpu())
        lt_t.append(lt_true_bpm[ok_lt].cpu())

    lt_p = torch.cat(lt_p)
    lt_t = torch.cat(lt_t)

    mae = (lt_p - lt_t).abs().mean().item()
    rmse = ((lt_p - lt_t) ** 2).mean().sqrt().item()
    r2 = (1.0 - ((lt_t - lt_p) ** 2).sum() 
          / (((lt_t - lt_t.mean()) ** 2).sum() + 1e-8)).item()

    return mae, rmse, r2


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BASELINE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    ds = LactateDataset("data")
    groups = [ds[i][-1] for i in range(len(ds))]
    gkf = GroupKFold(n_splits=5)
    input_dim = ds[0][0].shape[1]

    results = {
        "Linear Regression": [],
        "Random Forest": [],
        "XGBoost": [],
        "LSTM": [],
        "Transformer": [],
    }

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(np.arange(len(ds)), groups=groups)
    ):
        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1} — BASELINE COMPARISON")
        print(f"{'='*70}")
        print(f"Train N={len(train_idx)}, Test N={len(test_idx)}\n")

        # Prepare data
        X_train_clf, y_train_clf, X_test_clf, y_test_clf = prepare_classical_data(
            ds, train_idx, test_idx
        )

        # ────────────────────────────────────────────────────────────────────
        # 1. LINEAR REGRESSION
        # ────────────────────────────────────────────────────────────────────
        lr = LinearRegression()
        lr.fit(X_train_clf, y_train_clf)
        y_pred_lr = lr.predict(X_test_clf)
        mae_lr = np.abs(y_pred_lr - y_test_clf).mean()
        rmse_lr = np.sqrt(((y_pred_lr - y_test_clf) ** 2).mean())
        r2_lr = 1.0 - ((y_test_clf - y_pred_lr) ** 2).sum() / \
                      ((y_test_clf - y_test_clf.mean()) ** 2).sum()
        print(f"Linear Regression:  MAE={mae_lr:.2f} bpm, R²={r2_lr:.3f}")
        results["Linear Regression"].append({"mae": mae_lr, "rmse": rmse_lr, "r2": r2_lr})

        # ────────────────────────────────────────────────────────────────────
        # 2. RANDOM FOREST
        # ────────────────────────────────────────────────────────────────────
        rf = RandomForestRegressor(n_estimators=100, max_depth=15,
                                    random_state=SEED, n_jobs=-1)
        rf.fit(X_train_clf, y_train_clf)
        y_pred_rf = rf.predict(X_test_clf)
        mae_rf = np.abs(y_pred_rf - y_test_clf).mean()
        rmse_rf = np.sqrt(((y_pred_rf - y_test_clf) ** 2).mean())
        r2_rf = 1.0 - ((y_test_clf - y_pred_rf) ** 2).sum() / \
                      ((y_test_clf - y_test_clf.mean()) ** 2).sum()
        print(f"Random Forest:      MAE={mae_rf:.2f} bpm, R²={r2_rf:.3f}")
        results["Random Forest"].append({"mae": mae_rf, "rmse": rmse_rf, "r2": r2_rf})

        # ────────────────────────────────────────────────────────────────────
        # 3. XGBOOST
        # ────────────────────────────────────────────────────────────────────
        xgb = XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=100,
                           random_state=SEED, n_jobs=-1, verbosity=0)
        xgb.fit(X_train_clf, y_train_clf)
        y_pred_xgb = xgb.predict(X_test_clf)
        mae_xgb = np.abs(y_pred_xgb - y_test_clf).mean()
        rmse_xgb = np.sqrt(((y_pred_xgb - y_test_clf) ** 2).mean())
        r2_xgb = 1.0 - ((y_test_clf - y_pred_xgb) ** 2).sum() / \
                       ((y_test_clf - y_test_clf.mean()) ** 2).sum()
        print(f"XGBoost:            MAE={mae_xgb:.2f} bpm, R²={r2_xgb:.3f}")
        results["XGBoost"].append({"mae": mae_xgb, "rmse": rmse_xgb, "r2": r2_xgb})

        # ────────────────────────────────────────────────────────────────────
        # 4. LSTM
        # ────────────────────────────────────────────────────────────────────
        train_loader = DataLoader(
            Subset(ds, train_idx), batch_size=BATCH_SIZE,
            shuffle=True, collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            Subset(ds, test_idx), batch_size=BATCH_SIZE,
            shuffle=False, collate_fn=collate_fn,
        )

        lstm_model = LSTMBaseline(input_dim).to(DEVICE)
        opt = torch.optim.AdamW(lstm_model.parameters(),
                                lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=MAX_LR, epochs=N_EPOCHS,
            steps_per_epoch=len(train_loader),
        )

        best_mae_lstm = float("inf")
        for epoch in range(N_EPOCHS):
            lstm_model.train()
            for X, y, L, lt, hrmax, sport, _ in train_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                lt = lt.to(DEVICE)
                sport = sport.to(DEVICE)

                opt.zero_grad()
                pred, lt_pred, logvar = lstm_model(X, L, sport)
                loss = loss_fn(pred, y, lt_pred, lt, logvar, L)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
                    opt.step()
                    scheduler.step()

            mae_lstm, rmse_lstm, r2_lstm = evaluate_lstm(lstm_model, test_loader)
            best_mae_lstm = min(best_mae_lstm, mae_lstm)

        print(f"LSTM:               MAE={mae_lstm:.2f} bpm, R²={r2_lstm:.3f}")
        results["LSTM"].append({"mae": mae_lstm, "rmse": rmse_lstm, "r2": r2_lstm})

        # ────────────────────────────────────────────────────────────────────
        # 5. TRANSFORMER (LOAD CHECKPOINT)
        # ────────────────────────────────────────────────────────────────────
        ckpt_path = f"checkpoint_full_fold{fold + 1}.pt"
        if os.path.exists(ckpt_path):
            transformer_model = PhysioTransformerFull(input_dim).to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            transformer_model.load_state_dict(ckpt["state_dict"])
            mae_tf, rmse_tf, r2_tf = evaluate_lstm(transformer_model, test_loader)
            print(f"Transformer:        MAE={mae_tf:.2f} bpm, R²={r2_tf:.3f}")
            results["Transformer"].append({"mae": mae_tf, "rmse": rmse_tf, "r2": r2_tf})
        else:
            print(f"Transformer:        Checkpoint not found")

    # ────────────────────────────────────────────────────────────────────────
    # SUMMARY TABLE
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("BASELINE COMPARISON SUMMARY")
    print(f"{'='*70}\n")

    summary = []
    for model_name, fold_results in results.items():
        if len(fold_results) == 0:
            continue
        maes = np.array([r["mae"] for r in fold_results])
        rmses = np.array([r["rmse"] for r in fold_results])
        r2s = np.array([r["r2"] for r in fold_results])

        summary.append({
            "Model": model_name,
            "MAE (bpm)": f"{maes.mean():.2f} ± {maes.std():.2f}",
            "RMSE (bpm)": f"{rmses.mean():.2f} ± {rmses.std():.2f}",
            "R²": f"{r2s.mean():.3f} ± {r2s.std():.3f}",
        })

    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))

    # Save results
    import json
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to baseline_results.json")

    # Statistical significance: Transformer vs LSTM
    if "LSTM" in results and "Transformer" in results:
        lstm_maes = np.array([r["mae"] for r in results["LSTM"]])
        tf_maes = np.array([r["mae"] for r in results["Transformer"]])
        from scipy import stats
        stat, p = stats.wilcoxon(lstm_maes, tf_maes, alternative="greater")
        print(f"\n{'─'*70}")
        print(f"Transformer vs LSTM (Wilcoxon signed-rank test):")
        print(f"  Transformer MAE: {tf_maes.mean():.2f} ± {tf_maes.std():.2f} bpm")
        print(f"  LSTM MAE:        {lstm_maes.mean():.2f} ± {lstm_maes.std():.2f} bpm")
        print(f"  p-value: {p:.4f}")
        if p < 0.05:
            print(f"  → Transformer significantly outperforms LSTM")
        else:
            print(f"  → No significant difference")
