

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")


# CONFIG

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

LT_THRESHOLD = 2.0
N_EPOCHS     = 51
BATCH_SIZE   = 32
LR           = 3e-4
MAX_LR       = 1e-3
WEIGHT_DECAY = 1e-4


# SPORT MAP

SPORT_TO_IDX = {
    "running": 0, "cycling": 1, "rowing": 2, "kayak": 3, "unknown": 4
}
IDX_TO_SPORT = {v: k for k, v in SPORT_TO_IDX.items()}

SPORT_POWER_MAX = {
    "running": 22.0, "cycling": 450.0, "rowing": 500.0, "kayak": 300.0
}


# HELPERS

def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(str(x).replace(",", ".").rstrip("."))
    except Exception:
        return np.nan


def clean(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def extract_extra(sheet3):
    hrmax, hr_2mmol = None, None
    if sheet3.shape[1] < 2:
        return hrmax, hr_2mmol
    for i in range(len(sheet3)):
        k = str(sheet3.iloc[i, 0]).lower()
        v = safe_float(sheet3.iloc[i, 1])
        if "hrmax" in k:
            hrmax = v
        if "hr @ 2mmol" in k:
            hr_2mmol = v
    return hrmax, hr_2mmol



# DATASET  (identical to model_full.py)

class LactateDataset(Dataset):

    def __init__(self, root):
        self.files = [
            os.path.join(root, f)
            for f in sorted(os.listdir(root))
            if f.endswith(".xlsx")
        ]
        self.samples = []
        for f in self.files:
            try:
                s = self.load_file(f)
                if s is not None:
                    self.samples.append(s)
            except Exception as e:
                print(f"  SKIP {os.path.basename(f)}: {e}")
        print(f"{'='*60}")
        print(f"FINAL VALID SAMPLES: {len(self.samples)}")
        print(f"{'='*60}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def interpolate_lt(self, hr, lac, threshold=2.0):
        cross = np.where(lac >= threshold)[0]
        if len(cross) == 0:
            return float(hr[-1])
        i = cross[0]
        if i == 0:
            return float(hr[0])
        x1, x2 = hr[i - 1], hr[i]
        y1, y2 = lac[i - 1], lac[i]
        return float(x1 + (threshold - y1) * (x2 - x1) / (y2 - y1 + 1e-8))

    def load_file(self, file_path):
        sheets = pd.read_excel(file_path, sheet_name=None)
        if len(sheets) < 3:
            return None
        keys = list(sheets.keys())
        s1, s2, s3 = sheets[keys[0]], sheets[keys[1]], sheets[keys[2]]

        meta = {
            str(s1.iloc[i, 0]).lower(): s1.iloc[i, 1]
            for i in range(len(s1))
        }

        sport    = str(meta.get("test:", "unknown")).lower().strip()
        sport    = {"runnin": "running", "c2": "cycling"}.get(sport, sport)
        sport_id = SPORT_TO_IDX.get(sport, SPORT_TO_IDX["unknown"])

        fname      = Path(file_path).stem
        m          = re.match(r"(A\d+)", fname)
        athlete_id = m.group(1) if m else fname

        h      = float(meta.get("height", 0) or 0) / 200.0   # [FIX F1]
        w      = float(meta.get("weight", 0) or 0) / 120.0
        bmi    = float(meta.get("bmi",    0) or 0) / 40.0
        g_raw  = str(meta.get("gender:", "")).lower()
        gender = 1.0 if g_raw == "m" else 0.0 if g_raw == "f" else 0.5

        dob  = pd.to_datetime(meta.get("dob:",       None), errors="coerce", dayfirst=True)
        test = pd.to_datetime(meta.get("test date:", None), errors="coerce", dayfirst=True)
        age  = ((test - dob).days / 365.25
                if pd.notna(dob) and pd.notna(test) else 25.0)
        age  = np.clip(age, 10, 80) / 80.0

        hrmax_ref, hr_2mmol = extract_extra(s3)

        s2.columns = [str(c).lower() for c in s2.columns]
        if "hr" not in s2.columns or "bla" not in s2.columns:
            return None

        hr  = s2["hr"].apply(safe_float).values
        lac = s2["bla"].apply(safe_float).values
        ok  = ~np.isnan(hr) & ~np.isnan(lac)
        hr, lac = hr[ok], lac[ok]
        if len(hr) < 5:
            return None

        lac = pd.Series(lac).rolling(3, center=True, min_periods=1).mean().values

        # [FIX F2] hrmax computed once
        if hrmax_ref is not None and not np.isnan(hrmax_ref) and hrmax_ref >= 100:
            hrmax = float(hrmax_ref)
        else:
            hrmax = float(np.max(hr))
        if hrmax > 240:
            return None

        power = np.zeros(len(hr))
        for col in s2.columns:
            if "actual power" in col.lower().strip():
                raw   = s2[col].apply(safe_float).values
                denom = SPORT_POWER_MAX.get(sport, 1.0)
                power = raw / denom
                break

        if hr_2mmol is not None and not np.isnan(hr_2mmol) and hr_2mmol > 0:
            lt_hr = float(hr_2mmol)
        else:
            lt_hr = self.interpolate_lt(hr, lac, threshold=LT_THRESHOLD)
        lt_norm = float(np.clip(lt_hr / (hrmax + 1e-6), 0.40, 1.00))

        # [FIX F3, F4] single clean feature block
        hr_norm    = hr / (hrmax + 1e-6)
        hr_rest    = float(np.min(hr))
        hr_reserve = np.clip((hr - hr_rest) / (hrmax - hr_rest + 1e-6), 0.0, 1.0)

        power     = clean(power)
        t         = np.linspace(0.0, 1.0, len(hr))
        hr_slope  = np.gradient(hr_norm)
        hr_accel  = np.gradient(hr_slope)
        hr_ma     = pd.Series(hr_norm).rolling(3, min_periods=1).mean().values
        hr_std    = pd.Series(hr_norm).rolling(3, min_periods=1).std().fillna(0.0).values
        vo2_proxy = hr_reserve * power
        fatigue   = np.cumsum(hr_reserve) / len(hr)
        static    = np.tile([h, w, bmi, gender, age], (len(hr), 1))

        X = clean(np.column_stack([
            hr_reserve, power, hr_slope, hr_accel,
            hr_ma, hr_std, vo2_proxy, fatigue, t, static
        ]))
        y = clean(np.log1p(lac).reshape(-1, 1))

        return (
            torch.tensor(X,        dtype=torch.float32),
            torch.tensor(y,        dtype=torch.float32),
            torch.tensor(lt_norm,  dtype=torch.float32),
            torch.tensor(hrmax,    dtype=torch.float32),
            torch.tensor(sport_id, dtype=torch.long),
            athlete_id,
        )



# COLLATE

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    X, y, lt, hrmax, sport, athlete = zip(*batch)
    L = torch.tensor([x.size(0) for x in X])
    return (
        pad_sequence(X,    batch_first=True),
        pad_sequence(y,    batch_first=True),
        L,
        torch.stack(list(lt)),
        torch.stack(list(hrmax)),
        torch.stack(list(sport)),
        athlete,
    )



# POSITIONAL ENCODING

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]





class PhysioTransformerSportHeads(nn.Module):
    def __init__(
        self,
        input_dim:      int,
        n_sports:       int = 5,
        d_model:        int = 128,    # [FIX B2]
        n_heads:        int = 8,
        n_layers:       int = 4,
        dim_ff:         int = 1024,
        dropout:        float = 0.15,
        sport_emb_dim:  int = 16,
        n_lt_bins:      int = 10,
    ):
        super().__init__()
        self.n_sports    = n_sports
        self.n_lt_bins   = n_lt_bins

        # Sport embedding — identical to model_full.py
        self.sport_emb = nn.Embedding(n_sports, sport_emb_dim)

        # Input projection — identical to model_full.py
        self.inp = nn.Linear(input_dim + sport_emb_dim, d_model)
        self.pos = PositionalEncoding(d_model)

        # Transformer encoder — identical to model_full.py
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Shared representation — identical to model_full.py
        self.repr = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 64),
            nn.GELU(),
        )

        # Attention pooling — identical to model_full.py
        self.attn_pool = nn.Sequential(
            nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
        )

        # Curve head — identical to model_full.py
        self.curve_head = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1)
        )

        # Per-sport LT heads — one ordinal head per sport label
        #          running / cycling / rowing / kayak / unknown → 5 heads
        self.lt_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, n_lt_bins),
            )
            for _ in range(n_sports)
        ])
        self.register_buffer(
            "lt_bin_centers",
            torch.linspace(0.4, 1.0, n_lt_bins),
        )

        # Uncertainty logvar — identical to model_full.py
        self.logvar = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1)
        )

    # ----------------------------------------------------------
    def forward(self, x, L, sport):
        B, T, _ = x.shape


        emb = self.sport_emb(sport).unsqueeze(1).expand(-1, T, -1)
        x   = torch.cat([x, emb], dim=-1)

        z    = self.pos(self.inp(x))
        mask = torch.arange(T, device=x.device)[None, :] >= L[:, None]
        h    = self.encoder(z, src_key_padding_mask=mask)
        h    = self.repr(h)

        score    = self.attn_pool(h).masked_fill(mask.unsqueeze(-1), -1e9)
        w        = torch.softmax(score, dim=1)
        h_global = (w * h).sum(dim=1)

        curve = self.curve_head(h)


        lt_logits_list = []
        for i in range(B):
            sport_i   = sport[i].item()
            head      = self.lt_heads[sport_i]
            lt_logits_list.append(head(h_global[i : i + 1]))
        lt_logits = torch.cat(lt_logits_list, dim=0)        # (B, n_lt_bins)

        lt_prob = torch.softmax(lt_logits, dim=-1)
        lt_pred = (lt_prob * self.lt_bin_centers).sum(dim=-1)

        logvar = self.logvar(h_global).clamp(-5.0, 2.0)

        return curve, lt_pred, logvar



# LOSS  (identical to model_full.py)

def loss_fn(curve, y, lt_pred, lt_true, logvar, L):
    T    = curve.size(1)
    mask = (torch.arange(T, device=curve.device)[None, :] < L[:, None])
    mask = mask.float().unsqueeze(-1)
    curve_loss = ((curve - y) ** 2 * mask).sum() / mask.sum()

    var    = torch.exp(logvar.squeeze(-1)).clamp(min=1e-6)
    lt_nll = (
        0.5 * torch.log(var)
        + (lt_pred - lt_true) ** 2 / (2.0 * var)
    ).mean()

    return curve_loss + 0.5 * lt_nll



# EVALUATE  (identical to model_full.py)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    curve_P, curve_T = [], []
    lt_p, lt_t = [], []

    for X, y, L, lt, hrmax, sport, _ in loader:
        X, y      = X.to(DEVICE), y.to(DEVICE)
        sport     = sport.to(DEVICE)
        hrmax_dev = hrmax.to(DEVICE)

        curve, lt_pred, logvar = model(X, L, sport)

        lt_true_bpm = (lt.to(DEVICE) * hrmax_dev).view(-1)
        lt_pred_bpm = (lt_pred        * hrmax_dev).view(-1)

        for i in range(X.size(0)):
            Li = L[i].item()
            p  = torch.expm1(curve[i, :Li, 0].clamp(-3.0, 4.0))
            t  = torch.expm1(y[i,    :Li, 0].clamp(-3.0, 4.0))
            ok = torch.isfinite(p) & torch.isfinite(t)
            if ok.sum() > 0:
                curve_P.append(p[ok].cpu())
                curve_T.append(t[ok].cpu())

        ok_lt = torch.isfinite(lt_pred_bpm) & torch.isfinite(lt_true_bpm)
        lt_p.append(lt_pred_bpm[ok_lt].cpu())
        lt_t.append(lt_true_bpm[ok_lt].cpu())

    curve_P = torch.cat(curve_P);  curve_T = torch.cat(curve_T)
    lt_p    = torch.cat(lt_p);     lt_t    = torch.cat(lt_t)

    r2_curve = (1.0 - ((curve_T - curve_P)**2).sum()
                / (((curve_T - curve_T.mean())**2).sum() + 1e-8)).item()
    mae  = (lt_p - lt_t).abs().mean().item()
    rmse = ((lt_p - lt_t)**2).mean().sqrt().item()
    r2_lt = (1.0 - ((lt_t - lt_p)**2).sum()
             / (((lt_t - lt_t.mean())**2).sum() + 1e-8)).item()

    return r2_curve, mae, rmse, r2_lt



# MAIN

if __name__ == "__main__":
    ds        = LactateDataset("data")
    groups    = [ds[i][-1] for i in range(len(ds))]
    gkf       = GroupKFold(n_splits=5)
    input_dim = ds[0][0].shape[1]      # 14 features

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(np.arange(len(ds)), groups=groups)
    ):
        print(f"\n{'='*60}\nFOLD {fold + 1}  |  "
              f"Train N={len(train_idx)}  Test N={len(test_idx)}\n{'='*60}")

        train_loader = DataLoader(
            Subset(ds, train_idx), batch_size=BATCH_SIZE,
            shuffle=True, collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            Subset(ds, test_idx), batch_size=BATCH_SIZE,
            shuffle=False, collate_fn=collate_fn,
        )

        model = PhysioTransformerSportHeads(input_dim).to(DEVICE)
        opt   = torch.optim.AdamW(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=MAX_LR, epochs=N_EPOCHS,
            steps_per_epoch=len(train_loader),
        )

        best_val_mae = float("inf")

        for epoch in range(N_EPOCHS):
            model.train()
            epoch_loss = 0.0
            n_batches  = 0

            for X, y, L, lt, hrmax, sport, _ in train_loader:
                X, y   = X.to(DEVICE), y.to(DEVICE)
                lt     = lt.to(DEVICE)
                sport  = sport.to(DEVICE)

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
                n_batches  += 1

            r2, val_mae, rmse, r2lt = evaluate(model, test_loader)
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch:02d} | Loss {avg_loss:.4f} | "
                  f"Val MAE {val_mae:.2f} bpm | Val R² {r2lt:.3f}")

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(
                    {"state_dict": model.state_dict(),
                     "fold": fold + 1, "epoch": epoch,
                     "val_mae": val_mae},
                    f"checkpoint_sport_heads_fold{fold + 1}.pt",
                )
                print(f"  ✓  Best checkpoint saved  (MAE={val_mae:.2f})")

        ckpt = torch.load(f"checkpoint_sport_heads_fold{fold + 1}.pt",
                          map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        r2, mae, rmse, r2lt = evaluate(model, test_loader)

        print(f"\n  Fold {fold + 1}  best epoch={ckpt['epoch']}")
        print(f"  Curve R²  = {r2:.3f}")
        print(f"  LT MAE    = {mae:.2f} bpm")
        print(f"  LT RMSE   = {rmse:.2f} bpm")
        print(f"  LT R²     = {r2lt:.3f}")
        fold_results.append([r2, mae, rmse, r2lt])

    R = np.array(fold_results)
    print(f"\n{'='*60}")
    print("5-FOLD GROUPKFOLD RESULTS  —  Ablation F: Sport-Specific LT Heads")
    print(f"{'='*60}")
    print(f"Curve R²  :  {R[:,0].mean():.3f} ± {R[:,0].std():.3f}")
    print(f"LT MAE    :  {R[:,1].mean():.2f} ± {R[:,1].std():.2f} bpm")
    print(f"LT RMSE   :  {R[:,2].mean():.2f} ± {R[:,2].std():.2f} bpm")
    print(f"LT R²     :  {R[:,3].mean():.3f} ± {R[:,3].std():.3f}")
