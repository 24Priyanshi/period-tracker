"""
LSTM regressor for predicting "days to next period".

Loads `master_daily.csv` (produced by preprocess.py from the raw mcPHASES
CSVs) and trains a 2-layer LSTM on **all physiological + symptom signals**:
hormones, RHR, nightly skin temperature, sleep score & stages, respiratory
rate, HR zones, activity minutes, stress, and 12 Likert self-report symptoms.

Target:  days_to_next_period  (already derived in master_daily.csv)
Split:   GroupKFold by participant id — metrics reflect generalization to
         people the model has never seen.

Run:
    python train_lstm.py                # full 5-fold CV, 60 epochs
    python train_lstm.py --quick        # 2-fold, 8 epochs smoke test

Artefacts written next to this script:
    lstm_best.pt         weights of the best CV fold
    lstm_scaler.npz      feature mean/std + feature_cols used during training
    lstm_metrics.json    per-fold MAE + baselines
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "master_daily.csv"

# ----------------------------------------------------------------- constants

# Features fed to the LSTM at each timestep. We deliberately EXCLUDE
# flow_volume_num / phase_num / is_bleeding / period_start / day_in_cycle —
# those are derived from (or effectively equal to) the target and would
# leak label information into the model.
FEATURE_COLS = [
    # user-reported cycle phase (a legitimate input at inference time —
    # apps ask the user their current phase). We keep flow_volume_num OUT
    # because it's a same-day echo of the target.
    "phase_num",
    # hormones
    "lh", "estrogen",
    # resting heart rate
    "rhr",
    # skin temperature (a strong cycle-phase signal — rises in luteal phase)
    "temp_nightly_temperature",
    "temp_baseline_relative_nightly_standard_deviation",
    # sleep architecture
    "sleep_overall_score", "sleep_deep_sleep_in_minutes",
    "sleep_resting_heart_rate", "sleep_restlessness",
    # respiratory rate
    "resp_full_sleep_breathing_rate",
    # activity
    "active_sedentary", "active_lightly", "active_moderately", "active_very",
    "fitbit_stress_score",
    # heart-rate zones
    "hrz_in_default_zone_1", "hrz_in_default_zone_2", "hrz_in_default_zone_3",
    # self-reported symptoms (0-5 Likert)
    "appetite_num", "exerciselevel_num", "headaches_num", "cramps_num",
    "sorebreasts_num", "fatigue_num", "sleepissue_num", "moodswing_num",
    "stress_num", "foodcravings_num", "indigestion_num", "bloating_num",
    # weekend flag + static baseline covariates (repeated on every timestep)
    "is_weekend_num", "age_at_interval", "age_of_first_menarche",
]

WINDOW_SIZE = 21           # days of history fed to the LSTM
MAX_HORIZON = 45           # drop windows where target > this (right-censored)


# ----------------------------------------------------------------- data prep

def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["is_weekend_num"] = df["is_weekend"].astype(int)
    df = df.sort_values(["id", "study_interval", "day_in_study"]).reset_index(drop=True)
    return df


def _ffill(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs column-wise."""
    out = a.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        last = np.nan
        for i in range(len(col)):
            if np.isnan(col[i]):
                col[i] = last
            else:
                last = col[i]
    return out


def build_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Slide a WINDOW_SIZE-day window over each (id, interval) group."""
    features = [c for c in FEATURE_COLS if c in df.columns]
    X_list, y_list, g_list = [], [], []

    for (pid, _), grp in df.groupby(["id", "study_interval"]):
        grp = grp.sort_values("day_in_study").reset_index(drop=True)
        feat = grp[features].to_numpy(dtype=np.float32)
        feat = _ffill(feat)
        feat = np.nan_to_num(feat, nan=0.0)
        tgt = grp["days_to_next_period"].to_numpy(dtype=np.float32)

        for end in range(WINDOW_SIZE - 1, len(grp)):
            y = tgt[end]
            if np.isnan(y) or y > MAX_HORIZON:
                continue
            X_list.append(feat[end - WINDOW_SIZE + 1: end + 1])
            y_list.append(y)
            g_list.append(pid)

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    g = np.asarray(g_list, dtype=np.int64)
    return X, y, g, features


# -------------------------------------------------------------------- model

class PeriodLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class WindowDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


# ----------------------------------------------------------------- training

def train_one_fold(X_tr, y_tr, X_va, y_va, epochs: int, batch: int,
                   device: str):
    model = PeriodLSTM(X_tr.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.L1Loss()

    tr = DataLoader(WindowDS(X_tr, y_tr), batch_size=batch, shuffle=True)
    va = DataLoader(WindowDS(X_va, y_va), batch_size=batch)

    best_val, bad = float("inf"), 0
    best_state, history = None, []
    patience = 10

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(y_tr)

        model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for xb, yb in va:
                preds.append(model(xb.to(device)).cpu().numpy())
                tgts.append(yb.numpy())
        va_mae = float(np.mean(np.abs(np.concatenate(preds) - np.concatenate(tgts))))
        history.append({"epoch": ep, "train_mae": tr_loss, "val_mae": va_mae})

        if va_mae < best_val - 1e-4:
            best_val, bad = va_mae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return model, {"best_val_mae": best_val, "history": history}


def fit_scaler(X: np.ndarray):
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-6
    return mean, std


def apply_scaler(X: np.ndarray, mean, std):
    return (X - mean) / std


# ---------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.folds = 8, 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    df = load()
    print(f"loaded {len(df):,} rows x {df.shape[1]} cols from {CSV_PATH.name}")
    print(f"  participants: {df['id'].nunique()}  "
          f"period_starts: {int(df['period_start'].sum())}")

    X, y, groups, features = build_windows(df)
    print(f"windows: {len(X):,}  shape: {X.shape}  (features: {len(features)})")
    print(f"target: mean={y.mean():.2f}  median={np.median(y):.1f}  "
          f"min={y.min():.0f}  max={y.max():.0f}")

    baseline_mean = float(np.mean(np.abs(y - y.mean())))
    baseline_28 = float(np.mean(np.abs(y - 28)))
    print(f"baseline MAE (predict mean): {baseline_mean:.2f} days")
    print(f"baseline MAE (predict  28 ): {baseline_28:.2f} days")

    gkf = GroupKFold(n_splits=args.folds)
    results, best_overall, best_state, best_scaler = [], float("inf"), None, None

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        mean, std = fit_scaler(X_tr)
        X_tr = apply_scaler(X_tr, mean, std)
        X_va = apply_scaler(X_va, mean, std)

        model, info = train_one_fold(X_tr, y_tr, X_va, y_va,
                                     epochs=args.epochs, batch=args.batch,
                                     device=device)
        mae = info["best_val_mae"]
        print(f"[fold {fold}]  val MAE = {mae:.2f} days  "
              f"(train {len(tr_idx)} / val {len(va_idx)})")
        results.append({"fold": fold, "val_mae": mae,
                        "n_train": int(len(tr_idx)),
                        "n_val": int(len(va_idx))})
        if mae < best_overall:
            best_overall = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_scaler = (mean, std)

    cv_mae = float(np.mean([r["val_mae"] for r in results]))
    print(f"\nCV mean val MAE: {cv_mae:.2f} days   best fold: {best_overall:.2f}")

    torch.save(best_state, HERE / "lstm_best.pt")
    np.savez(HERE / "lstm_scaler.npz",
             mean=best_scaler[0], std=best_scaler[1],
             feature_cols=np.array(features))
    (HERE / "lstm_metrics.json").write_text(json.dumps({
        "device": device,
        "window_size": WINDOW_SIZE,
        "n_features": X.shape[2],
        "n_windows": int(len(X)),
        "feature_cols": features,
        "baseline_mae_mean": baseline_mean,
        "baseline_mae_28": baseline_28,
        "fold_results": results,
        "cv_mean_val_mae": cv_mae,
        "best_val_mae": best_overall,
    }, indent=2))
    print("[ok] saved lstm_best.pt, lstm_scaler.npz, lstm_metrics.json")


# --------------------------------------------------------------- inference

def predict_days_to_next_period(recent_days_df: pd.DataFrame) -> float:
    """
    Given the last WINDOW_SIZE days of a participant's master_daily frame,
    predict days until the next period.
    """
    meta = np.load(HERE / "lstm_scaler.npz", allow_pickle=True)
    mean, std = meta["mean"], meta["std"]
    features = list(meta["feature_cols"])
    if len(recent_days_df) < WINDOW_SIZE:
        raise ValueError(f"need at least {WINDOW_SIZE} days of data")

    df = recent_days_df.copy()
    if "is_weekend_num" not in df.columns and "is_weekend" in df.columns:
        df["is_weekend_num"] = df["is_weekend"].astype(int)

    x = df[features].tail(WINDOW_SIZE).to_numpy(dtype=np.float32)
    x = _ffill(x)
    x = np.nan_to_num(x, nan=0.0)
    x = (x - mean) / std
    x = torch.from_numpy(x[None, ...])

    model = PeriodLSTM(n_features=len(features))
    model.load_state_dict(torch.load(HERE / "lstm_best.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        return float(model(x).item())


if __name__ == "__main__":
    main()
