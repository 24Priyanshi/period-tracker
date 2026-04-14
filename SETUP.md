# Setup & Reproduction Guide

How to run the menstrual-period prediction pipeline — both **locally** and on **Google Colab**.

## What's in this repo

| File | Purpose |
|------|---------|
| **`master_daily.csv`** | **Main training dataset** — per-(id, day) table with 54 columns: hormones, RHR, nightly temperature, sleep, respiratory rate, HR zones, activity, stress, 12 symptoms, baselines. Derived from the raw mcPHASES CSVs by `preprocess.py`. |
| `cycles.csv` | Per-cycle table (cycle_length, period_duration) for cycle-level models |
| `processed_period_data.csv` | User's earlier merged file (24 cols) — kept for reference / EDA |
| `preprocess.py` | Builds `master_daily.csv` + `cycles.csv` from the raw mcPHASES CSVs (not included — download separately if you want to re-run this) |
| `eda.py` | Produces 10 diagnostic plots → `figures/*.png` (runs on `processed_period_data.csv`) |
| `train_lstm.py` | Trains the LSTM on **`master_daily.csv`** with 34 features + saves model artefacts |
| `colab_train_lstm.ipynb` | Self-contained notebook — runs end-to-end on Google Colab |
| `requirements.txt` | Python dependencies |
| `APPROACH.md` / `DATASET.md` | Problem framing & full column reference |

---

## Option A — Run on Google Colab (recommended if you have no local GPU)

1. Go to **https://colab.research.google.com/** and click **File → Upload notebook**, then upload `colab_train_lstm.ipynb`.
2. (Optional but recommended) **Runtime → Change runtime type → Hardware accelerator: GPU**.
3. Run the first cell — it installs `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `torch`.
4. Run the second cell — a file picker appears. Upload **`master_daily.csv`** (~1.4 MB).
5. Run the remaining cells top-to-bottom. Training takes **~5–10 minutes on CPU, ~2 minutes on GPU**.
6. The last cell auto-downloads `lstm_best.pt`, `lstm_scaler.npz`, `lstm_metrics.json`.

**Expected output (local CPU reference run):**
```
device: cuda  (or cpu)
rows: 5,659  cols: 54  participants: 42
windows: 2,085   shape: (2085, 21, 34)    ← 34 features now, not 19
baseline MAE (predict mean): 7.92 days
[fold 1]  val MAE ≈ 4.3 days
[fold 2]  val MAE ≈ 4.4 days
...
CV mean val MAE: ~4.3 days                ← honest, no label leakage
```

---

## Option B — Run locally

### Prerequisites
- Python ≥ 3.10 (works on 3.11, 3.12, 3.13)
- ~3 GB disk for PyTorch

### Install
```bash
cd period-tracker
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Run EDA
```bash
python eda.py
# → figures/ populated with 10 PNGs
```

### Train the LSTM
```bash
python train_lstm.py --quick     # ~30s smoke test (2 folds, 8 epochs)
python train_lstm.py             # full run (5 folds, 60 epochs, ~5–10 min on CPU)
```

Outputs:
- `lstm_best.pt` — weights of the best CV fold
- `lstm_scaler.npz` — feature normalization (mean, std, feature_cols)
- `lstm_metrics.json` — per-fold MAE + baseline comparison

### Use the trained model
```python
import pandas as pd
from train_lstm import predict_days_to_next_period

recent_21_days = pd.read_csv('processed_period_data.csv').tail(21)
days_left = predict_days_to_next_period(recent_21_days)
print(f'Next period in {days_left:.1f} days')
```

---

## How the model works (one-paragraph summary)

We slide a **21-day window** over each participant's daily record in `master_daily.csv`. At window position *t*, the input is a `(21, 34)` tensor containing **hormones** (LH, estrogen), **self-reported cycle phase**, **resting heart rate**, **nightly skin temperature** + baseline SD, **sleep** (overall score, deep-sleep minutes, RHR, restlessness), **respiratory rate**, **activity** minutes (sedentary / lightly / moderately / very) + stress score, **heart-rate zones**, **12 Likert self-report symptoms** (cramps, fatigue, bloating, stress, mood, etc.), weekend flag, and participant age / menarche age. The target `days_to_next_period` is already derived in `master_daily.csv` from period-start detection (first Menstrual day after ≥ 7 quiet days). A 2-layer LSTM(64) → MLP(32 → 1) regresses this number. We train with Adam, MAE loss, dropout 0.3, early stopping, and **GroupKFold by participant id** so the reported MAE reflects generalization to people the model has never seen.

**Important — no leakage:** `flow_volume_num`, `is_bleeding`, `period_start`, `cycle_id`, and `day_in_cycle` are excluded from the feature set; they are either the target itself or trivially derive the target.

**Current best MAE: ~4.3 days** (baseline predict-mean: 7.9 days) — a 46 % improvement. An earlier 2.6-day number you may have seen came from including `flow_volume_num`, which leaked label info; the current setup is the honest one.

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'`** → `pip install torch` (on some platforms you need the CPU wheel: `pip install torch --index-url https://download.pytorch.org/whl/cpu`).
- **Colab "Runtime disconnected"** → re-run from cell 1; training resumes clean.
- **Predicting for a new participant with < 21 days of data** → the model needs at least `WINDOW_SIZE` days. Pad with the population mean or wait until more data is collected.
- **Different MAE on GPU vs CPU** → small differences (< 0.2 days) are normal due to CUDA nondeterminism; set `torch.manual_seed(0)` before training if reproducibility matters.
