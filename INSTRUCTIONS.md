# Project Package — Forecasting Energy Demand

Instructions to set up the environment, run training, and reproduce the reported zonal TFT results (**overall MAPE 2.95%**, day-ahead **2.21%**).

## Contents

| Path | Purpose |
|------|---------|
| `Sangar/05_train_refit2023_fx.py` | Train the reported zonal TFT |
| `Sangar/06_eval_refit2023_fx.py` | Lead-matched evaluation (reproduces metrics) |
| `Sangar/07_permutation_importance.py` | Optional feature importance |
| `Sangar/ckpt_utils.py` | Portable CUDA→CPU/MPS checkpoint loader |
| `Sangar/zonal_features_fx.parquet` | Full feature table used by train/eval |
| `Sangar/weather_leads_zonal.parquet` | Lead-matched weather for leak-free eval |
| `Sangar/checkpoints_zonal/zonal_tft_refit2023_fx_best.ckpt` | Trained model for reported metrics |
| `data/splits/` | Explicit train / validation / test partitions |
| `data/processed/` | Classic daily feature train/test CSVs (baselines) |
| `app/streamlit_app.py` | Dashboard (optional) |
| `params.yaml` | Split dates and model hyperparameters |
| `requirements.txt` | **Use this file** for TFT train/eval |

> Note: `requirements-training.txt` is a legacy extras list (Streamlit/TF/xgboost/MLflow). It is **not** required for reproducing the reported TFT metrics.

## 1. Environment setup

Python **3.10** recommended (3.10–3.12 should work). From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verified on macOS Apple Silicon with a fresh venv and `pip install -r requirements.txt` as written (no special PyTorch index required).

GPU/CUDA is optional. Scripts auto-select `cuda`, Apple `mps`, or `cpu`. The included checkpoint was trained on CUDA; `ckpt_utils.py` remaps metric devices so load works on CPU/MPS hosts.

**Runtime:** full lead-matched eval on CPU is typically **~1–2 hours**. GPU is much faster. Training from scratch is slow on CPU — prefer a GPU host for section 4.

## 2. Datasets (train / validation / test)

Temporal splits (UTC), matching `params.yaml`:

| Split | Start | End | File |
|-------|-------|-----|------|
| Train | 2015-07-01 04:00 | 2024-01-01 05:00 (exclusive) | `data/splits/train_zonal_fx.parquet` |
| Validation | 2024-01-01 05:00 | 2024-01-23 12:00 (exclusive) | `data/splits/validation_zonal_fx.parquet` |
| Test | 2024-01-23 12:00 | end of series | `data/splits/test_zonal_fx.parquet` |

See `data/splits/split_metadata.json` for row counts.

**Important:** TFT uses a 168-hour encoder, so lookback crosses split boundaries. Train and eval scripts therefore load the full table `Sangar/zonal_features_fx.parquet` and apply the same date cutoffs internally. The files under `data/splits/` are the explicit partitions requested for submission/inspection.

Supporting raw inputs (rebuild path):

- `Sangar/nyiso_zonal_hourly.parquet`
- `Sangar/weather_observed_zonal.parquet`
- `Sangar/zonal_features.parquet` (pre–fx-feature table)

## 3. Reproduce reported results (recommended)

Uses the included checkpoint — no retrain required.

```bash
cd Sangar
python 06_eval_refit2023_fx.py
```

Expected output (approx.):

- Day 1 MAPE ≈ **2.21%**
- Overall MAPE ≈ **2.95%**
- Metrics written to `Sangar/metrics_zonal_refit2023_fx.json`

First run caches `Sangar/fx_lead{1..5}.npz` (large intermediate files; safe to delete). Restarts reuse them and finish much faster. For a clean cold-start timing check:

```bash
rm -f Sangar/fx_lead*.npz
cd Sangar && python 06_eval_refit2023_fx.py
```

## 4. Run training

Retrains the zonal TFT (2 epochs, lr `3e-4`, seed `42`) and writes a new checkpoint under `Sangar/checkpoints_zonal/`.

```bash
cd Sangar
python 05_train_refit2023_fx.py
```

Then re-run evaluation (section 3). Full retrain can take a long time on CPU; a GPU host is strongly preferred.

Optional: set `NUM_WORKERS` > 0 in `05_train_refit2023_fx.py` on multi-core machines.

## 5. Optional: permutation importance

```bash
cd Sangar
python 07_permutation_importance.py day1
```

## 6. Optional: Streamlit dashboard

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The Forecasting tab reads `reports/tft_zonal_predictions.csv` and `reports/tft_hourly_predictions.parquet` when present.

## 7. Reported metrics reference

From `Sangar/metrics_zonal_refit2023_fx.json`:

| Horizon | MAPE |
|---------|------|
| Day 1 | 2.206% |
| Day 2 | 2.543% |
| Day 3 | 2.859% |
| Day 4 | 3.319% |
| Day 5 | 3.814% |
| **Overall** | **2.948%** |

Checkpoint: `zonal_tft_refit2023_fx_best.ckpt` (2 epochs, lr=3e-4, train through 2023, +fx features).
