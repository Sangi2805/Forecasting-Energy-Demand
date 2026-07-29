# Forecasting Energy Demand

NYISO zonal energy demand forecasting with a Temporal Fusion Transformer (TFT).
Reported result: **2.95% overall MAPE** (2.21% day-ahead) on the held-out test period.

## Quick start

See **[INSTRUCTIONS.md](INSTRUCTIONS.md)** for environment setup, datasets, training, and how to reproduce reported metrics.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd Sangar && python 06_eval_refit2023_fx.py
```

## Layout

```text
INSTRUCTIONS.md          # Setup + reproduction steps
params.yaml              # Splits and hyperparameters
requirements.txt
Sangar/                  # TFT train/eval + zonal datasets + checkpoint
data/splits/             # Explicit train / validation / test partitions
data/processed/          # Classic daily feature CSVs
app/streamlit_app.py     # Optional dashboard
reports/                 # Predictions and figures
src/                     # Shared utilities / earlier baselines
```
