---
title: EnergyAI NYISO Demand Forecasting
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.58.0
app_file: app/streamlit_app.py
python_version: "3.12"
pinned: false
license: mit
---

# Forecasting New York Electricity Demand

Hourly electricity demand forecasting for the eleven NYISO load zones, five days
ahead, using a Temporal Fusion Transformer.

The dashboard runs the trained model on **live** data — NYISO's published load
and Open-Meteo's weather forecast — and puts the result next to NYISO's own
published forecast for the same hours.

**Held-out accuracy: 2.95% MAPE overall, 2.21% at 24 hours** (statewide,
lead-matched evaluation over 2024-01-23 onward).

| lead | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 |
|------|-------|-------|-------|-------|-------|
| MAPE | 2.21% | 2.54% | 2.86% | 3.32% | 3.81% |

Seasonal: Winter 2.93% · Spring 3.11% · Summer 3.40% · Fall 2.27%

---

## The dashboard

**Forecasting** — banked predictions across 854 issued forecasts on the held-out
test period, scored against realised demand. Pick an issue date, a lead day and a
zone.

**Live forecast** — the same checkpoint run now: 168 hours of realised load as
encoder, 120 hours of forecast weather as decoder. Shows our median with its
P10–P90 band alongside NYISO's own forecast, hour by hour and by zone.

---

## Running it

Python 3.11 or 3.12.

```bash
python -m venv .venv
.venv/Scripts/Activate.ps1        # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The first live forecast takes ~30 s on a cold start (checkpoint load plus 19
parallel HTTP requests); afterwards a refresh is ~2 s. Manual refreshes are
limited to one per 10 minutes, to be kind to two free public APIs.

---

## How it works

```
NYISO palIntegrated  ─┐
                      ├─→ live_data ─→ live_features ─→ live_forecast ─→ dashboard
Open-Meteo forecast  ─┘                                       ↑
                                              zonal_tft_refit2023_fx_best.ckpt
```

| module | role |
|--------|------|
| `app/tft_core.py` | model + dataset contract, mirrored from the eval script so the live and training paths cannot drift |
| `app/live_data.py` | NYISO demand, Open-Meteo weather, NYISO's own forecast |
| `app/live_features.py` | calendar, lags, rolling stats and heat-memory features |
| `app/live_forecast.py` | inference, disk cache, refresh rate limit |
| `app/streamlit_app.py` | dashboard |

### Model

Temporal Fusion Transformer, 450K parameters, `hidden_size=64`, 4 attention
heads, trained with `QuantileLoss` over 7 quantiles. 168-hour encoder,
120-hour decoder, one series per zone with a `GroupNormalizer` keyed on zone.

Inputs are six weather variables per zone (temperature, apparent temperature,
humidity, wind, shortwave radiation, cloud cover), a V-shaped temperature
deviation, three heat-memory features, calendar terms derived from Eastern local
time, and demand lags and rolling statistics.

### Details that matter

**Heat memory.** `fx_hot_streak_day` counts consecutive days above 24 °C without
bound. Computing it over a short live window restarts the count mid-streak — 23
days of error for an August origin — and because it is a *known* real it feeds
the decoder directly and shifts the forecast. It is seeded from ~400 days of
archive apparent temperature, long enough to always span a winter.

**DST.** NYISO stamps every row with an explicit `EST`/`EDT` column; that is used
rather than inferring an offset, which is ambiguous for the repeated 01:00 hour
each November.

**Unknown futures.** Demand and its lag/rolling derivatives are unknown across
the horizon but `TimeSeriesDataSet` still requires finite values. They are
persistence-filled, which is safe because the TFT feeds unknown reals to the
encoder only — verified, not assumed: perturbing them by 3,782 MW changes
predictions by 0.00000000 MW.

**Prediction intervals** come from the model's own quantiles, not a fitted error
band. Measured coverage over 22 origins and 2,640 statewide hours is **78.3%**
against a nominal 80%. Per zone it is 47–75%, i.e. over-confident, and the UI
says so rather than presenting both alike.

---

## Validation

Every stage is checked against something independent.

```bash
python scripts/validate_inference.py       # local inference vs the training cluster
python scripts/validate_live_data.py       # fetchers vs the banked archives
python scripts/validate_live_features.py   # feature builder + decoder-fill inertness
python scripts/backtest_live_pipeline.py   # live path vs realised demand
python scripts/validate_live_forecast.py   # today's forecast vs NYISO's own
```

| check | result |
|-------|--------|
| inference reproduces the training cluster | 0.07 MW max over 264 rows |
| fetchers reproduce the archives | 0.000000 MW over 744 hours |
| decoder unknowns are inert | 0.00000000 MW under a 3,782 MW perturbation |
| live pipeline vs realised demand | 2.65% MAPE, +0.34% bias |

`validate_live_features.py` takes an origin via `VLF_ORIGIN`; run it against a
summer date, since a spring window will not exercise the heat-streak path.

---

## Repository

```
app/          dashboard and live forecasting
scripts/      validation harnesses
Sangar/       data pipeline, training and evaluation; the checkpoint
reports/      banked predictions and figures
```

Training and evaluation were run separately on GPU; `Sangar/04`–`09` cover data
download, feature engineering, training, evaluation, permutation importance and
prediction export.

Note the prediction archive ends **2026-05-31**. The Forecasting tab is bounded
by that; the Live tab is not, since it fetches current data.
