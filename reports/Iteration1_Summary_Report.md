# Iteration 1 Summary Report
**Project:** End-to-End Multivariate Time Series Forecasting for Energy Demand (FED)  
**Group 1 · AI 6002 Capstone · Memorial University of Newfoundland**  
**Date:** June 11, 2026  

**References:** [FED Software Requirements Document (SRD)](../../FED_Software_Requirements_Document.tex) · Git branch [`iteration1`](https://github.com/Sangi2805/Forecasting-Energy-Demand/tree/iteration1) · [DagsHub repo & MLflow](https://dagshub.com/Sangi2805/Forecasting-Energy-Demand) · Presentation deck *Iteration1_1.pdf*

---

## 1. Context & Iteration 1 Scope

The FED system (per the SRD, May 23, 2026) targets end-to-end multivariate forecasting of regional electricity demand: unified data ingestion, model training and comparison (LSTM, XGBoost, Prophet, SARIMAX), MLOps tracking via MLflow/DagsHub, and a Streamlit dashboard. Iteration 1 (May 25 – Jun 12, 2026) was scoped to **data pipeline, EDA, and baseline models**—aligned with SRD Chapter 10 milestones and deliverables (preprocessed dataset, EDA visualizations, ≥2 baseline models with MLflow records).

All Iteration 1 work is captured on the **`iteration1`** branch (snapshot of `shakeel/ml-forecasting-pipelines`), including preprocessing, four trained baselines, prediction CSVs, feature-importance outputs, and the Streamlit UI. Experiment runs are logged on DagsHub under the **Forecasting-Energy-Demand** repository: MLflow tracking URI `https://dagshub.com/Sangi2805/Forecasting-Energy-Demand.mlflow`, with runs in experiment **`energy-demand-forecasting`** (XGBoost tuning/training) and per-model runs (`lgbm_3day_forecast`, Prophet, SARIMAX) logging MAE, RMSE, MAPE, parameters, and artifacts (models, plots, prediction CSVs).

---

## 2. What We Delivered

### Data pipeline & versioning
- **Sources:** EIA Grid Monitor (NY demand), Open-Meteo weather, US holidays, optional GDP/population (Jul 2015 – May 2026).
- **Pipeline:** `preprocessing.py` — read → standardize → merge → engineer → impute. Features include day-of-week, month, season, hour, holiday flags, and 1–2 day weather lags.
- **Split:** 80/20 chronological hold-out → `features_selected_train.csv` / `features_selected_test.csv`.
- **DVC + DagsHub:** Frozen datasets versioned with DVC; remote on DagsHub S3. Pull via `python pull_data.py` → `dvc pull`. **Challenge:** `dvc push` mirroring break prevented full team sync; some teammates used local copies.

### EDA — key findings (*Iteration1_1.pdf*, slides 3–4, 26–41)
1. **Time features matter most** — season, month, day-of-week, and hour strongly shape demand.
2. **V-shaped temperature effect** — demand lowest at 10–18°C; rises at cold and hot extremes.
3. **Snowfall elevates demand** — consistently higher consumption on snowfall days.
4. **Holidays shift patterns** — measurable deviations from normal weekly cycles.
5. **Long-term trend** — gradual downward drift over the study period; summer (Jul–Aug) peaks, spring/autumn troughs.
6. **Hourly patterns** — off-peak 03:00–05:00; peak 17:00–19:00; weekdays stable, weekends lower (Sunday lowest).

### Models & DagsHub experiments
Four 3-day-ahead baselines trained on identical frozen test data; metrics logged per horizon (day 1–3) and averaged:

| Model | Avg MAE (MW) | Avg RMSE (MW) | Avg MAPE (%) | Best horizon |
|-------|-------------|---------------|--------------|--------------|
| **XGBoost** | 20,575 | **29,105** | **5.79%** | Day 1 (RMSE 21,126) |
| LightGBM | 20,877 | 29,799 | 5.79% | Day 1 (RMSE 21,055) |
| Prophet | 23,175 | 32,048 | 6.27% | Day 1 (RMSE 25,523) |
| SARIMAX | 83,515 | 96,838 | 21.79% | Day 3 (RMSE 61,622) |

**XGBoost** is the overall winner. Tree models (XGBoost, LightGBM) dominate; error grows with horizon (Day 1 MAPE ~3.6% → Day 3 ~8.5%). **SARIMAX** is a clear outlier—strong in-sample metrics (~15.5K MAE) but poor out-of-sample (avg RMSE 96,838 MW), indicating overfitting / feature-set mismatch.

### Feature importance (cross-model alignment)
- **XGBoost:** `apparent_temperature` (#1, 14.8% gain), `demand`, `season`, `day_of_week`.
- **LightGBM:** `demand`, `demand_lag_4d`, `total_interchange` (flagged as potentially leaky).
- **Prophet:** temperature-driven (`apparent_temperature`, `temperature_2m`, `temp_max`).
- **SARIMAX:** `is_holiday` dominates all other exogenous regressors.
- **Gaps not yet engineered:** explicit V-shape feature |temp − 14°C|, soft snowfall flag.

### UI
Streamlit + Plotly dashboard (`app/streamlit_app.py`): model selector ranked by avg RMSE, 3-day forecast table, MAE/RMSE/MAPE, actual-vs-predicted chart, expandable compare-all-models panel. Predictions read from `reports/*_predictions.csv`.

### Deferred to Iteration 2
LSTM (Toan’s branch `Xuan_Toan_Doan` in progress); tuned SARIMAX; engineered V-shape/snowfall features; unified MLflow leaderboard; DVC remote fix; confidence intervals and explainability in UI.

---

## 3. Presentation Feedback & Iteration 2 Action Items

Feedback from the Iteration 1 presentation (*Iteration1_1.pdf* demo) and Q&A, consolidated for the next sprint:

| # | Feedback | Planned response |
|---|----------|------------------|
| 1 | **UI clarity** — in “Compare all models,” predicted vs actual lines are hard to distinguish | Use distinct colors, line styles (solid vs dashed), and legend labels; add a **dotted line for actual demand** across all forecast charts |
| 2 | **SARIMAX metrics** — results are an outlier vs other models | Audit train vs test metrics in DagsHub (`day*_train_mape` vs test MAE/RMSE/MAPE); reduce exogenous dimensionality, revisit order/seasonal terms, and compare in-sample vs out-of-sample gap explicitly |
| 3 | **Feature engineering** — needs dedicated iteration | Model-specific feature design (see rows 9–11); remove or justify leaky features (`total_interchange`, raw `demand` as predictor) |
| 4 | **EDA hourly graphs** — reorganize into sections | Split hourly analysis into off-peak / morning ramp / peak / evening decline blocks to guide preprocessing and feature bins |
| 5 | **Why is `demand` a feature?** | Document rationale (autoregressive lags/rolling means for tree models) vs leakage risk; separate target history features from contemporaneous demand if inappropriate for a given model family |
| 6 | **Train vs test gap** | Log and report both splits for every model in MLflow; surface gap in UI and final model presentation to detect overfitting early |
| 7 | **Best-model presentation** | When presenting the final winner, show **top features with rank** (not only a bar chart) |
| 8 | **Model-aware feature engineering** | Tailor features to how each algorithm consumes inputs — e.g. **hour as classification** (peak/off-peak bins) for trees; Fourier/seasonal terms for Prophet/SARIMAX |
| 9 | **Extreme-data flags** | Add binary flags for heat waves, cold snaps, and snowfall extremes to help models learn non-linear weather effects |
| 10 | **Temperature classification** | Implement **K-means clustering** on temperature (or apparent temperature) as an alternative/complement to the V-shape engineered feature |

---

## 4. Conclusion

Iteration 1 met the SRD’s core milestones for data engineering and baseline modeling: a reproducible multivariate pipeline, frozen train/test data (DVC), four comparable baselines with DagsHub-tracked experiments, EDA-backed feature insights, and a working forecast dashboard. XGBoost leads on held-out 3-day average RMSE (29,105 MW, 5.79% MAPE). Known gaps—SARIMAX instability, incomplete team DVC sync, missing LSTM, and shallow feature engineering—are documented with a clear Iteration 2 roadmap shaped by supervisor and client feedback. The `iteration1` branch and DagsHub experiment history provide the audit trail for this milestone.

**Team:** Xuan Toan Doan · MD Shahriar Rashid · Sangaranarayanan SV · Mohammad Shakeel · Sai Sushanth Chandrasekaran
