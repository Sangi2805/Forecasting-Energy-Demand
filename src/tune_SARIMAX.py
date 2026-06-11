import pandas as pd
import numpy as np
import statsmodels.api as sm
import mlflow
import mlflow.statsmodels
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from itertools import product
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import config as cfg

# ==========================================
# 1. DAGSHUB & MLFLOW SETUP
# ==========================================
load_dotenv()
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USER_NAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
mlflow.set_tracking_uri("https://dagshub.com/Sangi2805/Forecasting-Energy-Demand.mlflow")

# ==========================================
# 2. DATA LOADING
# ==========================================
print("Loading datasets...")
train = pd.read_csv("data/processed/features_selected_train.csv", parse_dates=["date"], index_col="date")
test  = pd.read_csv("data/processed/features_selected_test.csv",  parse_dates=["date"], index_col="date")

exog_features = [
    # Weather — LASSO coef #1 (464K standardized); encodes temp+humidity → HVAC load; corr=0.42
    "apparent_temperature",
    # Wind — LASSO coef 29.7K; physically independent from temperature; corr=0.18
    "wind_speed_10m",
    # Demand baseline — highest overall corr=0.76; LASSO coef 26.8K; recent demand level
    "demand_roll_mean_7d",
    # Demand volatility — low VIF=4.6 (independent signal); LASSO coef 3.7K
    "demand_roll_std_7d",
    # Demand lags — 3d: minimum safe lag, LASSO coef 11.5K, corr=0.69
    #               14d: bi-weekly cycle, different time scale from 3d
    "demand_lag_3d",
    "demand_lag_14d",
    # Calendar — all confirmed *** significant; is_weekend largest effect (-28K coef)
    "is_weekend",
    "is_holiday",
    "day_of_week",
]

train = train.dropna(subset=exog_features + ["demand"])
test  = test.dropna(subset=exog_features + ["demand"])

train.index.freq = "D"
test.index.freq  = "D"

y_train = train["demand"]
X_train = train[exog_features]
y_test  = test["demand"]
X_test  = test[exog_features]

# ==========================================
# 3. HYPERPARAMETER GRID
# ==========================================
S = 7
param_grid = list(product(
    [0, 1, 2],  # p
    [0, 1],     # d
    [0, 1, 2],  # q
    [0, 1],     # P
    [0, 1],     # D
    [0, 1],     # Q
))
print(f"Grid size: {len(param_grid)} combos × 5 folds = {len(param_grid) * 5} fits")

# ==========================================
# 4. 5-FOLD WALK-FORWARD CROSS-VALIDATION
# ==========================================
tscv = TimeSeriesSplit(n_splits=5)

best_cv_rmse = np.inf
best_params  = None
cv_log       = []

print("Starting grid search (this will take a while)...")
for i, (p, d, q, P, D, Q) in enumerate(param_grid):
    order          = (p, d, q)
    seasonal_order = (P, D, Q, S)
    fold_rmses     = []

    for train_idx, val_idx in tscv.split(y_train):
        y_tr  = y_train.iloc[train_idx]
        X_tr  = X_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]
        X_val = X_train.iloc[val_idx]

        try:
            mdl = sm.tsa.statespace.SARIMAX(
                endog=y_tr,
                exog=X_tr,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res   = mdl.fit(disp=False, maxiter=50)
            preds = res.forecast(steps=3, exog=X_val.iloc[:3])
            fold_rmses.append(root_mean_squared_error(y_val.iloc[:3], preds))
        except Exception:
            fold_rmses = [np.inf]
            break

    avg_rmse = float(np.mean(fold_rmses))
    cv_log.append({
        "p": p, "d": d, "q": q,
        "P": P, "D": D, "Q": Q,
        "cv_rmse": avg_rmse,
    })

    if avg_rmse < best_cv_rmse:
        best_cv_rmse = avg_rmse
        best_params  = (order, seasonal_order)

    if (i + 1) % 20 == 0 or (i + 1) == len(param_grid):
        print(f"  [{i+1}/{len(param_grid)}] Best CV RMSE: {best_cv_rmse:.2f}")

best_order, best_seasonal_order = best_params
print(f"\nBest order={best_order}  seasonal={best_seasonal_order}  CV RMSE={best_cv_rmse:.2f}")

# ==========================================
# 5. REFIT ON FULL TRAIN + LOG TO DAGSHUB
# ==========================================
print("Refitting on full training data and logging to DagsHub...")
with mlflow.start_run(run_name="sarimax_tuned"):

    # --- params ---
    mlflow.log_params({
        "model_type":     "SARIMAX_tuned",
        "order":          str(best_order),
        "seasonal_order": str(best_seasonal_order),
        "features":       str(exog_features),
        "train_size":     len(train),
        "cv_folds":       5,
        "grid_size":      len(param_grid),
    })
    mlflow.log_metric("cv_best_rmse", best_cv_rmse)

    # --- fit ---
    final_model = sm.tsa.statespace.SARIMAX(
        endog=y_train,
        exog=X_train,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    final_results = final_model.fit(disp=False)
    mlflow.statsmodels.log_model(final_results, "sarimax_model")

    # --- forecast ---
    print("Forecasting next 3 days...")
    forecast = final_results.forecast(steps=3, exog=X_test.iloc[:3])
    actuals  = y_test.iloc[:3]

    # per-day metrics
    for i in range(3):
        day = i + 1
        a   = actuals.iloc[i]
        p   = forecast.iloc[i]
        mlflow.log_metric(f"day{day}_mae",  abs(a - p))
        mlflow.log_metric(f"day{day}_rmse", float(np.sqrt((a - p) ** 2)))
        mlflow.log_metric(f"day{day}_mape", abs((a - p) / a) * 100)

    # average metrics
    avg_mae  = mean_absolute_error(actuals, forecast)
    avg_rmse = root_mean_squared_error(actuals, forecast)
    avg_mape = float(np.mean(np.abs((actuals.values - forecast.values) / actuals.values)) * 100)

    mlflow.log_metric("avg_mae",  avg_mae)
    mlflow.log_metric("avg_rmse", avg_rmse)
    mlflow.log_metric("avg_mape", avg_mape)

    print(f"avg RMSE: {avg_rmse:.2f}  MAE: {avg_mae:.2f}  MAPE: {avg_mape:.2f}%")

    # --- feature importance ---
    coefs   = final_results.params[exog_features]
    tstats  = final_results.tvalues[exog_features]
    pvalues = final_results.pvalues[exog_features]

    print("\n--- Feature Importance (ranked by |t-stat|) ---")
    print(f"{'Feature':<25} {'Coef':>12} {'t-stat':>10} {'p-value':>10}  Sig")
    print("-" * 65)
    for feat in sorted(exog_features, key=lambda f: abs(tstats[f]), reverse=True):
        sig = "***" if pvalues[feat] < 0.01 else ("**" if pvalues[feat] < 0.05 else ("*" if pvalues[feat] < 0.1 else ""))
        print(f"{feat:<25} {coefs[feat]:>12.4f} {tstats[feat]:>10.3f} {pvalues[feat]:>10.4f}  {sig}")
    print("--- p<0.1: * | p<0.05: ** | p<0.01: *** ---\n")

    for feat in exog_features:
        mlflow.log_metric(f"coef_{feat}",   float(coefs[feat]))
        mlflow.log_metric(f"tstat_{feat}",  float(tstats[feat]))
        mlflow.log_metric(f"pvalue_{feat}", float(pvalues[feat]))

    # feature importance chart (ranked by |t-stat|)
    abs_t      = np.abs(tstats.values)
    sorted_idx = np.argsort(abs_t)
    fig, ax    = plt.subplots(figsize=(8, 5))
    ax.barh(
        [exog_features[j] for j in sorted_idx],
        abs_t[sorted_idx],
        color="steelblue"
    )
    ax.set_xlabel("|t-statistic|")
    ax.set_title("Feature Importance — SARIMAX exogenous coefficients")
    plt.tight_layout()
    importance_path = cfg.REPORT_DIR / "SARIMAX_feature_importance.png"
    fig.savefig(importance_path)
    mlflow.log_artifact(str(importance_path))
    plt.close(fig)

    # --- artifacts ---
    forecast_df = pd.DataFrame(
        {"actual": actuals.values, "predicted": forecast.values},
        index=actuals.index,
    )
    forecast_df.to_csv("forecast_results_tuned.csv")
    mlflow.log_artifact("forecast_results_tuned.csv")

    cv_df = pd.DataFrame(cv_log).sort_values("cv_rmse")
    cv_df.to_csv("cv_results.csv", index=False)
    mlflow.log_artifact("cv_results.csv")

    print("Run successfully logged to DagsHub!")
