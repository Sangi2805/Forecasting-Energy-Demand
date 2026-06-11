import pandas as pd
import numpy as np
import statsmodels.api as sm
import mlflow
import mlflow.statsmodels
import matplotlib.pyplot as plt
import os
import ast
import warnings
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import config as cfg

warnings.filterwarnings("ignore")

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
    "apparent_temperature", "wind_speed_10m",
    "demand_roll_mean_7d", "demand_roll_std_7d",
    "demand_lag_3d", "demand_lag_14d",
    "is_weekend", "is_holiday", "day_of_week",
]

train = train.dropna(subset=exog_features + ["demand"])
test  = test.dropna(subset=exog_features + ["demand"])
train.index.freq = "D"
test.index.freq  = "D"

y_train = train["demand"]
X_train = train[exog_features]
y_test  = test["demand"]
X_test  = test[exog_features]

n_windows = len(y_test) // 3
print(f"Test set: {len(y_test)} days → {n_windows} rolling 3-day windows")

# ==========================================
# 3. FETCH TUNED PARAMS FROM MLFLOW
# ==========================================
print("Fetching tuned params from latest sarimax_tuned run...")
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(
    experiment_ids=["0"],
    filter_string="tags.`mlflow.runName` = 'sarimax_tuned'",
    order_by=["start_time DESC"],
    max_results=1,
)
if not runs:
    raise RuntimeError("No sarimax_tuned run found in MLflow. Run tune_SARIMAX.py first.")

tuned_params       = runs[0].data.params
TUNED_ORDER          = ast.literal_eval(tuned_params["order"])
TUNED_SEASONAL_ORDER = ast.literal_eval(tuned_params["seasonal_order"])
print(f"Tuned order={TUNED_ORDER}  seasonal={TUNED_SEASONAL_ORDER}")

# ==========================================
# 4. ROLLING WALK-FORWARD EVALUATION
# ==========================================
def rolling_eval(order, seasonal_order, label):
    print(f"\nFitting {label} SARIMAX{order}{seasonal_order} on training data...")
    mdl = sm.tsa.statespace.SARIMAX(
        endog=y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = mdl.fit(disp=False)

    print(f"Running {n_windows} rolling 3-day windows...")
    actuals_d1, actuals_d2, actuals_d3 = [], [], []
    preds_d1,   preds_d2,   preds_d3   = [], [], []
    all_actuals, all_preds              = [], []

    current = results
    for i in range(n_windows):
        s = i * 3
        y_win = y_test.iloc[s:s + 3]
        X_win = X_test.iloc[s:s + 3]

        fc = current.forecast(steps=3, exog=X_win)

        actuals_d1.append(y_win.iloc[0]); preds_d1.append(fc.iloc[0])
        actuals_d2.append(y_win.iloc[1]); preds_d2.append(fc.iloc[1])
        actuals_d3.append(y_win.iloc[2]); preds_d3.append(fc.iloc[2])
        all_actuals.extend(y_win.values)
        all_preds.extend(fc.values)

        current = current.append(endog=y_win, exog=X_win, refit=False)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_windows}] windows done")

    a = np.array(all_actuals)
    p = np.array(all_preds)

    def metrics(act, pred):
        mae  = mean_absolute_error(act, pred)
        rmse = root_mean_squared_error(act, pred)
        mape = float(np.mean(np.abs((act - pred) / act)) * 100)
        return mae, rmse, mape

    avg_mae,  avg_rmse,  avg_mape  = metrics(a, p)
    d1_mae,   d1_rmse,   d1_mape   = metrics(np.array(actuals_d1), np.array(preds_d1))
    d2_mae,   d2_rmse,   d2_mape   = metrics(np.array(actuals_d2), np.array(preds_d2))
    d3_mae,   d3_rmse,   d3_mape   = metrics(np.array(actuals_d3), np.array(preds_d3))

    return {
        "label":    label,
        "actuals":  all_actuals,
        "preds":    all_preds,
        "avg_mae":  avg_mae,  "avg_rmse":  avg_rmse,  "avg_mape":  avg_mape,
        "day1_mae": d1_mae,   "day1_rmse": d1_rmse,   "day1_mape": d1_mape,
        "day2_mae": d2_mae,   "day2_rmse": d2_rmse,   "day2_mape": d2_mape,
        "day3_mae": d3_mae,   "day3_rmse": d3_rmse,   "day3_mape": d3_mape,
    }

baseline = rolling_eval((1, 1, 1), (1, 1, 1, 7), "baseline")
tuned    = rolling_eval(TUNED_ORDER, TUNED_SEASONAL_ORDER, "tuned")

# ==========================================
# 5. PRINT COMPARISON TABLE
# ==========================================
print("\n" + "=" * 60)
print(f"{'Metric':<20} {'Baseline':>15} {'Tuned':>15}")
print("=" * 60)
for key in ["avg_mae", "avg_rmse", "avg_mape",
            "day1_mae", "day1_rmse", "day1_mape",
            "day2_mae", "day2_rmse", "day2_mape",
            "day3_mae", "day3_rmse", "day3_mape"]:
    unit = "%" if "mape" in key else ""
    print(f"{key:<20} {baseline[key]:>14.2f}{unit} {tuned[key]:>14.2f}{unit}")
print("=" * 60)

# ==========================================
# 6. COMPARISON CHART
# ==========================================
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Actual vs predicted — baseline
axes[0].plot(y_test.iloc[:n_windows * 3].values, label="Actual", alpha=0.7, linewidth=0.8)
axes[0].plot(baseline["preds"], label="Baseline forecast", alpha=0.7, linewidth=0.8)
axes[0].set_title(f"Baseline SARIMAX(1,1,1)(1,1,1,7) — avg MAPE {baseline['avg_mape']:.2f}%")
axes[0].legend(); axes[0].set_ylabel("Demand")

# Actual vs predicted — tuned
axes[1].plot(y_test.iloc[:n_windows * 3].values, label="Actual", alpha=0.7, linewidth=0.8)
axes[1].plot(tuned["preds"], label="Tuned forecast", alpha=0.7, linewidth=0.8, color="orange")
axes[1].set_title(f"Tuned SARIMAX{TUNED_ORDER}{TUNED_SEASONAL_ORDER} — avg MAPE {tuned['avg_mape']:.2f}%")
axes[1].legend(); axes[1].set_ylabel("Demand")

# Per-day MAPE bar chart
days    = ["Day 1", "Day 2", "Day 3", "Average"]
b_mapes = [baseline["day1_mape"], baseline["day2_mape"], baseline["day3_mape"], baseline["avg_mape"]]
t_mapes = [tuned["day1_mape"],    tuned["day2_mape"],    tuned["day3_mape"],    tuned["avg_mape"]]
x = np.arange(len(days))
axes[2].bar(x - 0.2, b_mapes, 0.4, label="Baseline", color="steelblue")
axes[2].bar(x + 0.2, t_mapes, 0.4, label="Tuned",    color="orange")
axes[2].set_xticks(x); axes[2].set_xticklabels(days)
axes[2].set_ylabel("MAPE (%)"); axes[2].set_title("Per-day MAPE: Baseline vs Tuned")
axes[2].legend()

plt.tight_layout()
comparison_path = cfg.REPORT_DIR / "sarimax_comparison.png"
fig.savefig(comparison_path, dpi=150)
plt.close(fig)
print(f"\nChart saved: {comparison_path}")

# ==========================================
# 7. LOG BOTH TO DAGSHUB
# ==========================================
for res in [baseline, tuned]:
    run_name = f"sarimax_{res['label']}_full_eval"
    print(f"Logging {run_name} to DagsHub...")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("n_windows",    n_windows)
        mlflow.log_param("eval_days",    n_windows * 3)
        mlflow.log_param("features",     str(exog_features))

        for key in ["avg_mae", "avg_rmse", "avg_mape",
                    "day1_mae", "day1_rmse", "day1_mape",
                    "day2_mae", "day2_rmse", "day2_mape",
                    "day3_mae", "day3_rmse", "day3_mape"]:
            mlflow.log_metric(key, res[key])

        mlflow.log_artifact(str(comparison_path))

        pred_df = pd.DataFrame({
            "actual":    y_test.iloc[:n_windows * 3].values,
            "predicted": res["preds"],
        }, index=y_test.iloc[:n_windows * 3].index)
        fname = f"rolling_forecast_{res['label']}.csv"
        pred_df.to_csv(fname)
        mlflow.log_artifact(fname)

print("\nDone. Both runs logged to DagsHub.")
