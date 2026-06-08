import sys
import os
import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn

from xgboost import XGBRegressor
from sklearn.multioutput import RegressorChain
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

# ── Paths via config ──
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import PROCESSED_DATA_DIR

# ── Connect MLflow to DagsHub ──
dagshub.init(repo_owner="Sangi2805", repo_name="Forecasting-Energy-Demand", mlflow=True)
mlflow.set_experiment("energy-demand-forecasting")

# ── Load data ──
train = pd.read_csv(PROCESSED_DATA_DIR / "features_selected_train.csv", index_col="date", parse_dates=["date"])
test = pd.read_csv(PROCESSED_DATA_DIR / "features_selected_test.csv", index_col="date", parse_dates=["date"])

targets = ["target_day1", "target_day2", "target_day3"]

drop_cols = [
    "net_generation", "total_interchange", "ng_nuclear", "ng_hydro",
    "ng_solar", "ng_wind", "ng_natural_gas",
    "is_weekend", "quarter", "snow_depth", "snowfall",
    "wind_gusts_10m", "relative_humidity_2m",
    "relative_humidity_2m_lag_2d", "day_of_month",
]

X_train = train.drop(columns=targets + drop_cols)
y_train = train[targets]
X_test = test.drop(columns=targets + drop_cols)
y_test = test[targets]

X_train["season"] = (X_train.index.month % 12) // 3
X_test["season"] = (X_test.index.month % 12) // 3

# ── RegressorChain: feeds day1 -> day2 -> day3 ──
base = RegressorChain(
    XGBRegressor(random_state=42, tree_method="hist", device="cpu")
)

# Note the base_estimator__ prefix (RegressorChain wraps the model differently than MultiOutputRegressor)
param_dist = {
    "estimator__n_estimators":     [300, 500, 800],
    "estimator__max_depth":        [3, 4, 5, 6],
    "estimator__learning_rate":    [0.01, 0.03, 0.05, 0.07],
    "estimator__subsample":        [0.7, 0.8, 0.9, 1.0],
    "estimator__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "estimator__min_child_weight": [1, 3, 5],
    "estimator__reg_lambda":       [1, 2, 5],
    "estimator__gamma":            [0, 0.1, 0.3],
}

search = RandomizedSearchCV(
    base,
    param_dist,
    n_iter=25,
    cv=TimeSeriesSplit(n_splits=5),
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

with mlflow.start_run(run_name="xgb-sangar-chain-tuned"):

    search.fit(X_train, y_train)
    best = search.best_estimator_

    best_cv_rmse = -search.best_score_
    mlflow.log_metric("best_cv_rmse", best_cv_rmse)
    print(f"\nBest CV RMSE: {best_cv_rmse:,.2f}")

    mlflow.log_params({
        k.replace("estimator__", ""): v
        for k, v in search.best_params_.items()
    })

    preds = best.predict(X_test)

    mae_list, mape_list, rmse_list = [], [], []
    metric_names = ["day1", "day2", "day3"]

    for i, name in enumerate(metric_names):
        mae = mean_absolute_error(y_test.iloc[:, i], preds[:, i])
        mape = mean_absolute_percentage_error(y_test.iloc[:, i], preds[:, i]) * 100
        rmse = mean_squared_error(y_test.iloc[:, i], preds[:, i]) ** 0.5

        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)

        mlflow.log_metric(f"{name}_mae", mae)
        mlflow.log_metric(f"{name}_mape", mape)
        mlflow.log_metric(f"{name}_rmse", rmse)

        print(f"{name}: MAE={mae:,.2f} MAPE={mape:.2f}% RMSE={rmse:,.2f}")

    mlflow.log_metric("avg_mae", sum(mae_list) / 3)
    mlflow.log_metric("avg_mape", sum(mape_list) / 3)
    mlflow.log_metric("avg_rmse", sum(rmse_list) / 3)

    print("\nAverage Metrics")
    print(f"avg_mae  = {sum(mae_list)/3:,.2f}")
    print(f"avg_mape = {sum(mape_list)/3:.2f}%")
    print(f"avg_rmse = {sum(rmse_list)/3:,.2f}")

    mlflow.sklearn.log_model(best, name="model")