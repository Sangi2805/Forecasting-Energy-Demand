import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

dagshub.init(
    repo_owner="Sangi2805",
    repo_name="Forecasting-Energy-Demand",
    mlflow=True
)

mlflow.set_experiment("energy-demand-forecasting")

train = pd.read_csv(
    "data/processed/features_selected_train.csv",
    index_col="date",
    parse_dates=["date"]
)

test = pd.read_csv(
    "data/processed/features_selected_test.csv",
    index_col="date",
    parse_dates=["date"]
)

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

param_dist = {
    "estimator__n_estimators": [200, 300, 500, 800],
    "estimator__max_depth": [3, 4, 5, 6],
    "estimator__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "estimator__subsample": [0.7, 0.8, 1.0],
    "estimator__colsample_bytree": [0.7, 0.8, 1.0],
}

base = MultiOutputRegressor(
    XGBRegressor(random_state=42)
)

search = RandomizedSearchCV(
    base,
    param_dist,
    n_iter=15,
    cv=TimeSeriesSplit(n_splits=4),
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)

with mlflow.start_run(run_name="xgb-sangar-tuned"):

    search.fit(X_train, y_train)

    best = search.best_estimator_

    mlflow.log_params({
        k.replace("estimator__", ""): v
        for k, v in search.best_params_.items()
    })

    preds = best.predict(X_test)

    mae_list = []
    mape_list = []
    rmse_list = []

    metric_names = ["day1", "day2", "day3"]

    for i, name in enumerate(metric_names):

        mae = mean_absolute_error(
            y_test.iloc[:, i],
            preds[:, i]
        )

        mape = mean_absolute_percentage_error(
            y_test.iloc[:, i],
            preds[:, i]
        ) * 100

        rmse = mean_squared_error(
            y_test.iloc[:, i],
            preds[:, i]
        ) ** 0.5

        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)

        mlflow.log_metric(f"{name}_mae", mae)
        mlflow.log_metric(f"{name}_mape", mape)
        mlflow.log_metric(f"{name}_rmse", rmse)

        print(
            f"{name}: "
            f"MAE={mae:,.2f} "
            f"MAPE={mape:.2f}% "
            f"RMSE={rmse:,.2f}"
        )

    mlflow.log_metric("avg_mae", sum(mae_list) / 3)
    mlflow.log_metric("avg_mape", sum(mape_list) / 3)
    mlflow.log_metric("avg_rmse", sum(rmse_list) / 3)

    print("\nAverage Metrics")
    print(f"avg_mae  = {sum(mae_list)/3:,.2f}")
    print(f"avg_mape = {sum(mape_list)/3:.2f}%")
    print(f"avg_rmse = {sum(rmse_list)/3:,.2f}")

    mlflow.sklearn.log_model(best, name="model")