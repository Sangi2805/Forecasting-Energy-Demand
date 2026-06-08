import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from prophet import Prophet

import src.config as cfg
from src.evaluate import regression_metrics
from src.predictions import save_combined_predictions_csv
from src.prophet.prophet_features import (
    engineer_features,
    get_regressor_columns,
    load_datasets,
    prepare_prophet_frame,
)

FORECAST_HORIZONS = [1, 2, 3]
DAGSHUB_METRICS = ["mae", "mape", "rmse"]
VALIDATION_DAYS = 365

BASE_PROPHET_PARAMS = {
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "interval_width": 0.9,
}

PARAM_GRID = [
    {"changepoint_prior_scale": 0.01, "seasonality_prior_scale": 10.0, "seasonality_mode": "multiplicative"},
    {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0, "seasonality_mode": "multiplicative"},
    {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 5.0, "seasonality_mode": "multiplicative"},
    {"changepoint_prior_scale": 0.1, "seasonality_prior_scale": 10.0, "seasonality_mode": "multiplicative"},
    {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0, "seasonality_mode": "additive"},
]


def build_prophet_model(regressors: list[str], params: dict) -> Prophet:
    model = Prophet(**BASE_PROPHET_PARAMS, **params)
    model.add_country_holidays(country_name="US")
    for regressor in regressors:
        model.add_regressor(regressor, standardize=True)
    return model


def split_train_validation(
    train_df: pd.DataFrame, validation_days: int = VALIDATION_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = train_df["date"].max() - pd.Timedelta(days=validation_days)
    fit_df = train_df[train_df["date"] <= cutoff].copy()
    val_df = train_df[train_df["date"] > cutoff].copy()
    return fit_df, val_df


def fit_horizon_model(
    train_df: pd.DataFrame,
    horizon: int,
    regressors: list[str],
    params: dict,
) -> Prophet:
    train_frame = prepare_prophet_frame(train_df, horizon, regressors)
    model = build_prophet_model(regressors, params)
    model.fit(train_frame)
    return model


def predict_horizon(
    model: Prophet,
    df: pd.DataFrame,
    regressors: list[str],
) -> pd.DataFrame:
    future = df[["date", *regressors]].rename(columns={"date": "ds"})
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def tune_horizon_params(
    train_df: pd.DataFrame,
    horizon: int,
    regressors: list[str],
) -> dict:
    fit_df, val_df = split_train_validation(train_df)
    target_col = f"target_day{horizon}"
    best_params = None
    best_score = float("inf")

    for params in PARAM_GRID:
        model = fit_horizon_model(fit_df, horizon, regressors, params)
        forecast = predict_horizon(model, val_df, regressors)
        actual = val_df[["date", target_col]].rename(columns={target_col: "y"})
        merged = actual.merge(forecast, left_on="date", right_on="ds", how="inner")
        if merged.empty:
            continue
        score = regression_metrics(merged["y"].values, merged["yhat"].values)["mape"]
        if score < best_score:
            best_score = score
            best_params = params

    return best_params or {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "seasonality_mode": "multiplicative",
    }


def log_dagshub_metrics(prefix: str, metrics: dict) -> None:
    """Log only the metric names used across the team on DagsHub."""
    for metric_name in DAGSHUB_METRICS:
        mlflow.log_metric(f"{prefix}_{metric_name}", metrics[metric_name])


def save_forecast_plot(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    horizon: int,
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(actual_df["date"], actual_df["y"], label="Actual", linewidth=1.5)
    plt.plot(forecast_df["ds"], forecast_df["yhat"], label="Forecast", linewidth=1.5)
    plt.fill_between(
        forecast_df["ds"],
        forecast_df["yhat_lower"],
        forecast_df["yhat_upper"],
        alpha=0.2,
        label="90% interval",
    )
    plt.title(f"Prophet {horizon}-day ahead demand forecast")
    plt.xlabel("Date")
    plt.ylabel("Energy demand (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_training(run_name: str = "prophet_3day_forecast") -> dict:
    model_dir = cfg.get_model_dir("prophet")
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_datasets()
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    regressors = get_regressor_columns(train_df)

    results: dict = {
        "horizons": {},
        "regressors": regressors,
        "tuned_params": {},
    }
    artifact_dir = cfg.REPORT_DIR / "prophet"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    horizon_predictions: dict[int, pd.DataFrame] = {}

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "prophet")
        mlflow.log_param("forecast_horizons", FORECAST_HORIZONS)
        mlflow.log_param("regressor_count", len(regressors))
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("validation_days", VALIDATION_DAYS)
        mlflow.log_param("us_holidays", True)
        mlflow.log_param("feature_strategy", "original_best")
        mlflow.log_param("regressors", ", ".join(regressors))

        for horizon in FORECAST_HORIZONS:
            tuned_params = tune_horizon_params(train_df, horizon, regressors)
            results["tuned_params"][horizon] = tuned_params
            for key, value in tuned_params.items():
                mlflow.log_param(f"day{horizon}_{key}", value)

            target_col = f"target_day{horizon}"
            model = fit_horizon_model(train_df, horizon, regressors, tuned_params)

            model_path = model_dir / f"prophet_day{horizon}.pkl"
            with model_path.open("wb") as handle:
                pickle.dump(model, handle)

            forecast = predict_horizon(model, test_df, regressors)
            actual = test_df[["date", target_col]].rename(columns={target_col: "y"})
            merged = actual.merge(forecast, left_on="date", right_on="ds", how="inner")
            metrics = regression_metrics(merged["y"].values, merged["yhat"].values)
            results["horizons"][horizon] = metrics
            horizon_predictions[horizon] = merged[["date", "y", "yhat"]].copy()

            log_dagshub_metrics(f"day{horizon}", metrics)

            plot_path = artifact_dir / f"forecast_day{horizon}.png"
            save_forecast_plot(actual, forecast, horizon, plot_path)
            mlflow.log_artifact(str(plot_path))

            forecast_path = artifact_dir / f"predictions_day{horizon}.csv"
            merged.to_csv(forecast_path, index=False)
            mlflow.log_artifact(str(forecast_path))
            mlflow.log_artifact(str(model_path))

        predictions_path = cfg.REPORT_DIR / "prophet_predictions.csv"
        save_combined_predictions_csv(horizon_predictions, predictions_path)
        mlflow.log_artifact(str(predictions_path))

        avg_metrics = {
            metric: float(
                sum(results["horizons"][h][metric] for h in FORECAST_HORIZONS)
                / len(FORECAST_HORIZONS)
            )
            for metric in DAGSHUB_METRICS
        }
        results["average"] = avg_metrics
        log_dagshub_metrics("avg", avg_metrics)

        summary_path = artifact_dir / "prophet_results.json"
        summary_path.write_text(json.dumps(results, indent=2))
        mlflow.log_artifact(str(summary_path))

    return results


def main() -> None:
    results = run_training()
    print("Prophet training complete.")
    for horizon, metrics in results["horizons"].items():
        print(
            f"day{horizon}_mae={metrics['mae']:.0f}, "
            f"day{horizon}_rmse={metrics['rmse']:.0f}, "
            f"day{horizon}_mape={metrics['mape']:.2f}%"
        )
    avg = results["average"]
    print(
        f"avg_mae={avg['mae']:.0f}, avg_rmse={avg['rmse']:.0f}, "
        f"avg_mape={avg['mape']:.2f}%"
    )
    print("MLflow tracking URI:", mlflow.get_tracking_uri())


if __name__ == "__main__":
    main()
