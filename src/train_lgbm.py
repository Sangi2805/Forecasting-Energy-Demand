import json
import pickle
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

import src.config as cfg
from src.evaluate import regression_metrics
from src.lgbm_features import (
    get_feature_columns,
    load_featured_datasets,
    prepare_xy,
)

FORECAST_HORIZONS = [1, 2, 3]
DAGSHUB_METRICS = ["mae", "mape", "rmse"]
VALIDATION_DAYS = 365
MAX_ESTIMATORS = 3000
EARLY_STOPPING_ROUNDS = 150

PARAM_GRID = [
    {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20, "reg_alpha": 0.0, "reg_lambda": 0.1, "max_depth": -1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 0.1, "max_depth": -1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 15, "reg_alpha": 0.1, "reg_lambda": 1.0, "max_depth": 12, "subsample": 0.85, "colsample_bytree": 0.85},
    {"num_leaves": 127, "learning_rate": 0.03, "min_child_samples": 10, "reg_alpha": 0.1, "reg_lambda": 1.0, "max_depth": 10, "subsample": 0.9, "colsample_bytree": 0.8},
    {"num_leaves": 31, "learning_rate": 0.1, "min_child_samples": 20, "reg_alpha": 0.0, "reg_lambda": 0.0, "max_depth": 8, "subsample": 0.8, "colsample_bytree": 0.9},
    {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 30, "reg_alpha": 1.0, "reg_lambda": 1.0, "max_depth": -1, "subsample": 0.75, "colsample_bytree": 0.75},
    {"num_leaves": 95, "learning_rate": 0.04, "min_child_samples": 20, "reg_alpha": 0.5, "reg_lambda": 0.5, "max_depth": 11, "subsample": 0.85, "colsample_bytree": 0.8},
    {"num_leaves": 47, "learning_rate": 0.06, "min_child_samples": 25, "reg_alpha": 0.2, "reg_lambda": 0.3, "max_depth": 9, "subsample": 0.8, "colsample_bytree": 0.85},
]

BASE_LGBM_PARAMS = {
    "objective": "regression_l1",
    "metric": "l1",
    "verbosity": -1,
    "n_estimators": MAX_ESTIMATORS,
    "random_state": 42,
    "n_jobs": -1,
}


def split_train_validation(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = train_df["date"].max() - pd.Timedelta(days=VALIDATION_DAYS)
    fit_df = train_df[train_df["date"] <= cutoff].copy()
    val_df = train_df[train_df["date"] > cutoff].copy()
    return fit_df, val_df


def build_model(params: dict, n_estimators: int | None = None) -> lgb.LGBMRegressor:
    model_params = {**BASE_LGBM_PARAMS, **params}
    if n_estimators is not None:
        model_params["n_estimators"] = n_estimators
    return lgb.LGBMRegressor(**model_params)


def fit_with_early_stopping(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    horizon: int,
    feature_columns: list[str],
    params: dict,
) -> lgb.LGBMRegressor:
    x_train, y_train = prepare_xy(train_df, horizon, feature_columns)
    x_val, y_val = prepare_xy(val_df, horizon, feature_columns)
    model = build_model(params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def fit_final_model(
    train_df: pd.DataFrame,
    horizon: int,
    feature_columns: list[str],
    params: dict,
    n_estimators: int,
) -> lgb.LGBMRegressor:
    x_train, y_train = prepare_xy(train_df, horizon, feature_columns)
    model = build_model(params, n_estimators=n_estimators)
    model.fit(x_train, y_train)
    return model


def log_dagshub_metrics(prefix: str, metrics: dict) -> None:
    for metric_name in DAGSHUB_METRICS:
        mlflow.log_metric(f"{prefix}_{metric_name}", metrics[metric_name])


def save_forecast_plot(
    dates: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
    horizon: int,
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="Actual", linewidth=1.5)
    plt.plot(dates, y_pred, label="Forecast", linewidth=1.5)
    plt.title(f"LightGBM {horizon}-day ahead demand forecast")
    plt.xlabel("Date")
    plt.ylabel("Energy demand (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_importance_plot(model: lgb.LGBMRegressor, horizon: int, output_path: Path) -> None:
    importance = pd.Series(
        model.feature_importances_,
        index=model.feature_name_,
    ).sort_values(ascending=True).tail(15)

    plt.figure(figsize=(10, 6))
    importance.plot(kind="barh")
    plt.title(f"LightGBM top features — day +{horizon}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def tune_horizon(
    train_df: pd.DataFrame,
    horizon: int,
    feature_columns: list[str],
) -> tuple[dict, int]:
    fit_df, val_df = split_train_validation(train_df)
    best_params = PARAM_GRID[0]
    best_score = float("inf")
    best_iteration = 500

    for params in PARAM_GRID:
        model = fit_with_early_stopping(fit_df, val_df, horizon, feature_columns, params)
        x_val, y_val = prepare_xy(val_df, horizon, feature_columns)
        preds = model.predict(x_val)
        score = regression_metrics(y_val.values, preds)["mape"]
        iteration = model.best_iteration_ or MAX_ESTIMATORS
        if score < best_score:
            best_score = score
            best_params = params
            best_iteration = iteration

    return best_params, best_iteration


def run_training(run_name: str = "lgbm_3day_forecast") -> dict:
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_featured_datasets()
    feature_columns = get_feature_columns(train_df)

    results: dict = {
        "horizons": {},
        "feature_columns": feature_columns,
        "tuned_params": {},
        "best_iterations": {},
    }
    artifact_dir = cfg.REPORT_DIR / "lgbm"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "lightgbm")
        mlflow.log_param("forecast_horizons", FORECAST_HORIZONS)
        mlflow.log_param("feature_count", len(feature_columns))
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("validation_days", VALIDATION_DAYS)
        mlflow.log_param("param_grid_size", len(PARAM_GRID))
        mlflow.log_param("objective", "regression_l1")

        for horizon in FORECAST_HORIZONS:
            tuned_params, best_iteration = tune_horizon(train_df, horizon, feature_columns)
            results["tuned_params"][horizon] = tuned_params
            results["best_iterations"][horizon] = best_iteration

            for key, value in tuned_params.items():
                mlflow.log_param(f"day{horizon}_{key}", value)
            mlflow.log_param(f"day{horizon}_best_iteration", best_iteration)

            model = fit_final_model(
                train_df, horizon, feature_columns, tuned_params, best_iteration
            )

            model_path = cfg.MODEL_DIR / f"lgbm_day{horizon}.pkl"
            with model_path.open("wb") as handle:
                pickle.dump(model, handle)

            x_test, y_test = prepare_xy(test_df, horizon, feature_columns)
            dates = test_df.loc[y_test.index, "date"]
            preds = model.predict(x_test)
            metrics = regression_metrics(y_test.values, preds)
            results["horizons"][horizon] = metrics

            log_dagshub_metrics(f"day{horizon}", metrics)

            predictions = pd.DataFrame(
                {"date": dates.values, "y": y_test.values, "yhat": preds}
            )
            pred_path = artifact_dir / f"predictions_day{horizon}.csv"
            predictions.to_csv(pred_path, index=False)
            mlflow.log_artifact(str(pred_path))

            plot_path = artifact_dir / f"forecast_day{horizon}.png"
            save_forecast_plot(dates, y_test, preds, horizon, plot_path)
            mlflow.log_artifact(str(plot_path))

            importance_path = artifact_dir / f"importance_day{horizon}.png"
            save_importance_plot(model, horizon, importance_path)
            mlflow.log_artifact(str(importance_path))
            mlflow.log_artifact(str(model_path))

        avg_metrics = {
            metric: float(
                sum(results["horizons"][h][metric] for h in FORECAST_HORIZONS)
                / len(FORECAST_HORIZONS)
            )
            for metric in DAGSHUB_METRICS
        }
        results["average"] = avg_metrics
        log_dagshub_metrics("avg", avg_metrics)

        summary_path = artifact_dir / "lgbm_results.json"
        summary_path.write_text(json.dumps(results, indent=2))
        mlflow.log_artifact(str(summary_path))

    return results


def main() -> None:
    results = run_training()
    print("LightGBM training complete.")
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
