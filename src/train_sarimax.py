import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

import src.config as cfg
from src.evaluate import regression_metrics
from src.sarimax_features import (
    DEFAULT_FEATURES,
    SARIMAX_ORDER,
    SEASONAL_ORDER,
    get_exog_columns,
    load_datasets,
    prepare_horizon_series,
)

warnings.filterwarnings("ignore")

FORECAST_HORIZONS = [1, 2, 3]
DAGSHUB_METRICS = ["mae", "mape", "rmse"]


def fit_sarimax(
    endog: pd.Series,
    exog: pd.DataFrame,
    order: tuple = SARIMAX_ORDER,
    seasonal_order: tuple = SEASONAL_ORDER,
):
    model = SARIMAX(
        endog,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=200)


def forecast_horizon(
    fit_result,
    exog_future: pd.DataFrame,
) -> pd.Series:
    forecast = fit_result.get_forecast(steps=len(exog_future), exog=exog_future)
    return forecast.predicted_mean


def in_sample_predict(fit_result, exog: pd.DataFrame) -> pd.Series:
    return fit_result.fittedvalues


def log_dagshub_metrics(prefix: str, metrics: dict) -> None:
    for metric_name in DAGSHUB_METRICS:
        mlflow.log_metric(f"{prefix}_{metric_name}", metrics[metric_name])


def save_plot(
    dates: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="Actual", linewidth=1.5)
    plt.plot(dates, y_pred, label="Forecast", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Energy demand (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_split(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    horizon: int,
    exog_columns: list[str],
    split_name: str,
) -> tuple[dict, pd.DataFrame, object]:
    endog_train, exog_train = prepare_horizon_series(train_df, horizon, exog_columns)
    endog_eval, exog_eval = prepare_horizon_series(eval_df, horizon, exog_columns)

    fit_result = fit_sarimax(endog_train, exog_train)

    if split_name == "train_insample":
        preds = in_sample_predict(fit_result, exog_eval)
        preds = preds.reindex(endog_eval.index)
    else:
        preds = forecast_horizon(fit_result, exog_eval)
        preds.index = endog_eval.index

    aligned = pd.DataFrame(
        {"date": endog_eval.index, "y": endog_eval.values, "yhat": preds.values}
    ).dropna()

    metrics = regression_metrics(aligned["y"].values, aligned["yhat"].values)
    return metrics, aligned, fit_result


def run_training(
    run_name: str = "sarimax_baseline",
    feature_set: list[str] | None = None,
) -> dict:
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_datasets()
    exog_columns = get_exog_columns(train_df, feature_set)

    results: dict = {
        "horizons": {},
        "train_insample": {},
        "exog_columns": exog_columns,
        "order": SARIMAX_ORDER,
        "seasonal_order": SEASONAL_ORDER,
    }
    artifact_dir = cfg.REPORT_DIR / "sarimax"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "SARIMAX")
        mlflow.log_param("order", str(SARIMAX_ORDER))
        mlflow.log_param("seasonal_order", str(SEASONAL_ORDER))
        mlflow.log_param("features", str(exog_columns))
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("test_size", len(test_df))
        mlflow.log_param("eval_split", "test_out_of_sample")

        for horizon in FORECAST_HORIZONS:
            # Out-of-sample test evaluation (same split as Prophet / LightGBM)
            test_metrics, test_aligned, fit_result = evaluate_split(
                train_df, test_df, horizon, exog_columns, "test"
            )
            results["horizons"][horizon] = test_metrics

            train_metrics, _, _ = evaluate_split(
                train_df, train_df, horizon, exog_columns, "train_insample"
            )
            results["train_insample"][horizon] = train_metrics

            log_dagshub_metrics(f"day{horizon}", test_metrics)

            model_path = cfg.MODEL_DIR / f"sarimax_day{horizon}.pkl"
            with model_path.open("wb") as handle:
                pickle.dump(fit_result, handle)

            pred_path = artifact_dir / f"predictions_day{horizon}.csv"
            test_aligned.to_csv(pred_path, index=False)
            mlflow.log_artifact(str(pred_path))

            plot_path = artifact_dir / f"forecast_day{horizon}.png"
            save_plot(
                test_aligned["date"],
                test_aligned["y"],
                test_aligned["yhat"],
                f"SARIMAX day +{horizon} — test set",
                plot_path,
            )
            mlflow.log_artifact(str(plot_path))
            mlflow.log_artifact(str(model_path))

            mlflow.log_metric(f"day{horizon}_train_mape", train_metrics["mape"])

        avg_metrics = {
            metric: float(
                sum(results["horizons"][h][metric] for h in FORECAST_HORIZONS)
                / len(FORECAST_HORIZONS)
            )
            for metric in DAGSHUB_METRICS
        }
        train_avg = {
            metric: float(
                sum(results["train_insample"][h][metric] for h in FORECAST_HORIZONS)
                / len(FORECAST_HORIZONS)
            )
            for metric in DAGSHUB_METRICS
        }
        results["average"] = avg_metrics
        results["train_insample_average"] = train_avg

        log_dagshub_metrics("avg", avg_metrics)
        mlflow.log_metric("train_avg_mape", train_avg["mape"])

        summary_path = artifact_dir / "sarimax_results.json"
        summary_path.write_text(json.dumps(results, indent=2, default=str))
        mlflow.log_artifact(str(summary_path))

    return results


def main() -> None:
    results = run_training()
    print("SARIMAX training complete.\n")

    print("=== OUT-OF-SAMPLE TEST (fair comparison with Prophet / LightGBM) ===")
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

    print("\n=== IN-SAMPLE TRAIN (explains teammate's low DagsHub numbers) ===")
    for horizon, metrics in results["train_insample"].items():
        print(f"day{horizon}_train_mape={metrics['mape']:.2f}%")
    train_avg = results["train_insample_average"]
    print(f"train_avg_mape={train_avg['mape']:.2f}%")

    print("\nMLflow tracking URI:", mlflow.get_tracking_uri())


if __name__ == "__main__":
    main()
