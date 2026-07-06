import json
import pickle
import random
from contextlib import nullcontext

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

import config as cfg
from config import configure_mlflow_tracking

from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is not installed. Install it with: pip install xgboost"
    ) from exc


# ==================================================
# Configuration
# ==================================================
DATA_PATH = cfg.REPORT_DIR / "all_features_dataset.csv"
MODEL_PATH = cfg.MODEL_DIR / "xgboost_toan_72h.pkl"
PLOT_DIR = cfg.REPORT_DIR / "plots"

FORECAST_HORIZON = 72  # 72 hours = 3 days

TRAIN_END_DATE = "2024-04-30"
VALIDATION_START_DATE = "2024-05-01"
VALIDATION_END_DATE = "2025-04-30"
TEST_START_DATE = "2025-05-01"
TEST_END_DATE = "2026-04-30"

TARGET_COL = "Demand"
RUN_NAME = "xgboost_Toan_direct_72h_forecast"
SEED = 42
ENABLE_MLFLOW = False
SCRIPT_VERSION = "sequential_72h_xgboost_v2"

N_ESTIMATORS = 250
MAX_DEPTH = 4
LEARNING_RATE = 0.03
SUBSAMPLE = 0.9
COLSAMPLE_BYTREE = 0.9
REG_LAMBDA = 1.0
TREE_METHOD = "hist"
N_JOBS = 4

random.seed(SEED)
np.random.seed(SEED)

DROP_COLUMNS = [
    "datetime",
    "date",
    "day_name",
    "month_name",
    "holiday",
]


def get_time_column(df):
    if "datetime" in df.columns:
        return "datetime"
    if "date" in df.columns:
        return "date"
    raise KeyError("Dataset must contain either 'datetime' or 'date'.")


def load_dataset():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    time_col = get_time_column(df)

    required_columns = [TARGET_COL, time_col]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise KeyError(f"Missing required columns in dataset: {missing_columns}")

    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)

    print("Raw data shape:", df.shape)
    print("Date range:", df[time_col].min(), "to", df[time_col].max())

    return df, time_col


def select_feature_columns(df):
    feature_df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns])

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != TARGET_COL]

    if not feature_cols:
        raise ValueError("No numeric feature columns found for XGBoost.")

    return feature_cols


def create_direct_forecast_dataset(df, feature_cols, time_col):
    working_df = df[[time_col, TARGET_COL] + feature_cols].copy()

    for col in [TARGET_COL] + feature_cols:
        working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    target_df = pd.concat(
        [
            working_df[TARGET_COL]
            .shift(-horizon)
            .rename(f"target_t_plus_{horizon:02d}")
            for horizon in range(1, FORECAST_HORIZON + 1)
        ],
        axis=1,
    )

    y_cols = target_df.columns.tolist()
    working_df = pd.concat([working_df, target_df], axis=1)

    working_df = working_df.dropna().reset_index(drop=True)

    X = working_df[feature_cols].to_numpy(dtype=np.float32)
    y = working_df[y_cols].to_numpy(dtype=np.float32)
    forecast_times = working_df[time_col].reset_index(drop=True)

    return X, y, forecast_times, working_df


def split_by_date(X, y, forecast_times):
    train_end = pd.to_datetime(TRAIN_END_DATE)
    val_start = pd.to_datetime(VALIDATION_START_DATE)
    val_end = pd.to_datetime(VALIDATION_END_DATE)
    test_start = pd.to_datetime(TEST_START_DATE)
    test_end = pd.to_datetime(TEST_END_DATE)

    forecast_dates = pd.to_datetime(forecast_times).dt.normalize()

    train_mask = forecast_dates <= train_end
    val_mask = (forecast_dates >= val_start) & (forecast_dates <= val_end)
    test_mask = (forecast_dates >= test_start) & (forecast_dates <= test_end)

    splits = {
        "train": (X[train_mask], y[train_mask], forecast_times[train_mask]),
        "validation": (X[val_mask], y[val_mask], forecast_times[val_mask]),
        "test": (X[test_mask], y[test_mask], forecast_times[test_mask]),
    }

    for split_name, (split_X, split_y, split_times) in splits.items():
        if len(split_X) == 0:
            raise ValueError(f"{split_name} split is empty. Check date ranges.")

        print()
        print(split_name.upper())
        print("Date range:", split_times.min(), "to", split_times.max())
        print("X shape:", split_X.shape)
        print("y shape:", split_y.shape)

    return splits


def build_xgboost_estimator():
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        reg_lambda=REG_LAMBDA,
        random_state=SEED,
        tree_method=TREE_METHOD,
        n_jobs=N_JOBS,
        eval_metric="rmse",
        verbosity=0,
    )


def train_horizon_models(X_train, y_train):
    models = []

    for horizon_idx in range(FORECAST_HORIZON):
        print(
            f"Training horizon {horizon_idx + 1:02d}/{FORECAST_HORIZON}...",
            flush=True,
        )

        model = build_xgboost_estimator()
        model.fit(X_train, y_train[:, horizon_idx])
        models.append(model)

    return models


def predict_horizon_models(models, X):
    predictions = np.zeros((len(X), len(models)), dtype=np.float32)

    for horizon_idx, model in enumerate(models):
        predictions[:, horizon_idx] = model.predict(X)

    return predictions


def calculate_mape(y_true, y_pred, epsilon=1e-6):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    nonzero_mask = np.abs(y_true) > epsilon
    excluded_zeros = np.size(y_true) - np.count_nonzero(nonzero_mask)

    if np.any(nonzero_mask):
        mape = np.mean(
            np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])
        ) * 100
    else:
        mape = np.nan

    return mape, excluded_zeros


def calculate_extreme_metrics(y_true, y_pred):
    true_min = float(np.min(y_true))
    pred_min = float(np.min(y_pred))
    true_max = float(np.max(y_true))
    pred_max = float(np.max(y_pred))
    true_range = true_max - true_min
    pred_range = pred_max - pred_min

    if true_range > 0:
        range_coverage = pred_range / true_range
    else:
        range_coverage = np.nan

    return {
        "actual_min": true_min,
        "pred_min": pred_min,
        "min_gap": pred_min - true_min,
        "actual_max": true_max,
        "pred_max": pred_max,
        "max_gap": pred_max - true_max,
        "actual_range": true_range,
        "pred_range": pred_range,
        "range_coverage": range_coverage,
    }


def evaluate_predictions(y_true, y_pred, prefix):
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mape, excluded_zeros = calculate_mape(y_true_flat, y_pred_flat)
    extreme_metrics = calculate_extreme_metrics(y_true_flat, y_pred_flat)

    print()
    print(f"===== {prefix.upper()} RESULTS =====")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

    if np.isfinite(mape):
        print(f"MAPE : {mape:.2f}%")
    else:
        print("MAPE : not available")

    print(f"MAPE excluded zero-demand points: {excluded_zeros}")
    print(
        "Min actual/pred/gap: "
        f"{extreme_metrics['actual_min']:.2f} / "
        f"{extreme_metrics['pred_min']:.2f} / "
        f"{extreme_metrics['min_gap']:.2f}"
    )
    print(
        "Max actual/pred/gap: "
        f"{extreme_metrics['actual_max']:.2f} / "
        f"{extreme_metrics['pred_max']:.2f} / "
        f"{extreme_metrics['max_gap']:.2f}"
    )
    print(f"Range coverage: {extreme_metrics['range_coverage']:.3f}")

    metrics = {
        f"{prefix}_mae": float(mae),
        f"{prefix}_rmse": float(rmse),
        f"{prefix}_excluded_zero_demand_points": int(excluded_zeros),
    }

    if np.isfinite(mape):
        metrics[f"{prefix}_mape"] = float(mape)

    for key, value in extreme_metrics.items():
        if np.isfinite(value):
            metrics[f"{prefix}_{key}"] = float(value)

    return metrics


def save_forecast_plot(y_true, y_pred, output_path, limit=500):
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    plt.figure(figsize=(15, 5))
    plt.plot(y_true_flat[:limit], label="Actual")
    plt.plot(y_pred_flat[:limit], label="Forecast")
    plt.legend()
    plt.title("XGBoost Actual vs Forecast on Test Set")
    plt.xlabel("Forecasted point")
    plt.ylabel(TARGET_COL)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_first_72h_plot(y_true, y_pred, output_path):
    horizon = np.arange(1, FORECAST_HORIZON + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(horizon, y_true[0], marker="o", label="Actual")
    plt.plot(horizon, y_pred[0], marker="o", label="Forecast")
    plt.legend()
    plt.title("First Test Sample: 72-Hour XGBoost Forecast")
    plt.xlabel("Hours ahead")
    plt.ylabel(TARGET_COL)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_feature_importance_plot(models, feature_cols, output_path):
    importances = np.zeros(len(feature_cols), dtype=np.float64)

    for model in models:
        importances += model.feature_importances_

    importances = importances / len(models)

    importance_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    top_df = importance_df.head(25).sort_values("importance")

    plt.figure(figsize=(10, 8))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Mean feature importance across 72 horizon models")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return importance_df


def log_mlflow_params(feature_cols, dataset_rows, split_sizes):
    params = {
        "data_path": str(DATA_PATH),
        "target_col": TARGET_COL,
        "forecast_horizon_hours": FORECAST_HORIZON,
        "forecast_horizon_days": FORECAST_HORIZON / 24,
        "train_end_date_config": TRAIN_END_DATE,
        "validation_start_date_config": VALIDATION_START_DATE,
        "validation_end_date_config": VALIDATION_END_DATE,
        "test_start_date_config": TEST_START_DATE,
        "test_end_date_config": TEST_END_DATE,
        "model_type": "direct_72_XGBRegressor_models",
        "objective": "reg:squarederror",
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        "reg_lambda": REG_LAMBDA,
        "tree_method": TREE_METHOD,
        "n_jobs_per_model": N_JOBS,
        "feature_count": len(feature_cols),
        "row_count_after_dropna": dataset_rows,
        "train_sequence_count": split_sizes["train"],
        "validation_sequence_count": split_sizes["validation"],
        "test_sequence_count": split_sizes["test"],
        "seed": SEED,
    }

    mlflow.log_params(params)
    mlflow.log_text(json.dumps(feature_cols, indent=2), "features.json")


def train_xgboost_72h():
    print(f"Running {SCRIPT_VERSION}", flush=True)

    if ENABLE_MLFLOW:
        configure_mlflow_tracking()

    df, time_col = load_dataset()
    feature_cols = select_feature_columns(df)

    print()
    print("Selected feature count:", len(feature_cols))
    print("Creating direct 72-hour forecast dataset...")

    X, y, forecast_times, supervised_df = create_direct_forecast_dataset(
        df=df,
        feature_cols=feature_cols,
        time_col=time_col,
    )

    print("Supervised data shape:", supervised_df.shape)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    splits = split_by_date(X, y, forecast_times)

    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["validation"]
    X_test, y_test, _ = splits["test"]

    run_context = mlflow.start_run(run_name=RUN_NAME) if ENABLE_MLFLOW else nullcontext()

    with run_context:
        split_sizes = {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        }

        if ENABLE_MLFLOW:
            log_mlflow_params(
                feature_cols=feature_cols,
                dataset_rows=len(supervised_df),
                split_sizes=split_sizes,
            )

        print()
        print("Training XGBoost direct 72-hour forecast models...")
        models = train_horizon_models(X_train, y_train)

        print("Predicting validation set...")
        y_val_pred = predict_horizon_models(models, X_val)

        print("Predicting test set...")
        y_test_pred = predict_horizon_models(models, X_test)

        val_metrics = evaluate_predictions(y_val, y_val_pred, "validation")
        test_metrics = evaluate_predictions(y_test, y_test_pred, "test")

        if ENABLE_MLFLOW:
            mlflow.log_metrics(val_metrics)
            mlflow.log_metrics(test_metrics)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        with open(MODEL_PATH, "wb") as file:
            pickle.dump(
                {
                    "models": models,
                    "feature_cols": feature_cols,
                    "forecast_horizon": FORECAST_HORIZON,
                    "target_col": TARGET_COL,
                },
                file,
            )

        forecast_plot_path = PLOT_DIR / "xgboost_toan_test_actual_vs_forecast.png"
        first_72h_plot_path = PLOT_DIR / "xgboost_toan_first_72h_forecast.png"
        importance_plot_path = PLOT_DIR / "xgboost_toan_feature_importance.png"
        importance_csv_path = PLOT_DIR / "xgboost_toan_feature_importance.csv"

        save_forecast_plot(y_test, y_test_pred, forecast_plot_path)
        save_first_72h_plot(y_test, y_test_pred, first_72h_plot_path)

        importance_df = save_feature_importance_plot(
            models=models,
            feature_cols=feature_cols,
            output_path=importance_plot_path,
        )

        importance_df.to_csv(importance_csv_path, index=False)

        if ENABLE_MLFLOW:
            mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
            mlflow.log_artifact(str(forecast_plot_path), artifact_path="plots")
            mlflow.log_artifact(str(first_72h_plot_path), artifact_path="plots")
            mlflow.log_artifact(str(importance_plot_path), artifact_path="plots")
            mlflow.log_artifact(str(importance_csv_path), artifact_path="feature_importance")

        print()
        print(f"Model saved: {MODEL_PATH}")
        print(f"Forecast plot saved: {forecast_plot_path}")
        print(f"First 72h plot saved: {first_72h_plot_path}")
        print(f"Feature importance saved: {importance_csv_path}")
        if ENABLE_MLFLOW:
            print("MLflow run logged successfully.")
        else:
            print("MLflow disabled. Set ENABLE_MLFLOW = True to log this run.")


if __name__ == "__main__":
    train_xgboost_72h()
