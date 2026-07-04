import json
import random

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import config as cfg
from config import configure_mlflow_tracking

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


# ==================================================
# Configuration
# ==================================================
DATA_PATH = cfg.REPORT_DIR / "all_features_dataset.csv"
MODEL_PATH = cfg.MODEL_DIR / "lstm_model.keras"
PLOT_DIR = cfg.REPORT_DIR / "plots"

WINDOW_SIZE = 24 * 7
FORECAST_HORIZON = 72

TRAIN_END_DATE = "2024-04-30"
VALIDATION_START_DATE = "2024-05-01"
VALIDATION_END_DATE = "2025-04-30"
TEST_START_DATE = "2025-05-01"
TEST_END_DATE = "2026-04-30"

EPOCHS = 20
BATCH_SIZE = 64 # Batch size for training
DROPOUT_RATE = 0.2  # Dropout rate for LSTM layers
EARLY_STOPPING_PATIENCE = 5 # Number of epochs with no improvement after which training will be stopped
EXTREME_LOSS_WEIGHT = 4.0  # Weight for high/low demand values far from the scaled center 0.5
UNDER_PREDICTION_WEIGHT = 3.0  # Extra penalty when the model predicts lower than actual demand
LOSS_FUNCTION_NAME = "asymmetric_peak_mse"
  
PERMUTATION_SAMPLE_SIZE = 500
RUN_PERMUTATION_IMPORTANCE = False
RUN_NAME = "lstm_Toan_asymmetric_peak_mse_loss_for_peak_forecasting"
TARGET_COL = "Demand"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

FEATURES = [
    "Demand",
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "wind_speed_10m",
    "snowfall",
    "precipitation",
    "cloud_cover",
    "holiday_encoded",

    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",

    "demand_lag_1h",
    "demand_lag_2h",
    "demand_lag_3h",
    "demand_lag_4h",
    "demand_lag_24h",
    "demand_lag_48h",
    "demand_lag_72h",
    "demand_lag_168h",

    "demand_rolling_24h_mean",
    "demand_rolling_48h_mean",
    "demand_rolling_72h_mean",
    "demand_rolling_168h_mean",

    "demand_std_24h",
    "demand_std_48h",
    "demand_std_72h",
    "demand_std_168h",

    "demand_min_24h",
    "demand_min_48h",
    "demand_min_72h",
    "demand_min_168h",

    "demand_max_24h",
    "demand_max_48h",
    "demand_max_72h",
    "demand_max_168h",

    "NYNGSP",
    "NYPOP",
]






def inverse_transform_target_sequences(scaler, sequences):
    reshaped = np.asarray(sequences).reshape(-1, 1)

    dummy = np.zeros((reshaped.shape[0], len(FEATURES)))
    dummy[:, 0] = reshaped[:, 0]

    restored = scaler.inverse_transform(dummy)[:, 0]

    return restored.reshape(np.asarray(sequences).shape)


def create_sequences(scaled_data):
    min_required_rows = WINDOW_SIZE + FORECAST_HORIZON

    if len(scaled_data) < min_required_rows:
        raise ValueError(
            "Not enough rows to create sequences for the current "
            f"WINDOW_SIZE={WINDOW_SIZE} and FORECAST_HORIZON={FORECAST_HORIZON}. "
            f"Need at least {min_required_rows} rows, got {len(scaled_data)}."
        )

    X = []
    y = []

    for i in range(WINDOW_SIZE, len(scaled_data) - FORECAST_HORIZON + 1):
        X.append(scaled_data[i - WINDOW_SIZE:i])
        y.append(scaled_data[i:i + FORECAST_HORIZON, 0])

    return np.array(X), np.array(y)


def build_lstm_model():
    model = Sequential(
        [
            Input(shape=(WINDOW_SIZE, len(FEATURES))),

            LSTM(
                units=128,
                return_sequences=True,
            ),
            Dropout(DROPOUT_RATE),

            LSTM(
                units=64,
                return_sequences=False,
            ),
            Dropout(DROPOUT_RATE),

            Dense(FORECAST_HORIZON),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss=asymmetric_peak_mse_loss,
    )

    return model

#==================================================
# Custom Loss Functions
# y_true and y_pred are scaled by MinMaxScaler, so target Demand is usually in [0, 1].
# weighted_peak_mse_loss:
#   - uses MSE instead of Huber, so large peak errors are penalized strongly.
#   - increases loss weight when y_true is far from 0.5, meaning high peaks and low valleys.
# asymmetric_peak_mse_loss:
#   - adds one more penalty when prediction is lower than actual demand.
#   - this is useful when forecast peaks are consistently lower than actual peaks.
#=================================================
def weighted_peak_mse_loss(y_true, y_pred):
    error = y_true - y_pred
    mse = tf.square(error)

    extreme_distance = tf.abs(y_true - 0.5) * 2.0
    extreme_weights = 1.0 + EXTREME_LOSS_WEIGHT * extreme_distance

    return tf.reduce_mean(extreme_weights * mse)


def asymmetric_peak_mse_loss(y_true, y_pred):
    error = y_true - y_pred
    mse = tf.square(error)

    extreme_distance = tf.abs(y_true - 0.5) * 2.0
    extreme_weights = 1.0 + EXTREME_LOSS_WEIGHT * extreme_distance

    under_prediction = tf.cast(error > 0, tf.float32)
    under_prediction_weights = 1.0 + UNDER_PREDICTION_WEIGHT * under_prediction

    return tf.reduce_mean(
        extreme_weights * under_prediction_weights * mse
    )


def save_training_loss_plot(history, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_forecast_plot(y_test_flat, y_pred_flat, output_path, limit=500):
    plt.figure(figsize=(15, 5))
    plt.plot(y_test_flat[:limit], label="Actual")
    plt.plot(y_pred_flat[:limit], label="Forecast")
    plt.legend()
    plt.title("Actual vs Forecast on Test Set")
    plt.xlabel("Forecasted point")
    plt.ylabel(TARGET_COL)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_permutation_importance_plot(importance_df, output_path):
    plt.figure(figsize=(10, 10))

    plt.barh(
        importance_df["feature"],
        importance_df["importance"],
    )

    plt.xlabel("Increase in MAE")
    plt.title("Permutation Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def calculate_mape(y_true, y_pred, epsilon=1e-6):
    nonzero_mask = np.abs(y_true) > epsilon
    excluded_zeros = np.size(y_true) - np.count_nonzero(nonzero_mask)

    if np.any(nonzero_mask):
        mape = (
            np.mean(
                np.abs(
                    (y_true[nonzero_mask] - y_pred[nonzero_mask])
                    / y_true[nonzero_mask]
                )
            )
            * 100
        )
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
        "test_actual_min": true_min,
        "test_pred_min": pred_min,
        "test_min_gap": pred_min - true_min,
        "test_actual_max": true_max,
        "test_pred_max": pred_max,
        "test_max_gap": pred_max - true_max,
        "test_actual_range": true_range,
        "test_pred_range": pred_range,
        "test_range_coverage": range_coverage,
    }


def print_extreme_metrics(title, metrics):
    print(f"\n===== {title} =====")
    print(
        "Min  actual/pred/gap: "
        f"{metrics['test_actual_min']:.2f} / "
        f"{metrics['test_pred_min']:.2f} / "
        f"{metrics['test_min_gap']:.2f}"
    )
    print(
        "Max  actual/pred/gap: "
        f"{metrics['test_actual_max']:.2f} / "
        f"{metrics['test_pred_max']:.2f} / "
        f"{metrics['test_max_gap']:.2f}"
    )
    print(
        "Range coverage: "
        f"{metrics['test_range_coverage']:.3f}"
    )


def evaluate_mae(model, X, y_true, scaler, batch_size=64):
    y_pred = model.predict(
        X,
        batch_size=batch_size,
        verbose=0,
    )

    y_pred_real = inverse_transform_target_sequences(
        scaler,
        y_pred,
    )

    y_true_real = inverse_transform_target_sequences(
        scaler,
        y_true,
    )

    mae = mean_absolute_error(
        y_true_real.flatten(),
        y_pred_real.flatten(),
    )

    return mae


def permutation_importance(
    model,
    X_test,
    y_test,
    scaler,
    feature_names,
    sample_size=500,
    batch_size=64,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    n_samples = min(sample_size, len(X_test))

    sample_idx = rng.choice(
        len(X_test),
        size=n_samples,
        replace=False,
    )

    X_sample = X_test[sample_idx].copy()
    y_sample = y_test[sample_idx].copy()

    baseline_mae = evaluate_mae(
        model=model,
        X=X_sample,
        y_true=y_sample,
        scaler=scaler,
        batch_size=batch_size,
    )

    print("\n===== PERMUTATION IMPORTANCE =====")
    print(f"Sample size: {n_samples}")
    print(f"Baseline MAE on sampled test set: {baseline_mae:.3f}")
    print("-" * 60)

    results = []

    for feature_idx, feature_name in enumerate(feature_names):
        X_perm = X_sample.copy()

        perm_idx = rng.permutation(n_samples)

        X_perm[:, :, feature_idx] = X_perm[perm_idx, :, feature_idx]

        permuted_mae = evaluate_mae(
            model=model,
            X=X_perm,
            y_true=y_sample,
            scaler=scaler,
            batch_size=batch_size,
        )

        importance = permuted_mae - baseline_mae

        results.append(
            {
                "feature": feature_name,
                "baseline_mae": baseline_mae,
                "permuted_mae": permuted_mae,
                "importance": importance,
            }
        )

        print(
            f"{feature_idx + 1:02d}/{len(feature_names)} "
            f"{feature_name:35s} "
            f"importance = {importance:.3f}"
        )

    importance_df = (
        pd.DataFrame(results)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df


def log_mlflow_run_metadata(df, train_df, val_df, test_df, X_train, X_val, X_test):
    params = {
        "data_path": str(DATA_PATH),
        "target_col": TARGET_COL,

        "window_size": WINDOW_SIZE,
        "forecast_horizon": FORECAST_HORIZON,

        "train_end_date_config": TRAIN_END_DATE,
        "validation_start_date_config": VALIDATION_START_DATE,
        "validation_end_date_config": VALIDATION_END_DATE,
        "test_start_date_config": TEST_START_DATE,
        "test_end_date_config": TEST_END_DATE,

        "train_start_date_actual": str(train_df["date"].min()),
        "train_end_date_actual": str(train_df["date"].max()),
        "validation_start_date_actual": str(val_df["date"].min()),
        "validation_end_date_actual": str(val_df["date"].max()),
        "test_start_date_actual": str(test_df["date"].min()),
        "test_end_date_actual": str(test_df["date"].max()),

        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "dropout_rate": DROPOUT_RATE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,

        "optimizer": "adam",
        "loss": LOSS_FUNCTION_NAME,
        "extreme_loss_weight": EXTREME_LOSS_WEIGHT,
        "under_prediction_weight": UNDER_PREDICTION_WEIGHT,

        "feature_count": len(FEATURES),
        "row_count_after_dropna": len(df),
        "train_row_count": len(train_df),
        "validation_row_count": len(val_df),
        "test_row_count": len(test_df),

        "train_sequence_count": len(X_train),
        "validation_sequence_count": len(X_val),
        "test_sequence_count": len(X_test),

        "seed": SEED,
        "lstm_architecture": "Input_LSTM128_LSTM64_Dense",

        "reduce_lr_on_plateau": True,
        "reduce_lr_factor": 0.5,
        "reduce_lr_patience": 3,

        "scaler_fit_on": "train_only",
        "validation_history_context_hours": WINDOW_SIZE,
        "test_history_context_hours": WINDOW_SIZE,

        "permutation_importance_sample_size": PERMUTATION_SAMPLE_SIZE,
    }

    mlflow.log_params(params)
    mlflow.log_text(json.dumps(FEATURES, indent=2), "features.json")


def train_lstm():
    configure_mlflow_tracking()

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    missing_raw_columns = [TARGET_COL, "date"]
    missing_raw_columns = [col for col in missing_raw_columns if col not in df.columns]

    if missing_raw_columns:
        raise KeyError(f"Missing required columns in dataset: {missing_raw_columns}")

    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)

    missing_features = [feature for feature in FEATURES if feature not in df.columns]

    if missing_features:
        raise KeyError(f"Missing features in dataset: {missing_features}")

    df = df[["date"] + FEATURES].copy()

    df = df.dropna().reset_index(drop=True)

    print("Data shape:", df.shape)
    print("Date range:", df["date"].min(), "to", df["date"].max())

    print("Splitting data by date...")

    train_end = pd.to_datetime(TRAIN_END_DATE)
    val_start = pd.to_datetime(VALIDATION_START_DATE)
    val_end = pd.to_datetime(VALIDATION_END_DATE)
    test_start = pd.to_datetime(TEST_START_DATE)
    test_end = pd.to_datetime(TEST_END_DATE)

    train_df = df[df["date"] <= train_end].copy()

    val_df = df[
        (df["date"] >= val_start)
        & (df["date"] <= val_end)
    ].copy()

    test_df = df[
        (df["date"] >= test_start)
        & (df["date"] <= test_end)
    ].copy()

    if train_df.empty:
        raise ValueError("Training set is empty. Check TRAIN_END_DATE.")

    if val_df.empty:
        raise ValueError("Validation set is empty. Check validation date range.")

    if test_df.empty:
        raise ValueError("Test set is empty. Check test date range.")

    print("=" * 60)
    print("TRAIN")
    print("Date range:", train_df["date"].min(), "to", train_df["date"].max())
    print("Rows:", len(train_df))
    print()

    print("VALIDATION")
    print("Date range:", val_df["date"].min(), "to", val_df["date"].max())
    print("Rows:", len(val_df))
    print()

    print("TEST")
    print("Date range:", test_df["date"].min(), "to", test_df["date"].max())
    print("Rows:", len(test_df))
    print("=" * 60)

    print("Scaling data...")

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_df[FEATURES])

    val_input_df = pd.concat(
        [train_df.tail(WINDOW_SIZE), val_df],
        ignore_index=True,
    )

    val_scaled = scaler.transform(val_input_df[FEATURES])

    test_input_df = pd.concat(
        [val_df.tail(WINDOW_SIZE), test_df],
        ignore_index=True,
    )

    test_scaled = scaler.transform(test_input_df[FEATURES])

    print("Creating training sequences...")
    X_train, y_train = create_sequences(train_scaled)

    print("Creating validation sequences...")
    X_val, y_val = create_sequences(val_scaled)

    print("Creating testing sequences...")
    X_test, y_test = create_sequences(test_scaled)

    print("Train X shape:", X_train.shape)
    print("Train y shape:", y_train.shape)
    print("Validation X shape:", X_val.shape)
    print("Validation y shape:", y_val.shape)
    print("Test X shape :", X_test.shape)
    print("Test y shape :", y_test.shape)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        log_mlflow_run_metadata(
            df=df,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
        )

        print("Building model...")
        model = build_lstm_model()
        model.summary()

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        )

        print("Training...")
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            shuffle=False,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop, reduce_lr],
            verbose=1,
        )

        for step, loss in enumerate(history.history["loss"]):
            mlflow.log_metric("train_loss", float(loss), step=step)

        for step, val_loss in enumerate(history.history["val_loss"]):
            mlflow.log_metric("val_loss", float(val_loss), step=step)

        print("Predicting on test set...")
        y_pred = model.predict(
            X_test,
            batch_size=BATCH_SIZE,
            verbose=1,
        )

        y_pred_real = inverse_transform_target_sequences(scaler, y_pred)
        y_test_real = inverse_transform_target_sequences(scaler, y_test)

        y_pred_flat = y_pred_real.reshape(-1)
        y_test_flat = y_test_real.reshape(-1)

        mae = mean_absolute_error(y_test_flat, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
        mape, excluded_zeros = calculate_mape(y_test_flat, y_pred_flat)
        extreme_metrics = calculate_extreme_metrics(y_test_flat, y_pred_flat)

        print("\n===== TEST RESULTS =====")
        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")

        if np.isfinite(mape):
            print(f"MAPE : {mape:.2f}%")
        else:
            print("MAPE : not available")

        print(f"MAPE excluded zero-demand points: {excluded_zeros}")
        print_extreme_metrics("MIN/MAX CHECK", extreme_metrics)

        mlflow.log_metrics(
            {
                "test_mae": float(mae),
                "test_rmse": float(rmse),
                "test_excluded_zero_demand_points": int(excluded_zeros),
            }
        )

        mlflow.log_metrics(
            {
                key: float(value)
                for key, value in extreme_metrics.items()
                if np.isfinite(value)
            }
        )

        if np.isfinite(mape):
            mlflow.log_metric("test_mape", float(mape))
        else:
            mlflow.log_param("test_mape_status", "not_available_all_targets_are_zero")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        model.save(MODEL_PATH)
        print(f"\nModel saved: {MODEL_PATH}")

        training_loss_plot_path = PLOT_DIR / "lstm_training_loss.png"
        forecast_plot_path = PLOT_DIR / "lstm_test_actual_vs_forecast.png"

        save_training_loss_plot(history, training_loss_plot_path)
        save_forecast_plot(y_test_flat, y_pred_flat, forecast_plot_path)

        
        # run permutation importance analysis if enabled
        if RUN_PERMUTATION_IMPORTANCE:
            print("\nRunning permutation importance analysis...")

            importance_df = permutation_importance(
                model=model,
                X_test=X_test,
                y_test=y_test,
                scaler=scaler,
                feature_names=FEATURES,
                sample_size=PERMUTATION_SAMPLE_SIZE,
                batch_size=BATCH_SIZE,
                random_state=SEED,
            )
            print("\n===== TOP FEATURE IMPORTANCE =====")
            print(importance_df.head(20))

            importance_csv_path = PLOT_DIR / "permutation_importance.csv"
            importance_plot_path = PLOT_DIR / "permutation_importance.png"

            importance_df.to_csv(
                importance_csv_path,
                index=False,
                )

            save_permutation_importance_plot(
                importance_df,
                importance_plot_path,
                )

            # End of permutation importance analysis
            # Log permutation importance results to MLflow

            mlflow.log_artifact(
                str(importance_csv_path),
                artifact_path="feature_importance",
            )

            mlflow.log_artifact(
                str(importance_plot_path),
                artifact_path="feature_importance",
            )

            mlflow.log_text(
                importance_df.to_json(orient="records", indent=2),
                "feature_importance/permutation_importance.json",
            )

            print(f"\nPermutation importance CSV saved: {importance_csv_path}")
            print(f"Permutation importance plot saved: {importance_plot_path}")

            # End of permutation importance analysis logging

        # Log artifacts to MLflow
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        mlflow.log_artifact(str(training_loss_plot_path), artifact_path="plots")
        mlflow.log_artifact(str(forecast_plot_path), artifact_path="plots")
        print("MLflow run logged successfully.")


if __name__ == "__main__":
    train_lstm()
