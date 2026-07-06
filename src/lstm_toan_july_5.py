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
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


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

EPOCHS = 40
BATCH_SIZE = 64 # Batch size for training
DROPOUT_RATE = 0.1  # Dropout rate for LSTM layers
EARLY_STOPPING_PATIENCE = 8 # Number of epochs with no improvement after which training will be stopped
LOSS_FUNCTION_NAME = "mse"
  
PERMUTATION_SAMPLE_SIZE = 500
RUN_PERMUTATION_IMPORTANCE = False
RUN_NAME = "lstm_Toan_dual_input_future_features_mse_sigmoid_output"
TARGET_COL = "Demand"
TIME_COL = "datetime"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ENCODER_FEATURES = [
    "Demand",

    # weather nonlinear
    "temperature_2m",
    "temp_squared",
    "cooling_degree",
    "heating_degree",

    # time
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "holiday_encoded",

    # lag / rolling
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_48h",
    "demand_lag_72h",
    "demand_lag_168h",
    "demand_rolling_24h_mean",
    "demand_max_24h",
    "demand_max_168h",

    # economic
    "NYNGSP",
    "NYPOP",
]


FUTURE_FEATURES = [
    # future weather forecast
    "temperature_2m",
    "temp_squared",
    "cooling_degree",
    "heating_degree",

    # future calendar
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "holiday_encoded",
]


FEATURES = list(dict.fromkeys(ENCODER_FEATURES + FUTURE_FEATURES))


def inverse_transform_target_sequences(target_scaler, sequences):
    reshaped = np.asarray(sequences).reshape(-1, 1)

    restored = target_scaler.inverse_transform(reshaped)[:, 0]

    return restored.reshape(np.asarray(sequences).shape)


def create_sequences(data_encoder, data_future, target):
    min_required_rows = WINDOW_SIZE + FORECAST_HORIZON

    if len(target) < min_required_rows:
        raise ValueError(
            "Not enough rows to create sequences for the current "
            f"WINDOW_SIZE={WINDOW_SIZE} and FORECAST_HORIZON={FORECAST_HORIZON}. "
            f"Need at least {min_required_rows} rows, got {len(target)}."
        )

    X_encoder = []
    X_future = []
    y = []

    for i in range(WINDOW_SIZE, len(target) - FORECAST_HORIZON + 1):
        X_encoder.append(data_encoder[i - WINDOW_SIZE:i])
        X_future.append(data_future[i:i + FORECAST_HORIZON])
        y.append(target[i:i + FORECAST_HORIZON])

    return np.array(X_encoder), np.array(X_future), np.array(y)


def build_lstm_model(encoder_shape, future_shape):
    encoder_input = Input(shape=encoder_shape, name="encoder_input")

    encoder_output = LSTM(
        128,
        activation="tanh",
        name="encoder_lstm",
    )(encoder_input)

    repeated = RepeatVector(
        FORECAST_HORIZON,
        name="repeat_encoder_state",
    )(encoder_output)

    future_input = Input(shape=future_shape, name="future_input")

    decoder_input = Concatenate(name="decoder_context")(
        [
            repeated,
            future_input,
        ]
    )

    decoder = LSTM(
        64,
        activation="tanh",
        return_sequences=True,
        name="decoder_lstm",
    )(decoder_input)

    decoder = Dropout(
        DROPOUT_RATE,
        name="decoder_dropout",
    )(decoder)

    output = TimeDistributed(
        Dense(1, activation="sigmoid"),
        name="forecast_output",
    )(decoder)

    model = Model(
        inputs=[
            encoder_input,
            future_input,
        ],
        outputs=output,
    )

    model.compile(
        optimizer=Adam(
            learning_rate=0.001,
            clipnorm=1.0,
        ),
        loss=LOSS_FUNCTION_NAME,
        metrics=["mae"],
    )

    return model


def validate_sequence_alignment(name, input_df, X_encoder, X_future, y):
    first_target_idx = WINDOW_SIZE
    first_target_end_idx = WINDOW_SIZE + FORECAST_HORIZON - 1
    first_encoder_end_idx = WINDOW_SIZE - 1

    first_encoder_end_time = input_df.loc[first_encoder_end_idx, TIME_COL]
    first_future_start_time = input_df.loc[first_target_idx, TIME_COL]
    first_future_end_time = input_df.loc[first_target_end_idx, TIME_COL]

    expected_future_start_time = first_encoder_end_time + pd.Timedelta(hours=1)

    if first_future_start_time != expected_future_start_time:
        raise ValueError(
            f"{name} sequence alignment error: future starts at "
            f"{first_future_start_time}, expected {expected_future_start_time}."
        )

    if X_future.shape[1] != FORECAST_HORIZON or y.shape[1] != FORECAST_HORIZON:
        raise ValueError(
            f"{name} sequence shape error: X_future={X_future.shape}, y={y.shape}."
        )

    print(f"{name} first encoder ends:", first_encoder_end_time)
    print(
        f"{name} first future/y window:",
        first_future_start_time,
        "to",
        first_future_end_time,
    )

def save_training_loss_plot(history, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
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


def calculate_horizon_mae(y_true, y_pred):
    horizon_mae = []

    for horizon_idx in range(y_true.shape[1]):
        mae = mean_absolute_error(
            y_true[:, horizon_idx],
            y_pred[:, horizon_idx],
        )
        horizon_mae.append(mae)

    return np.asarray(horizon_mae)


def save_first_72h_forecast_plot(y_true, y_pred, output_path):
    horizon = np.arange(1, FORECAST_HORIZON + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(horizon, y_true[0], marker="o", label="Actual")
    plt.plot(horizon, y_pred[0], marker="o", label="Forecast")
    plt.legend()
    plt.title("First Test Sample: 72-Hour LSTM Forecast")
    plt.xlabel("Hours ahead")
    plt.ylabel(TARGET_COL)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_horizon_mae_plot(horizon_mae, output_path):
    horizon = np.arange(1, FORECAST_HORIZON + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(horizon, horizon_mae, marker="o")
    plt.title("LSTM MAE by Forecast Horizon")
    plt.xlabel("Hours ahead")
    plt.ylabel("MAE")
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


def evaluate_mae(model, X_encoder, X_future, y_true, target_scaler, batch_size=64):
    y_pred = model.predict(
        [
            X_encoder,
            X_future,
        ],
        batch_size=batch_size,
        verbose=0,
    )

    y_pred_real = inverse_transform_target_sequences(
        target_scaler,
        y_pred,
    )

    y_true_real = inverse_transform_target_sequences(
        target_scaler,
        y_true,
    )

    mae = mean_absolute_error(
        y_true_real.flatten(),
        y_pred_real.flatten(),
    )

    return mae


def permutation_importance(
    model,
    X_encoder_test,
    X_future_test,
    y_test,
    target_scaler,
    encoder_feature_names,
    future_feature_names,
    sample_size=500,
    batch_size=64,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    n_samples = min(sample_size, len(X_encoder_test))

    sample_idx = rng.choice(
        len(X_encoder_test),
        size=n_samples,
        replace=False,
    )

    X_encoder_sample = X_encoder_test[sample_idx].copy()
    X_future_sample = X_future_test[sample_idx].copy()
    y_sample = y_test[sample_idx].copy()

    baseline_mae = evaluate_mae(
        model=model,
        X_encoder=X_encoder_sample,
        X_future=X_future_sample,
        y_true=y_sample,
        target_scaler=target_scaler,
        batch_size=batch_size,
    )

    print("\n===== PERMUTATION IMPORTANCE =====")
    print(f"Sample size: {n_samples}")
    print(f"Baseline MAE on sampled test set: {baseline_mae:.3f}")
    print("-" * 60)

    results = []
    all_features = (
        [("encoder", idx, name) for idx, name in enumerate(encoder_feature_names)]
        + [("future", idx, name) for idx, name in enumerate(future_feature_names)]
    )

    for feature_number, (feature_group, feature_idx, feature_name) in enumerate(
        all_features,
        start=1,
    ):
        X_encoder_perm = X_encoder_sample.copy()
        X_future_perm = X_future_sample.copy()

        perm_idx = rng.permutation(n_samples)

        if feature_group == "encoder":
            X_encoder_perm[:, :, feature_idx] = X_encoder_perm[perm_idx, :, feature_idx]
        else:
            X_future_perm[:, :, feature_idx] = X_future_perm[perm_idx, :, feature_idx]

        permuted_mae = evaluate_mae(
            model=model,
            X_encoder=X_encoder_perm,
            X_future=X_future_perm,
            y_true=y_sample,
            target_scaler=target_scaler,
            batch_size=batch_size,
        )

        importance = permuted_mae - baseline_mae

        results.append(
            {
                "feature": feature_name,
                "feature_group": feature_group,
                "baseline_mae": baseline_mae,
                "permuted_mae": permuted_mae,
                "importance": importance,
            }
        )

        print(
            f"{feature_number:02d}/{len(all_features)} "
            f"{feature_group:7s} "
            f"{feature_name:35s} "
            f"importance = {importance:.3f}"
        )

    importance_df = (
        pd.DataFrame(results)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df


def log_mlflow_run_metadata(
    df,
    train_df,
    val_df,
    test_df,
    X_train_encoder,
    X_val_encoder,
    X_test_encoder,
):
    params = {
        "data_path": str(DATA_PATH),
        "target_col": TARGET_COL,
        "time_col": TIME_COL,

        "window_size": WINDOW_SIZE,
        "forecast_horizon": FORECAST_HORIZON,

        "train_end_date_config": TRAIN_END_DATE,
        "validation_start_date_config": VALIDATION_START_DATE,
        "validation_end_date_config": VALIDATION_END_DATE,
        "test_start_date_config": TEST_START_DATE,
        "test_end_date_config": TEST_END_DATE,

        "train_start_datetime_actual": str(train_df[TIME_COL].min()),
        "train_end_datetime_actual": str(train_df[TIME_COL].max()),
        "validation_start_datetime_actual": str(val_df[TIME_COL].min()),
        "validation_end_datetime_actual": str(val_df[TIME_COL].max()),
        "test_start_datetime_actual": str(test_df[TIME_COL].min()),
        "test_end_datetime_actual": str(test_df[TIME_COL].max()),

        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "dropout_rate": DROPOUT_RATE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,

        "optimizer": "adam",
        "loss": LOSS_FUNCTION_NAME,
        "output_activation": "sigmoid",

        "encoder_feature_count": len(ENCODER_FEATURES),
        "future_feature_count": len(FUTURE_FEATURES),
        "feature_count": len(FEATURES),
        "row_count_after_dropna": len(df),
        "train_row_count": len(train_df),
        "validation_row_count": len(val_df),
        "test_row_count": len(test_df),

        "train_sequence_count": len(X_train_encoder),
        "validation_sequence_count": len(X_val_encoder),
        "test_sequence_count": len(X_test_encoder),

        "seed": SEED,
        "lstm_architecture": "DualInput_EncoderLSTM128_RepeatVector_FutureFeatures_DecoderLSTM64_TimeDistributedDense",

        "reduce_lr_on_plateau": True,
        "reduce_lr_factor": 0.5,
        "reduce_lr_patience": 3,

        "scaler_fit_on": "train_only",
        "validation_history_context_hours": WINDOW_SIZE,
        "test_history_context_hours": WINDOW_SIZE,

        "permutation_importance_sample_size": PERMUTATION_SAMPLE_SIZE,
    }

    mlflow.log_params(params)
    mlflow.log_text(json.dumps(ENCODER_FEATURES, indent=2), "encoder_features.json")
    mlflow.log_text(json.dumps(FUTURE_FEATURES, indent=2), "future_features.json")
    mlflow.log_text(json.dumps(FEATURES, indent=2), "features.json")


def train_lstm():
    configure_mlflow_tracking()

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    missing_raw_columns = [TARGET_COL, TIME_COL]
    missing_raw_columns = [col for col in missing_raw_columns if col not in df.columns]

    if missing_raw_columns:
        raise KeyError(f"Missing required columns in dataset: {missing_raw_columns}")

    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

    df = df.sort_values(TIME_COL).reset_index(drop=True)
    df["date"] = df[TIME_COL].dt.normalize()

    if not df[TIME_COL].is_monotonic_increasing:
        raise ValueError(f"{TIME_COL} must be sorted in increasing order.")

    missing_features = [feature for feature in FEATURES if feature not in df.columns]

    if missing_features:
        raise KeyError(f"Missing features in dataset: {missing_features}")

    df = df[[TIME_COL, "date"] + FEATURES].copy()

    df = df.dropna().reset_index(drop=True)

    print("Data shape:", df.shape)
    print("Datetime range:", df[TIME_COL].min(), "to", df[TIME_COL].max())

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
    print("Datetime range:", train_df[TIME_COL].min(), "to", train_df[TIME_COL].max())
    print("Rows:", len(train_df))
    print()

    print("VALIDATION")
    print("Datetime range:", val_df[TIME_COL].min(), "to", val_df[TIME_COL].max())
    print("Rows:", len(val_df))
    print()

    print("TEST")
    print("Datetime range:", test_df[TIME_COL].min(), "to", test_df[TIME_COL].max())
    print("Rows:", len(test_df))
    print("=" * 60)

    print("Scaling data...")

    encoder_scaler = MinMaxScaler()
    future_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_encoder_scaled = encoder_scaler.fit_transform(train_df[ENCODER_FEATURES])
    train_future_scaled = future_scaler.fit_transform(train_df[FUTURE_FEATURES])
    train_target_scaled = target_scaler.fit_transform(train_df[[TARGET_COL]]).reshape(-1)

    val_input_df = pd.concat(
        [train_df.tail(WINDOW_SIZE), val_df],
        ignore_index=True,
    )

    val_encoder_scaled = encoder_scaler.transform(val_input_df[ENCODER_FEATURES])
    val_future_scaled = future_scaler.transform(val_input_df[FUTURE_FEATURES])
    val_target_scaled = target_scaler.transform(val_input_df[[TARGET_COL]]).reshape(-1)

    test_input_df = pd.concat(
        [val_df.tail(WINDOW_SIZE), test_df],
        ignore_index=True,
    )

    test_encoder_scaled = encoder_scaler.transform(test_input_df[ENCODER_FEATURES])
    test_future_scaled = future_scaler.transform(test_input_df[FUTURE_FEATURES])
    test_target_scaled = target_scaler.transform(test_input_df[[TARGET_COL]]).reshape(-1)

    print("Creating training sequences...")
    X_train_encoder, X_train_future, y_train = create_sequences(
        train_encoder_scaled,
        train_future_scaled,
        train_target_scaled,
    )

    print("Creating validation sequences...")
    X_val_encoder, X_val_future, y_val = create_sequences(
        val_encoder_scaled,
        val_future_scaled,
        val_target_scaled,
    )

    print("Creating testing sequences...")
    X_test_encoder, X_test_future, y_test = create_sequences(
        test_encoder_scaled,
        test_future_scaled,
        test_target_scaled,
    )

    validate_sequence_alignment(
        "TRAIN",
        train_df.reset_index(drop=True),
        X_train_encoder,
        X_train_future,
        y_train,
    )
    validate_sequence_alignment(
        "VALIDATION",
        val_input_df,
        X_val_encoder,
        X_val_future,
        y_val,
    )
    validate_sequence_alignment(
        "TEST",
        test_input_df,
        X_test_encoder,
        X_test_future,
        y_test,
    )

    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    y_val = y_val.reshape((y_val.shape[0], y_val.shape[1], 1))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))

    print("Train encoder X shape:", X_train_encoder.shape)
    print("Train future X shape:", X_train_future.shape)
    print("Train y shape:", y_train.shape)
    print("Validation encoder X shape:", X_val_encoder.shape)
    print("Validation future X shape:", X_val_future.shape)
    print("Validation y shape:", y_val.shape)
    print("Test encoder X shape :", X_test_encoder.shape)
    print("Test future X shape :", X_test_future.shape)
    print("Test y shape :", y_test.shape)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        log_mlflow_run_metadata(
            df=df,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            X_train_encoder=X_train_encoder,
            X_val_encoder=X_val_encoder,
            X_test_encoder=X_test_encoder,
        )

        print("Building model...")
        model = build_lstm_model(
            encoder_shape=(WINDOW_SIZE, len(ENCODER_FEATURES)),
            future_shape=(FORECAST_HORIZON, len(FUTURE_FEATURES)),
        )
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
            [
                X_train_encoder,
                X_train_future,
            ],
            y_train,
            validation_data=(
                [
                    X_val_encoder,
                    X_val_future,
                ],
                y_val,
            ),
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
        y_pred_scaled = model.predict(
            [
                X_test_encoder,
                X_test_future,
            ],
            batch_size=BATCH_SIZE,
            verbose=1,
        )

        y_pred_scaled = y_pred_scaled.squeeze(-1)
        y_test_scaled = y_test.squeeze(-1)

        pred_scaled_min = float(np.min(y_pred_scaled))
        pred_scaled_max = float(np.max(y_pred_scaled))
        print(f"Scaled prediction min/max: {pred_scaled_min:.4f} / {pred_scaled_max:.4f}")

        y_pred_real = inverse_transform_target_sequences(target_scaler, y_pred_scaled)
        y_test_real = inverse_transform_target_sequences(target_scaler, y_test_scaled)

        y_pred_flat = y_pred_real.reshape(-1)
        y_test_flat = y_test_real.reshape(-1)

        mae = mean_absolute_error(y_test_flat, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
        mape, excluded_zeros = calculate_mape(y_test_flat, y_pred_flat)
        extreme_metrics = calculate_extreme_metrics(y_test_flat, y_pred_flat)
        horizon_mae = calculate_horizon_mae(y_test_real, y_pred_real)

        print("\n===== TEST RESULTS =====")
        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")

        if np.isfinite(mape):
            print(f"MAPE : {mape:.2f}%")
        else:
            print("MAPE : not available")

        print(f"MAPE excluded zero-demand points: {excluded_zeros}")
        print(f"Horizon MAE mean: {np.mean(horizon_mae):.2f}")
        print(f"Horizon MAE first hour: {horizon_mae[0]:.2f}")
        print(f"Horizon MAE last hour : {horizon_mae[-1]:.2f}")
        print_extreme_metrics("MIN/MAX CHECK", extreme_metrics)

        mlflow.log_metrics(
            {
                "test_mae": float(mae),
                "test_rmse": float(rmse),
                "test_excluded_zero_demand_points": int(excluded_zeros),
                "test_horizon_mae_mean": float(np.mean(horizon_mae)),
                "test_horizon_mae_first_hour": float(horizon_mae[0]),
                "test_horizon_mae_last_hour": float(horizon_mae[-1]),
                "test_pred_scaled_min": pred_scaled_min,
                "test_pred_scaled_max": pred_scaled_max,
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
        first_72h_plot_path = PLOT_DIR / "lstm_first_72h_forecast.png"
        horizon_mae_plot_path = PLOT_DIR / "lstm_horizon_mae.png"
        horizon_mae_csv_path = PLOT_DIR / "lstm_horizon_mae.csv"

        save_training_loss_plot(history, training_loss_plot_path)
        save_forecast_plot(y_test_flat, y_pred_flat, forecast_plot_path)
        save_first_72h_forecast_plot(y_test_real, y_pred_real, first_72h_plot_path)
        save_horizon_mae_plot(horizon_mae, horizon_mae_plot_path)

        horizon_mae_df = pd.DataFrame(
            {
                "hours_ahead": np.arange(1, FORECAST_HORIZON + 1),
                "mae": horizon_mae,
            }
        )
        horizon_mae_df.to_csv(horizon_mae_csv_path, index=False)

        
        # run permutation importance analysis if enabled
        if RUN_PERMUTATION_IMPORTANCE:
            print("\nRunning permutation importance analysis...")

            importance_df = permutation_importance(
                model=model,
                X_encoder_test=X_test_encoder,
                X_future_test=X_test_future,
                y_test=y_test,
                target_scaler=target_scaler,
                encoder_feature_names=ENCODER_FEATURES,
                future_feature_names=FUTURE_FEATURES,
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
        mlflow.log_artifact(str(first_72h_plot_path), artifact_path="plots")
        mlflow.log_artifact(str(horizon_mae_plot_path), artifact_path="plots")
        mlflow.log_artifact(str(horizon_mae_csv_path), artifact_path="metrics")
        print("MLflow run logged successfully.")


if __name__ == "__main__":
    train_lstm()
