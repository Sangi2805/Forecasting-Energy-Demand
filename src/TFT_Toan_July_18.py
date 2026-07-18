import json
import random
from contextlib import nullcontext

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

import config as cfg
from config import configure_mlflow_tracking

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Dense,
    Dropout,
    Input,
    LSTM,
    LayerNormalization,
    MultiHeadAttention,
    TimeDistributed,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# ==================================================
# Configuration
# ==================================================
DATA_PATH = cfg.REPORT_DIR / "all_features_dataset.csv"
MODEL_PATH = cfg.MODEL_DIR / "tft_quantile_model.keras"
PLOT_DIR = cfg.REPORT_DIR / "plots"

WINDOW_SIZE = 24 * 7
FORECAST_BLOCK_SIZE = 24
FORECAST_BLOCK_COUNT = 5
FORECAST_HORIZON = FORECAST_BLOCK_SIZE * FORECAST_BLOCK_COUNT

TRAIN_START_DATE = "2023-05-01"
TRAIN_END_DATE = "2024-04-30"
VALIDATION_START_DATE = "2024-05-01"
VALIDATION_END_DATE = "2025-04-30"
TEST_START_DATE = "2025-05-01"
TEST_END_DATE = "2026-04-30"

EPOCHS = 40
BATCH_SIZE = 64
HIDDEN_SIZE = 96
ATTENTION_HEADS = 4
DROPOUT_RATE = 0.15
EARLY_STOPPING_PATIENCE = 8
QUANTILES = [0.1, 0.5, 0.9]
QUANTILE_COLUMNS = ["q10", "q50", "q90"]
MEDIAN_QUANTILE_INDEX = QUANTILES.index(0.5)
LOSS_FUNCTION_NAME = "quantile_loss_q10_q50_q90"

RUN_NAME = f"TFT_Toan_July_18_{LOSS_FUNCTION_NAME}"
USE_MLFLOW = True
TARGET_COL = "Demand"
TIME_COL = "datetime"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Known features are available for the whole forecast horizon.
# This includes weather because data_collection.read_weather_data can fetch
# forecast weather from Open-Meteo for the future window.
KNOWN_FEATURES = [
    # weather forecast / weather observations
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    #"rain",
    "snowfall",
    #"cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "temp_squared",
    "cooling_degree",
    "heating_degree",
    "consecutive_hot_hours",
    "rolling_max_temperature_24h",
    "rolling_mean_temperature_24h",
    "temperature_anomaly",

    # calendar / scheduled features
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "holiday_encoded",
    "time_index",

    # annual macro features known for historical backtests
    "NYNGSP",
    "NYPOP",
]


# Unknown features are only available up to the forecast creation time.
# Future values of these columns depend on future demand, so they are used
# only in the encoder/history branch.
UNKNOWN_FEATURES = [
    TARGET_COL,
    "demand_lag_1h",
    #"demand_lag_2h",
    #"demand_lag_3h",
    #"demand_lag_4h",
    "demand_lag_24h",
    "demand_lag_48h",
    "demand_lag_72h",
    "demand_lag_168h",
    "demand_rolling_24h_mean",
    "demand_rolling_48h_mean",
    "demand_rolling_72h_mean",
    "demand_rolling_168h_mean",
    "demand_std_24h",
    #"demand_std_48h",
    #"demand_std_72h",
    #"demand_std_168h",
    "demand_min_24h",
    "demand_min_48h",
    #"demand_min_72h",
    #"demand_min_168h",
    "demand_max_24h",
    #"demand_max_48h",
    #"demand_max_72h",
    #"demand_max_168h",
]


@tf.keras.utils.register_keras_serializable(package="custom_losses")
def quantile_loss(y_true, y_pred):
    quantiles = tf.constant(QUANTILES, dtype=y_pred.dtype)
    error = y_true - y_pred
    loss = tf.maximum(quantiles * error, (quantiles - 1.0) * error)
    return tf.reduce_mean(loss)


@tf.keras.utils.register_keras_serializable(package="custom_metrics")
def median_mae(y_true, y_pred):
    y_pred_median = y_pred[..., MEDIAN_QUANTILE_INDEX:MEDIAN_QUANTILE_INDEX + 1]
    return tf.reduce_mean(tf.abs(y_true - y_pred_median))


def build_tft_model():
    encoder_known_input = Input(
        shape=(WINDOW_SIZE, len(KNOWN_FEATURES)),
        name="encoder_known_features",
    )
    encoder_unknown_input = Input(
        shape=(WINDOW_SIZE, len(UNKNOWN_FEATURES)),
        name="encoder_unknown_features",
    )
    decoder_known_input = Input(
        shape=(FORECAST_HORIZON, len(KNOWN_FEATURES)),
        name="decoder_known_features",
    )

    encoder_known = TimeDistributed(
        Dense(HIDDEN_SIZE, activation="elu"),
        name="encoder_known_projection",
    )(encoder_known_input)
    encoder_unknown = TimeDistributed(
        Dense(HIDDEN_SIZE, activation="elu"),
        name="encoder_unknown_projection",
    )(encoder_unknown_input)
    decoder_known = TimeDistributed(
        Dense(HIDDEN_SIZE, activation="elu"),
        name="decoder_known_projection",
    )(decoder_known_input)

    encoder_features = Concatenate(name="encoder_variable_concat")(
        [encoder_known, encoder_unknown]
    )
    encoder_features = TimeDistributed(
        Dense(HIDDEN_SIZE, activation="elu"),
        name="encoder_variable_selection",
    )(encoder_features)
    encoder_features = Dropout(DROPOUT_RATE)(encoder_features)

    encoder_sequence, state_h, state_c = LSTM(
        HIDDEN_SIZE,
        return_sequences=True,
        return_state=True,
        name="encoder_lstm",
    )(encoder_features)

    decoder_sequence = LSTM(
        HIDDEN_SIZE,
        return_sequences=True,
        name="decoder_lstm",
    )(decoder_known, initial_state=[state_h, state_c])

    attention_output = MultiHeadAttention(
        num_heads=ATTENTION_HEADS,
        key_dim=HIDDEN_SIZE // ATTENTION_HEADS,
        dropout=DROPOUT_RATE,
        name="temporal_self_attention",
    )(
        query=decoder_sequence,
        value=encoder_sequence,
        key=encoder_sequence,
    )

    decoder_context = Add(name="attention_residual")(
        [decoder_sequence, attention_output]
    )
    decoder_context = LayerNormalization(name="attention_norm")(decoder_context)

    grn_hidden = TimeDistributed(
        Dense(HIDDEN_SIZE, activation="elu"),
        name="forecast_grn_dense_1",
    )(decoder_context)
    grn_hidden = Dropout(DROPOUT_RATE)(grn_hidden)
    grn_hidden = TimeDistributed(
        Dense(HIDDEN_SIZE),
        name="forecast_grn_dense_2",
    )(grn_hidden)
    grn_gate = TimeDistributed(
        Dense(HIDDEN_SIZE, activation="sigmoid"),
        name="forecast_grn_gate",
    )(decoder_context)
    grn_output = tf.keras.layers.Multiply(name="forecast_grn_gated")(
        [grn_hidden, grn_gate]
    )
    grn_output = Add(name="forecast_grn_residual")([decoder_context, grn_output])
    grn_output = LayerNormalization(name="forecast_grn_norm")(grn_output)

    output = TimeDistributed(
        Dense(len(QUANTILES)),
        name="demand_quantile_forecast",
    )(grn_output)

    model = Model(
        inputs=[encoder_known_input, encoder_unknown_input, decoder_known_input],
        outputs=output,
        name="tft_known_unknown_demand_forecaster",
    )

    optimizer = Adam(
        learning_rate=0.001,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss=quantile_loss,
        metrics=[median_mae],
    )

    return model


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run src/preprocessing.py first."
        )

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

    missing_features = [
        feature
        for feature in KNOWN_FEATURES + UNKNOWN_FEATURES
        if feature not in df.columns
    ]
    if missing_features:
        raise KeyError(f"Missing features in dataset: {missing_features}")

    selected_columns = [TIME_COL, "date"] + KNOWN_FEATURES + UNKNOWN_FEATURES
    df = df[selected_columns].copy()

    df = df.dropna().reset_index(drop=True)

    if not df[TIME_COL].is_monotonic_increasing:
        raise ValueError(f"{TIME_COL} must be sorted in increasing order.")

    return df


def fit_transform_data(df):
    train_start = pd.to_datetime(TRAIN_START_DATE)
    train_end = pd.to_datetime(TRAIN_END_DATE)
    train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()

    if train_df.empty:
        raise ValueError("Training set is empty. Check TRAIN_END_DATE.")

    known_scaler = MinMaxScaler()
    unknown_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    known_scaler.fit(train_df[KNOWN_FEATURES])
    unknown_scaler.fit(train_df[UNKNOWN_FEATURES])
    target_scaler.fit(train_df[[TARGET_COL]])

    known_scaled = known_scaler.transform(df[KNOWN_FEATURES])
    unknown_scaled = unknown_scaler.transform(df[UNKNOWN_FEATURES])
    target_scaled = target_scaler.transform(df[[TARGET_COL]]).reshape(-1)

    return known_scaled, unknown_scaled, target_scaled, known_scaler, unknown_scaler, target_scaler


def create_tft_sequences(df, known_scaled, unknown_scaled, target_scaled):
    min_required_rows = WINDOW_SIZE + FORECAST_HORIZON

    if len(df) < min_required_rows:
        raise ValueError(
            "Not enough rows to create sequences for the current "
            f"WINDOW_SIZE={WINDOW_SIZE} and FORECAST_HORIZON={FORECAST_HORIZON}. "
            f"Need at least {min_required_rows} rows, got {len(df)}."
        )

    encoder_known = []
    encoder_unknown = []
    decoder_known = []
    targets = []
    target_start_dates = []
    target_start_times = []

    for target_start_idx in range(WINDOW_SIZE, len(df) - FORECAST_HORIZON + 1):
        encoder_start_idx = target_start_idx - WINDOW_SIZE
        target_end_idx = target_start_idx + FORECAST_HORIZON

        encoder_known.append(known_scaled[encoder_start_idx:target_start_idx])
        encoder_unknown.append(unknown_scaled[encoder_start_idx:target_start_idx])
        decoder_known.append(known_scaled[target_start_idx:target_end_idx])
        targets.append(target_scaled[target_start_idx:target_end_idx])

        target_start_dates.append(df.loc[target_start_idx, "date"])
        target_start_times.append(df.loc[target_start_idx, TIME_COL])

    X = {
        "encoder_known_features": np.asarray(encoder_known, dtype=np.float32),
        "encoder_unknown_features": np.asarray(encoder_unknown, dtype=np.float32),
        "decoder_known_features": np.asarray(decoder_known, dtype=np.float32),
    }
    y = np.asarray(targets, dtype=np.float32).reshape(-1, FORECAST_HORIZON, 1)
    metadata = pd.DataFrame(
        {
            "target_start_datetime": target_start_times,
            "target_start_date": target_start_dates,
        }
    )

    return X, y, metadata


def subset_sequences(X, y, metadata, mask):
    mask = np.asarray(mask)
    return (
        {
            key: value[mask]
            for key, value in X.items()
        },
        y[mask],
        metadata.loc[mask].reset_index(drop=True),
    )


def split_sequences(X, y, metadata):
    train_end = pd.to_datetime(TRAIN_END_DATE)
    val_start = pd.to_datetime(VALIDATION_START_DATE)
    val_end = pd.to_datetime(VALIDATION_END_DATE)
    test_start = pd.to_datetime(TEST_START_DATE)
    test_end = pd.to_datetime(TEST_END_DATE)

    dates = metadata["target_start_date"]
    train_mask = dates <= train_end
    val_mask = (dates >= val_start) & (dates <= val_end)
    test_mask = (dates >= test_start) & (dates <= test_end)

    X_train, y_train, train_meta = subset_sequences(X, y, metadata, train_mask)
    X_val, y_val, val_meta = subset_sequences(X, y, metadata, val_mask)
    X_test, y_test, test_meta = subset_sequences(X, y, metadata, test_mask)

    if len(y_train) == 0:
        raise ValueError("Training sequence set is empty. Check TRAIN_END_DATE.")
    if len(y_val) == 0:
        raise ValueError("Validation sequence set is empty. Check validation dates.")
    if len(y_test) == 0:
        raise ValueError("Test sequence set is empty. Check test dates.")

    return X_train, y_train, train_meta, X_val, y_val, val_meta, X_test, y_test, test_meta


def inverse_transform_target_sequences(target_scaler, sequences):
    original_shape = np.asarray(sequences).shape
    reshaped = np.asarray(sequences).reshape(-1, 1)
    restored = target_scaler.inverse_transform(reshaped)
    return restored.reshape(original_shape)


def calculate_horizon_mae(y_true, y_pred):
    horizon_mae = []

    for horizon_idx in range(y_true.shape[1]):
        mae = mean_absolute_error(
            y_true[:, horizon_idx],
            y_pred[:, horizon_idx],
        )
        horizon_mae.append(mae)

    return np.asarray(horizon_mae)


def calculate_24h_block_metrics(y_true, y_pred, block_size=FORECAST_BLOCK_SIZE):
    block_metrics = []

    for start_idx in range(0, y_true.shape[1], block_size):
        end_idx = min(start_idx + block_size, y_true.shape[1])

        y_true_block = y_true[:, start_idx:end_idx].reshape(-1)
        y_pred_block = y_pred[:, start_idx:end_idx].reshape(-1)

        mae = mean_absolute_error(y_true_block, y_pred_block)
        rmse = np.sqrt(mean_squared_error(y_true_block, y_pred_block))
        mape, excluded_zeros = calculate_mape(y_true_block, y_pred_block)

        block_metrics.append(
            {
                "block": f"block_{len(block_metrics) + 1}",
                "hour_range": f"hours_{start_idx + 1}_{end_idx}",
                "start_hour_ahead": start_idx + 1,
                "end_hour_ahead": end_idx,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "mape_excluded_zero_demand_points": excluded_zeros,
            }
        )

    return pd.DataFrame(block_metrics)


def print_24h_block_metrics(block_metrics_df):
    print("\n===== TEST MAE / MAPE BY 24-HOUR BLOCK =====")

    for _, row in block_metrics_df.iterrows():
        if np.isfinite(row["mape"]):
            mape_text = f"{row['mape']:.2f}%"
        else:
            mape_text = "not available"

        print(
            f"{row['block']} ({row['hour_range']}): "
            f"MAE = {row['mae']:.2f}, "
            f"RMSE = {row['rmse']:.2f}, "
            f"MAPE = {mape_text}, "
            "excluded zero-demand points = "
            f"{int(row['mape_excluded_zero_demand_points'])}"
        )


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


def save_training_loss_plot(history, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("TFT Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Quantile Loss")
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


def save_first_forecast_plot(
    y_true,
    y_pred,
    output_path,
    y_pred_lower=None,
    y_pred_upper=None,
):
    horizon = np.arange(1, FORECAST_HORIZON + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(horizon, y_true[0], marker="o", label="Actual")
    if y_pred_lower is not None and y_pred_upper is not None:
        plt.fill_between(
            horizon,
            y_pred_lower[0],
            y_pred_upper[0],
            alpha=0.2,
            label="Q10-Q90 interval",
        )
    plt.plot(horizon, y_pred[0], marker="o", label="Forecast Q50")
    plt.legend()
    plt.title(f"First Test Sample: {FORECAST_HORIZON}-Hour TFT Quantile Forecast")
    plt.xlabel("Hours ahead")
    plt.ylabel(TARGET_COL)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_first_forecast_block_plots(
    y_true,
    y_pred,
    output_dir,
    y_pred_lower=None,
    y_pred_upper=None,
    block_size=FORECAST_BLOCK_SIZE,
):
    block_plot_paths = []

    for block_idx, start_idx in enumerate(range(0, FORECAST_HORIZON, block_size), start=1):
        end_idx = min(start_idx + block_size, FORECAST_HORIZON)
        horizon = np.arange(start_idx + 1, end_idx + 1)
        output_path = output_dir / f"tft_quantile_first_24h_block_{block_idx}_forecast.png"

        plt.figure(figsize=(10, 5))
        plt.plot(horizon, y_true[0, start_idx:end_idx], marker="o", label="Actual")
        if y_pred_lower is not None and y_pred_upper is not None:
            plt.fill_between(
                horizon,
                y_pred_lower[0, start_idx:end_idx],
                y_pred_upper[0, start_idx:end_idx],
                alpha=0.2,
                label="Q10-Q90 interval",
            )
        plt.plot(
            horizon,
            y_pred[0, start_idx:end_idx],
            marker="o",
            label="Forecast Q50",
        )
        plt.legend()
        plt.title(
            "First Test Sample: "
            f"Block {block_idx} Forecast (Hours {start_idx + 1}-{end_idx})"
        )
        plt.xlabel("Hours ahead")
        plt.ylabel(TARGET_COL)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        block_plot_paths.append(output_path)

    return block_plot_paths


def save_horizon_mae_plot(horizon_mae, output_path):
    horizon = np.arange(1, FORECAST_HORIZON + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(horizon, horizon_mae, marker="o")
    plt.title("TFT MAE by Forecast Horizon")
    plt.xlabel("Hours ahead")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def configure_optional_mlflow():
    if not USE_MLFLOW:
        return False

    try:
        configure_mlflow_tracking()
        return True
    except RuntimeError as exc:
        print(f"MLflow tracking not configured: {exc}")
        print("Continuing without MLflow logging.")
        return False


def log_mlflow_run_metadata(df, train_meta, val_meta, test_meta, X_train, X_val, X_test):
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

        "train_start_target_datetime_actual": str(train_meta["target_start_datetime"].min()),
        "train_end_target_datetime_actual": str(train_meta["target_start_datetime"].max()),
        "validation_start_target_datetime_actual": str(val_meta["target_start_datetime"].min()),
        "validation_end_target_datetime_actual": str(val_meta["target_start_datetime"].max()),
        "test_start_target_datetime_actual": str(test_meta["target_start_datetime"].min()),
        "test_end_target_datetime_actual": str(test_meta["target_start_datetime"].max()),

        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "attention_heads": ATTENTION_HEADS,
        "dropout_rate": DROPOUT_RATE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,

        "optimizer": "adam",
        "loss": LOSS_FUNCTION_NAME,
        "quantiles": json.dumps(QUANTILES),
        "median_quantile": QUANTILES[MEDIAN_QUANTILE_INDEX],

        "known_feature_count": len(KNOWN_FEATURES),
        "unknown_feature_count": len(UNKNOWN_FEATURES),
        "row_count_after_dropna": len(df),

        "train_sequence_count": len(X_train["encoder_known_features"]),
        "validation_sequence_count": len(X_val["encoder_known_features"]),
        "test_sequence_count": len(X_test["encoder_known_features"]),

        "seed": SEED,
        "architecture": "keras_tft_style_encoder_decoder_attention",
        "weather_assumption": "weather_features_are_known_future_covariates",
        "scaler_fit_on": "train_only",
    }

    mlflow.log_params(params)
    mlflow.log_text(json.dumps(KNOWN_FEATURES, indent=2), "known_features.json")
    mlflow.log_text(json.dumps(UNKNOWN_FEATURES, indent=2), "unknown_features.json")


def train_tft():
    mlflow_enabled = configure_optional_mlflow()

    print("Loading data...")
    df = load_dataset()

    print("Data shape:", df.shape)
    print("Datetime range:", df[TIME_COL].min(), "to", df[TIME_COL].max())
    print("Known features:", len(KNOWN_FEATURES))
    print("Unknown features:", len(UNKNOWN_FEATURES))

    print("Scaling data with train-only scalers...")
    known_scaled, unknown_scaled, target_scaled, _, _, target_scaler = fit_transform_data(df)

    print("Creating TFT sequences...")
    X, y, metadata = create_tft_sequences(
        df=df,
        known_scaled=known_scaled,
        unknown_scaled=unknown_scaled,
        target_scaled=target_scaled,
    )

    (
        X_train,
        y_train,
        train_meta,
        X_val,
        y_val,
        val_meta,
        X_test,
        y_test,
        test_meta,
    ) = split_sequences(X, y, metadata)

    print("=" * 60)
    print("TRAIN")
    print(
        "Target start range:",
        train_meta["target_start_datetime"].min(),
        "to",
        train_meta["target_start_datetime"].max(),
    )
    print("Sequences:", len(y_train))
    print()

    print("VALIDATION")
    print(
        "Target start range:",
        val_meta["target_start_datetime"].min(),
        "to",
        val_meta["target_start_datetime"].max(),
    )
    print("Sequences:", len(y_val))
    print()

    print("TEST")
    print(
        "Target start range:",
        test_meta["target_start_datetime"].min(),
        "to",
        test_meta["target_start_datetime"].max(),
    )
    print("Sequences:", len(y_test))
    print("=" * 60)

    print("Encoder known shape:", X_train["encoder_known_features"].shape)
    print("Encoder unknown shape:", X_train["encoder_unknown_features"].shape)
    print("Decoder known shape:", X_train["decoder_known_features"].shape)
    print("Target shape:", y_train.shape)

    run_context = mlflow.start_run(run_name=RUN_NAME) if mlflow_enabled else nullcontext()

    with run_context:
        if mlflow_enabled:
            log_mlflow_run_metadata(
                df=df,
                train_meta=train_meta,
                val_meta=val_meta,
                test_meta=test_meta,
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
            )

        print("Building TFT model...")
        model = build_tft_model()
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

        if mlflow_enabled:
            for step, loss in enumerate(history.history["loss"]):
                mlflow.log_metric("train_loss", float(loss), step=step)

            for step, val_loss in enumerate(history.history["val_loss"]):
                mlflow.log_metric("val_loss", float(val_loss), step=step)

        print("Predicting on test set...")
        y_pred_quantiles_scaled = model.predict(
            X_test,
            batch_size=BATCH_SIZE,
            verbose=1,
        )

        y_pred_quantiles_real = inverse_transform_target_sequences(
            target_scaler,
            y_pred_quantiles_scaled,
        )
        y_test_real = inverse_transform_target_sequences(
            target_scaler,
            y_test.squeeze(-1),
        )

        y_pred_q10_real = y_pred_quantiles_real[..., 0]
        y_pred_q50_real = y_pred_quantiles_real[..., MEDIAN_QUANTILE_INDEX]
        y_pred_q90_real = y_pred_quantiles_real[..., 2]
        y_pred_interval_lower_real = np.minimum(y_pred_q10_real, y_pred_q90_real)
        y_pred_interval_upper_real = np.maximum(y_pred_q10_real, y_pred_q90_real)

        y_pred_flat = y_pred_q50_real.reshape(-1)
        y_test_flat = y_test_real.reshape(-1)

        mae = mean_absolute_error(y_test_flat, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
        mape, excluded_zeros = calculate_mape(y_test_flat, y_pred_flat)
        extreme_metrics = calculate_extreme_metrics(y_test_flat, y_pred_flat)
        horizon_mae = calculate_horizon_mae(y_test_real, y_pred_q50_real)
        block_24h_metrics_df = calculate_24h_block_metrics(
            y_test_real,
            y_pred_q50_real,
            block_size=24,
        )

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
        print_24h_block_metrics(block_24h_metrics_df)

        if mlflow_enabled:
            mlflow.log_metrics(
                {
                    "test_mae": float(mae),
                    "test_rmse": float(rmse),
                    "test_excluded_zero_demand_points": int(excluded_zeros),
                    "test_horizon_mae_mean": float(np.mean(horizon_mae)),
                    "test_horizon_mae_first_hour": float(horizon_mae[0]),
                    "test_horizon_mae_last_hour": float(horizon_mae[-1]),
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

            for _, row in block_24h_metrics_df.iterrows():
                block_name = row["block"]
                mlflow.log_metric(f"test_{block_name}_mae", float(row["mae"]))
                mlflow.log_metric(f"test_{block_name}_rmse", float(row["rmse"]))

                if np.isfinite(row["mape"]):
                    mlflow.log_metric(f"test_{block_name}_mape", float(row["mape"]))
                else:
                    mlflow.log_param(f"test_{block_name}_mape_status", "not_available")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        model.save(MODEL_PATH)
        print(f"\nModel saved: {MODEL_PATH}")

        training_loss_plot_path = PLOT_DIR / "tft_quantile_training_loss.png"
        forecast_plot_path = PLOT_DIR / "tft_quantile_test_actual_vs_q50_forecast.png"
        first_forecast_plot_path = (
            PLOT_DIR / f"tft_quantile_first_{FORECAST_HORIZON}h_forecast.png"
        )
        horizon_mae_plot_path = PLOT_DIR / "tft_quantile_horizon_mae_q50.png"
        horizon_mae_csv_path = PLOT_DIR / "tft_quantile_horizon_mae_q50.csv"
        block_24h_metrics_csv_path = PLOT_DIR / "tft_quantile_24h_block_metrics_q50.csv"

        save_training_loss_plot(history, training_loss_plot_path)
        save_forecast_plot(y_test_flat, y_pred_flat, forecast_plot_path)
        save_first_forecast_plot(
            y_test_real,
            y_pred_q50_real,
            first_forecast_plot_path,
            y_pred_lower=y_pred_interval_lower_real,
            y_pred_upper=y_pred_interval_upper_real,
        )
        first_block_plot_paths = save_first_forecast_block_plots(
            y_test_real,
            y_pred_q50_real,
            PLOT_DIR,
            y_pred_lower=y_pred_interval_lower_real,
            y_pred_upper=y_pred_interval_upper_real,
        )
        save_horizon_mae_plot(horizon_mae, horizon_mae_plot_path)

        horizon_mae_df = pd.DataFrame(
            {
                "hours_ahead": np.arange(1, FORECAST_HORIZON + 1),
                "mae": horizon_mae,
            }
        )
        horizon_mae_df.to_csv(horizon_mae_csv_path, index=False)
        block_24h_metrics_df.to_csv(block_24h_metrics_csv_path, index=False)

        if mlflow_enabled:
            mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
            mlflow.log_artifact(str(training_loss_plot_path), artifact_path="plots")
            mlflow.log_artifact(str(forecast_plot_path), artifact_path="plots")
            mlflow.log_artifact(str(first_forecast_plot_path), artifact_path="plots")
            for block_plot_path in first_block_plot_paths:
                mlflow.log_artifact(str(block_plot_path), artifact_path="plots")
            mlflow.log_artifact(str(horizon_mae_plot_path), artifact_path="plots")
            mlflow.log_artifact(str(horizon_mae_csv_path), artifact_path="metrics")
            mlflow.log_artifact(str(block_24h_metrics_csv_path), artifact_path="metrics")
            print("MLflow run logged successfully.")


if __name__ == "__main__":
    train_tft()
