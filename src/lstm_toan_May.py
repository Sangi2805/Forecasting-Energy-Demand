import json

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg
from config import configure_mlflow_tracking

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


# ==================================================
# Configuration
# ==================================================
DATA_PATH = cfg.REPORT_DIR / "all_features_dataset.csv"
MODEL_PATH = cfg.MODEL_DIR / "lstm_model.keras"
PLOT_DIR = cfg.REPORT_DIR / "plots"

TARGET_COL = "Demand"

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
    #####
    "demand_lag_1h",
    "demand_lag_2h",
    "demand_lag_3h",
    "demand_lag_4h", 
    "demand_lag_24h",
    "demand_lag_48h",
    "demand_lag_72h",
    "demand_rolling_24h_mean",
    "demand_rolling_168h_mean",
    "demand_std_24h",
    "demand_std_168h",
    "demand_min_24h",
    "demand_max_24h",
    #####

    "NYNGSP",
    "NYPOP",
]

WINDOW_SIZE = 24*7  # Use past 7 days (168 hours) to predict future demand
FORECAST_HORIZON = 72
TRAIN_RATIO = 0.8
VALIDATION_SPLIT = 0.2
EPOCHS = 20
BATCH_SIZE = 64
LSTM_UNITS = 64
DROPOUT_RATE = 0.2
EARLY_STOPPING_PATIENCE = 5


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
            "Not enough rows to create training sequences for the current "
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
            LSTM(
                units=LSTM_UNITS,
                input_shape=(WINDOW_SIZE, len(FEATURES)),
            ),
            Dropout(DROPOUT_RATE),
            Dense(FORECAST_HORIZON),
        ]
    )

    model.compile(optimizer="adam", loss="mse")
    return model


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
    plt.title("Actual vs Forecast")
    plt.xlabel("Forecasted point")
    plt.ylabel(TARGET_COL)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def log_mlflow_run_metadata(df, X_train, X_test):
    params = {
        "data_path": str(DATA_PATH),
        "target_col": TARGET_COL,
        "window_size": WINDOW_SIZE,
        "forecast_horizon": FORECAST_HORIZON,
        "train_ratio": TRAIN_RATIO,
        "validation_split": VALIDATION_SPLIT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lstm_units": LSTM_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "optimizer": "adam",
        "loss": "mse",
        "feature_count": len(FEATURES),
        "row_count_after_dropna": len(df),
        "train_sequence_count": len(X_train),
        "test_sequence_count": len(X_test),
    }
    mlflow.log_params(params)
    mlflow.log_text(json.dumps(FEATURES, indent=2), "features.json")


def train_lstm():
    configure_mlflow_tracking()

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL].astype(str).str.replace(",", "", regex=False)
    )

    df = df.dropna()

    missing_features = [feature for feature in FEATURES if feature not in df.columns]
    if missing_features:
        raise KeyError(f"Missing features in dataset: {missing_features}")

    print("Data shape:", df.shape)

    print("Scaling data...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[FEATURES])

    print("Creating sliding windows...")
    X, y = create_sequences(scaled_data)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    split_idx = int(len(X) * TRAIN_RATIO)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    with mlflow.start_run(run_name="lstm_Toan"):
        log_mlflow_run_metadata(df, X_train, X_test)

        print("Building model...")
        model = build_lstm_model()
        model.summary()

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )

        print("Training...")
        history = model.fit(
            X_train,
            y_train,
            validation_split=VALIDATION_SPLIT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=1,
        )

        for step, (loss, val_loss) in enumerate(
            zip(history.history["loss"], history.history["val_loss"])
        ):
            mlflow.log_metric("train_loss", loss, step=step)
            mlflow.log_metric("val_loss", val_loss, step=step)

        print("Predicting...")
        y_pred = model.predict(X_test)

        y_pred_real = inverse_transform_target_sequences(scaler, y_pred)
        y_test_real = inverse_transform_target_sequences(scaler, y_test)

        y_pred_flat = y_pred_real.reshape(-1)
        y_test_flat = y_test_real.reshape(-1)

        mae = mean_absolute_error(y_test_flat, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))

        epsilon = 1e-6
        nonzero_mask = np.abs(y_test_flat) > epsilon
        excluded_zeros = np.size(y_test_flat) - np.count_nonzero(nonzero_mask)

        if np.any(nonzero_mask):
            mape = (
                np.mean(
                    np.abs(
                        (y_test_flat[nonzero_mask] - y_pred_flat[nonzero_mask])
                        / y_test_flat[nonzero_mask]
                    )
                )
                * 100
            )
        else:
            mape = np.nan

        print("\n===== RESULTS =====")
        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"MAPE : {mape:.2f}%")
        print(f"MAPE excluded zero-demand points: {excluded_zeros}")

        mlflow.log_metrics(
            {
                "mae": float(mae),
                "rmse": float(rmse),
                "excluded_zero_demand_points": int(excluded_zeros),
            }
        )
        if np.isfinite(mape):
            mlflow.log_metric("mape", float(mape))
        else:
            mlflow.log_param("mape_status", "not_available_all_targets_are_zero")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        model.save(MODEL_PATH)
        print(f"\nModel saved: {MODEL_PATH}")

        training_loss_plot_path = PLOT_DIR / "lstm_training_loss.png"
        forecast_plot_path = PLOT_DIR / "lstm_actual_vs_forecast.png"

        save_training_loss_plot(history, training_loss_plot_path)
        save_forecast_plot(y_test_flat, y_pred_flat, forecast_plot_path)

        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        mlflow.log_artifact(str(training_loss_plot_path), artifact_path="plots")
        mlflow.log_artifact(str(forecast_plot_path), artifact_path="plots")

        print("MLflow run logged successfully.")


if __name__ == "__main__":
    train_lstm()
