

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_collection as dc
import mlflow
import config   as cfg
from config import configure_mlflow_tracking

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping


# ==================================================
# Configuration
# ==================================================

DATA_PATH = cfg.REPORT_DIR / "all_features_dataset.csv"
MODEL_PATH = cfg.MODEL_DIR / "lstm_model.keras"

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
    "NYNGSP",
    "NYPOP"
]

WINDOW_SIZE = 24

TRAIN_RATIO = 0.8

EPOCHS = 5

BATCH_SIZE = 64
'''
def encode_holiday_to_numeric(series):
    # Any non-empty holiday label is encoded as 1, otherwise 0.
    cleaned = series.fillna(0).astype(str).str.strip().str.lower()
    return (~cleaned.isin(["0", "", "none", "nan", "null"])).astype(int)


def add_cyclical_features(df_input):
    df_output = df_input.copy()

    hour_series = pd.to_numeric(df_output["Hour"], errors="coerce")
    month_series = pd.to_numeric(df_output["month"], errors="coerce")

    if "Local time" in df_output.columns:
        local_time_dt = pd.to_datetime(df_output["Local time"], errors="coerce")
        weekday_series = local_time_dt.dt.dayofweek
    elif "day_of_week" in df_output.columns:
        day_map = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6
        }
        weekday_series = (
            df_output["day_of_week"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(day_map)
        )
    else:
        raise KeyError("Missing both 'Local time' and 'day_of_week' columns.")

    df_output["hour_sin"] = np.sin(2 * np.pi * hour_series / 24.0)
    df_output["hour_cos"] = np.cos(2 * np.pi * hour_series / 24.0)

    df_output["weekday_sin"] = np.sin(2 * np.pi * weekday_series / 7.0)
    df_output["weekday_cos"] = np.cos(2 * np.pi * weekday_series / 7.0)

    df_output["month_sin"] = np.sin(2 * np.pi * (month_series - 1) / 12.0)
    df_output["month_cos"] = np.cos(2 * np.pi * (month_series - 1) / 12.0)

    return df_output
'''
# ==================================================
# Load Data
# ==================================================
def train_lstm():
    print("Loading data...")

    df = pd.read_csv(DATA_PATH)

    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    #df["holiday"] = encode_holiday_to_numeric(df["holiday"])

    #df = add_cyclical_features(df)

    df = df.dropna()

    missing_features = [feature for feature in FEATURES if feature not in df.columns]
    if missing_features:
        raise KeyError(f"Missing features in dataset: {missing_features}")

    print(df.shape)


    # ==================================================
    # Scaling
    # ==================================================

    print("Scaling data...")

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(
        df[FEATURES]
    )


    # ==================================================
    # Create Sequences
    # ==================================================

    print("Creating sliding windows...")

    X = []
    y = []

    for i in range(
        WINDOW_SIZE,
        len(scaled_data)
    ):

        X.append(
            scaled_data[
                i-WINDOW_SIZE:i
            ]
        )

        y.append(
            scaled_data[i, 0]
        )

    X = np.array(X)
    y = np.array(y)

    print("X shape:", X.shape)
    print("y shape:", y.shape)


    # ==================================================
    # Train Test Split
    # ==================================================

    split_idx = int(
        len(X) * TRAIN_RATIO
    )

    X_train = X[:split_idx]
    X_test = X[split_idx:]

    y_train = y[:split_idx]
    y_test = y[split_idx:]

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)


    # ==================================================
    # Build LSTM
    # ==================================================

    print("Building model...")

    model = Sequential()

    model.add(
        LSTM(
            units=64,
            input_shape=(
                WINDOW_SIZE,
                len(FEATURES)
            )
        )
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    model.summary()


    # ==================================================
    # Train
    # ==================================================

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    print("Training...")

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )


    # ==================================================
    # Predict
    # ==================================================

    print("Predicting...")

    y_pred = model.predict(X_test)


    # ==================================================
    # Inverse Scaling
    # ==================================================

    dummy_pred = np.zeros(
        (
            len(y_pred),
            len(FEATURES)
        )
    )

    dummy_test = np.zeros(
        (
            len(y_test),
            len(FEATURES)
        )
    )

    dummy_pred[:, 0] = y_pred.flatten()
    dummy_test[:, 0] = y_test.flatten()

    y_pred_real = scaler.inverse_transform(
        dummy_pred
    )[:, 0]

    y_test_real = scaler.inverse_transform(
        dummy_test
    )[:, 0]


    # ==================================================
    # Evaluation
    # ==================================================

    mae = mean_absolute_error(
        y_test_real,
        y_pred_real
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test_real,
            y_pred_real
        )
    )

    epsilon = 1e-6
    nonzero_mask = np.abs(y_test_real) > epsilon
    excluded_zeros = np.size(y_test_real) - np.count_nonzero(nonzero_mask)

    if np.any(nonzero_mask):
        mape = (
            np.mean(
                np.abs(
                    (
                        y_test_real[nonzero_mask] -
                        y_pred_real[nonzero_mask]
                    )
                    / y_test_real[nonzero_mask]
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

    # Log metrics to MLflow
    #mlflow.log_metric("MAE", mae)
    #mlflow.log_metric("RMSE", rmse)
    #mlflow.log_metric("MAPE", mape)
    #mlflow.log_param("excluded_zeros", excluded_zeros)



    # ==================================================
    # Save Model
    # ==================================================

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    model.save(
        MODEL_PATH
    )

    print(
        "\nModel saved:"
        f" {MODEL_PATH}"
    )


    # ==================================================
    # Plot Training Loss
    # ==================================================

    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["loss"],
        label="Train Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.legend()

    plt.title(
        "LSTM Training Loss"
    )

    plt.show()


    # ==================================================
    # Plot Forecast
    # ==================================================

    plt.figure(figsize=(15,5))

    plt.plot(
        y_test_real[:500],
        label="Actual"
    )

    plt.plot(
        y_pred_real[:500],
        label="Forecast"
    )

    plt.legend()

    plt.title(
        "Actual vs Forecast"
    )

    plt.show()




if __name__ == "__main__":
    train_lstm()
    #df=dc.read_csv_file("reports/all_features_dataset.csv")
    #df['Demand'].describe()
    #print(df[df['Demand']==0].head())
