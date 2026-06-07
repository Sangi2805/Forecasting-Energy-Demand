

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

DATA_PATH = "data/merged_dataset.csv"

TARGET_COL = "Demand"

FEATURES = [
    "Demand",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "snowfall",
    "hour",
    "month",
    "holiday"
]

WINDOW_SIZE = 24

TRAIN_RATIO = 0.8

EPOCHS = 20

BATCH_SIZE = 64


# ==================================================
# Load Data
# ==================================================

print("Loading data...")

df = pd.read_csv(DATA_PATH)

df[TARGET_COL] = pd.to_numeric(
    df[TARGET_COL]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df = df.dropna()

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

mape = (
    np.mean(
        np.abs(
            (
                y_test_real -
                y_pred_real
            )
            / y_test_real
        )
    )
    * 100
)

print("\n===== RESULTS =====")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")



# ==================================================
# Save Model
# ==================================================

model.save(
    "models/lstm_model.keras"
)

print(
    "\nModel saved:"
    " models/lstm_model.keras"
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