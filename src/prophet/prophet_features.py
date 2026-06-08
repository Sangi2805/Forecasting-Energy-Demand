from pathlib import Path

import numpy as np
import pandas as pd

import src.config as cfg

HORIZON_LABELS = {1: "target_day1", 2: "target_day2", 3: "target_day3"}

PROPHET_REGRESSORS = [
    "temperature_2m",
    "temp_max",
    "temp_min",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "precipitation",
    "rain",
    "is_holiday",
    "is_weekend",
    "day_of_week",
    "month",
    "demand_lag_7d",
    "demand_lag_14d",
    "demand_roll_mean_7d",
    "demand_roll_std_7d",
    "demand_roll_mean_30d",
    "temperature_2m_lag_1d",
    "temperature_2m_lag_2d",
    "relative_humidity_2m_lag_1d",
]

ENGINEERED_REGRESSORS = [
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "hdd",
    "cdd",
    "temp_range",
]


def load_datasets(
    train_path: Path | None = None,
    test_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = train_path or cfg.PROCESSED_DATA_DIR / "features_selected_train.csv"
    test_path = test_path or cfg.PROCESSED_DATA_DIR / "features_selected_test.csv"

    train_df = pd.read_csv(train_path, parse_dates=["date"])
    test_df = pd.read_csv(test_path, parse_dates=["date"])
    return train_df.sort_values("date").reset_index(drop=True), test_df.sort_values(
        "date"
    ).reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    engineered["month_sin"] = np.sin(2 * np.pi * engineered["month"] / 12)
    engineered["month_cos"] = np.cos(2 * np.pi * engineered["month"] / 12)
    engineered["dow_sin"] = np.sin(2 * np.pi * engineered["day_of_week"] / 7)
    engineered["dow_cos"] = np.cos(2 * np.pi * engineered["day_of_week"] / 7)
    engineered["hdd"] = np.clip(18 - engineered["temperature_2m"], 0, None)
    engineered["cdd"] = np.clip(engineered["temperature_2m"] - 22, 0, None)
    engineered["temp_range"] = engineered["temp_max"] - engineered["temp_min"]
    return engineered


def get_regressor_columns(df: pd.DataFrame) -> list[str]:
    base = [col for col in PROPHET_REGRESSORS if col in df.columns]
    engineered = [col for col in ENGINEERED_REGRESSORS if col in df.columns]
    return base + engineered


def prepare_prophet_frame(
    df: pd.DataFrame,
    horizon: int,
    regressors: list[str],
) -> pd.DataFrame:
    target_col = HORIZON_LABELS[horizon]
    prophet_df = df[["date", target_col, *regressors]].rename(
        columns={"date": "ds", target_col: "y"}
    )
    return prophet_df.dropna().reset_index(drop=True)
