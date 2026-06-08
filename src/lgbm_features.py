from pathlib import Path

import numpy as np
import pandas as pd

import src.config as cfg
from src.prophet_features import HORIZON_LABELS, engineer_features, load_datasets

EXCLUDE_COLUMNS = {
    "date",
    "target_day1",
    "target_day2",
    "target_day3",
}


def engineer_lgbm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extend base features with tree-friendly interactions and cyclical encodings."""
    featured = engineer_features(df)

    featured["week_sin"] = np.sin(2 * np.pi * featured["week_of_year"] / 52)
    featured["week_cos"] = np.cos(2 * np.pi * featured["week_of_year"] / 52)
    featured["demand_vs_roll7"] = featured["demand"] / featured["demand_roll_mean_7d"].clip(lower=1)
    featured["demand_vs_roll30"] = featured["demand"] / featured["demand_roll_mean_30d"].clip(lower=1)
    featured["temp_x_humidity"] = featured["temperature_2m"] * featured["relative_humidity_2m"]
    featured["temp_x_wind"] = featured["temperature_2m"] * featured["wind_speed_10m"]
    featured["weekend_x_hdd"] = featured["is_weekend"] * featured["hdd"]
    featured["holiday_x_demand_lag7"] = featured["is_holiday"] * featured["demand_lag_7d"]

    return featured


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return [col for col in numeric_cols if col not in EXCLUDE_COLUMNS]


def prepare_xy(
    df: pd.DataFrame,
    horizon: int,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    target_col = HORIZON_LABELS[horizon]
    frame = df[["date", target_col, *feature_columns]].dropna()
    x = frame[feature_columns]
    y = frame[target_col]
    return x, y


def load_featured_datasets(
    train_path: Path | None = None,
    test_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = load_datasets(train_path, test_path)
    return engineer_lgbm_features(train_df), engineer_lgbm_features(test_df)
