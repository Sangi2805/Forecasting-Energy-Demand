from pathlib import Path

import pandas as pd

import src.config as cfg
from src.prophet_features import HORIZON_LABELS
from src.prophet_features import load_datasets as _load_datasets

# Match teammate DagsHub baseline feature sets
BASELINE_FEATURES_A = [
    "apparent_temperature",
    "wind_speed_10m",
    "demand_roll_mean_7d",
    "demand_roll_std_7d",
    "demand_lag_3d",
    "demand_lag_14d",
    "is_weekend",
    "is_holiday",
    "day_of_week",
]

BASELINE_FEATURES_B = [
    "demand_lag_3d",
    "demand_lag_7d",
    "demand_roll_mean_7d",
    "is_weekend",
    "is_holiday",
]

DEFAULT_FEATURES = BASELINE_FEATURES_A

SARIMAX_ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 7)


def load_datasets(
    train_path: Path | None = None,
    test_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _load_datasets(train_path, test_path)


def get_exog_columns(df: pd.DataFrame, features: list[str] | None = None) -> list[str]:
    features = features or DEFAULT_FEATURES
    return [col for col in features if col in df.columns]


def prepare_horizon_series(
    df: pd.DataFrame,
    horizon: int,
    exog_columns: list[str],
) -> tuple[pd.Series, pd.DataFrame]:
    target_col = HORIZON_LABELS[horizon]
    frame = df[["date", target_col, *exog_columns]].dropna().sort_values("date")
    endog = frame.set_index("date")[target_col]
    exog = frame.set_index("date")[exog_columns]
    return endog, exog
