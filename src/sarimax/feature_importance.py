import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

import src.config as cfg
from src.feature_importance_utils import (
    average_horizon_scores,
    build_combined_importance_table,
    permutation_importance_scores,
    save_importance_artifacts,
)
from src.sarimax.sarimax_features import (
    get_exog_columns,
    load_datasets,
    prepare_horizon_series,
)

FORECAST_HORIZONS = [1, 2, 3]


class SarimaxHorizonRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, fit_result):
        self.fit_result = fit_result

    def fit(self, x, y):
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        forecast = self.fit_result.get_forecast(steps=len(x), exog=x)
        return forecast.predicted_mean.values


def load_saved_models() -> dict[int, SarimaxHorizonRegressor]:
    model_dir = cfg.get_model_dir("sarimax")
    models = {}
    for horizon in FORECAST_HORIZONS:
        path = model_dir / f"sarimax_day{horizon}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python -m src.sarimax.train_sarimax` first."
            )
        with path.open("rb") as handle:
            models[horizon] = SarimaxHorizonRegressor(pickle.load(handle))
    return models


def sarimax_exog_scores(fit_result, exog_columns: list[str]) -> np.ndarray:
    params = fit_result.params
    return np.array([
        abs(float(params[feature])) if feature in params.index else 0.0
        for feature in exog_columns
    ])


def run_feature_importance() -> pd.DataFrame:
    train_df, test_df = load_datasets()
    exog_columns = get_exog_columns(train_df)
    models = load_saved_models()

    gain_lists: list[np.ndarray] = []
    perm_lists: list[np.ndarray] = []

    for horizon in FORECAST_HORIZONS:
        wrapper = models[horizon]
        gain_lists.append(sarimax_exog_scores(wrapper.fit_result, exog_columns))

        endog_test, exog_test = prepare_horizon_series(test_df, horizon, exog_columns)
        perm_lists.append(
            permutation_importance_scores(
                wrapper,
                exog_test,
                endog_test,
                n_repeats=3,
                n_jobs=1,
            )
        )

    combined = build_combined_importance_table(
        exog_columns,
        average_horizon_scores(gain_lists),
        average_horizon_scores(perm_lists),
    )

    save_importance_artifacts(
        combined,
        cfg.REPORT_DIR / "sarimax_feature_importance.csv",
        cfg.REPORT_DIR / "sarimax_feature_importance.png",
        "SARIMAX exogenous feature importance\n"
        "(native = |beta| summed across day1, day2, day3 models)",
    )
    return combined


def main() -> None:
    combined = run_feature_importance()
    print(combined.to_string(index=False))
    print("\nSaved → reports/sarimax_feature_importance.csv")
    print("Saved → reports/sarimax_feature_importance.png")


if __name__ == "__main__":
    main()
