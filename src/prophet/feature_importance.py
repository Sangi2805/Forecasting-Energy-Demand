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
from src.prophet.prophet_features import (
    HORIZON_LABELS,
    engineer_features,
    get_regressor_columns,
    load_datasets,
)
from src.prophet.train_prophet import predict_horizon

FORECAST_HORIZONS = [1, 2, 3]


class ProphetHorizonRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model, regressors: list[str], dates: pd.Series):
        self.model = model
        self.regressors = regressors
        self.dates = pd.to_datetime(dates).reset_index(drop=True)

    def fit(self, x, y):
        return self

    def predict(self, x) -> np.ndarray:
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x, columns=self.regressors)
        frame = x.copy()
        frame["date"] = self.dates.iloc[: len(frame)].values
        forecast = predict_horizon(self.model, frame, self.regressors)
        return forecast["yhat"].values


def load_saved_models(regressors: list[str]) -> dict[int, object]:
    model_dir = cfg.get_model_dir("prophet")
    models = {}
    for horizon in FORECAST_HORIZONS:
        path = model_dir / f"prophet_day{horizon}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python -m src.prophet.train_prophet` first."
            )
        with path.open("rb") as handle:
            models[horizon] = pickle.load(handle)
    return models


def prophet_regressor_scores(model, regressors: list[str]) -> np.ndarray:
    beta = model.params["beta"]
    extra_cols = list(model.extra_regressors.keys())
    scores = []
    for regressor in regressors:
        if regressor not in extra_cols:
            scores.append(0.0)
            continue
        idx = extra_cols.index(regressor)
        scores.append(float(np.mean(np.abs(beta[:, idx]))))
    return np.array(scores)


def run_feature_importance() -> pd.DataFrame:
    train_df, test_df = load_datasets()
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    regressors = get_regressor_columns(train_df)
    prophet_models = load_saved_models(regressors)

    gain_lists: list[np.ndarray] = []
    perm_lists: list[np.ndarray] = []

    for horizon in FORECAST_HORIZONS:
        prophet_model = prophet_models[horizon]
        gain_lists.append(prophet_regressor_scores(prophet_model, regressors))

        target_col = HORIZON_LABELS[horizon]
        frame = test_df[["date", target_col, *regressors]].dropna()
        wrapper = ProphetHorizonRegressor(
            prophet_model, regressors, frame["date"]
        )
        x_test = frame[regressors]
        y_test = frame[target_col]
        perm_lists.append(
            permutation_importance_scores(wrapper, x_test, y_test, n_repeats=5)
        )

    combined = build_combined_importance_table(
        regressors,
        average_horizon_scores(gain_lists),
        average_horizon_scores(perm_lists),
    )

    save_importance_artifacts(
        combined,
        cfg.REPORT_DIR / "prophet_feature_importance.csv",
        cfg.REPORT_DIR / "prophet_feature_importance.png",
        "Prophet regressor importance\n"
        "(native = |beta| averaged across day1, day2, day3 models)",
    )
    return combined


def main() -> None:
    combined = run_feature_importance()
    print(combined.to_string(index=False))
    print("\nSaved → reports/prophet_feature_importance.csv")
    print("Saved → reports/prophet_feature_importance.png")


if __name__ == "__main__":
    main()
