import pickle

import numpy as np
import pandas as pd

import src.config as cfg
from src.feature_importance_utils import (
    average_horizon_scores,
    build_combined_importance_table,
    permutation_importance_scores,
    save_importance_artifacts,
)
from src.lgbm.lgbm_features import get_feature_columns, load_featured_datasets, prepare_xy

FORECAST_HORIZONS = [1, 2, 3]


def load_saved_models() -> dict[int, object]:
    model_dir = cfg.get_model_dir("lgbm")
    models = {}
    for horizon in FORECAST_HORIZONS:
        path = model_dir / f"lgbm_day{horizon}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python -m src.lgbm.train_lgbm` first."
            )
        with path.open("rb") as handle:
            models[horizon] = pickle.load(handle)
    return models


def run_feature_importance() -> pd.DataFrame:
    train_df, test_df = load_featured_datasets()
    feature_columns = get_feature_columns(train_df)
    models = load_saved_models()

    gain_lists: list[np.ndarray] = []
    perm_lists: list[np.ndarray] = []

    for horizon in FORECAST_HORIZONS:
        model = models[horizon]
        gain_lists.append(model.feature_importances_)
        x_test, y_test = prepare_xy(test_df, horizon, feature_columns)
        perm_lists.append(
            permutation_importance_scores(model, x_test, y_test, n_repeats=5)
        )

    combined = build_combined_importance_table(
        feature_columns,
        average_horizon_scores(gain_lists),
        average_horizon_scores(perm_lists),
    )

    save_importance_artifacts(
        combined,
        cfg.REPORT_DIR / "lgbm_feature_importance.csv",
        cfg.REPORT_DIR / "lgbm_feature_importance.png",
        "LightGBM feature importance\n(averaged across day1, day2, day3 models)",
    )
    return combined


def main() -> None:
    combined = run_feature_importance()
    print(combined.to_string(index=False))
    print("\nSaved → reports/lgbm_feature_importance.csv")
    print("Saved → reports/lgbm_feature_importance.png")


if __name__ == "__main__":
    main()
