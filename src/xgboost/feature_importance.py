import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import src.config as cfg
from src.feature_importance_utils import (
    build_combined_importance_table,
    save_importance_artifacts,
)

TARGETS = ["target_day1", "target_day2", "target_day3"]
DROP_COLS = [
    "net_generation", "total_interchange", "ng_nuclear", "ng_hydro",
    "ng_solar", "ng_wind", "ng_natural_gas",
    "is_weekend", "quarter", "snow_depth", "snowfall",
    "wind_gusts_10m", "relative_humidity_2m",
    "relative_humidity_2m_lag_2d", "day_of_month",
]


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(
        cfg.PROCESSED_DATA_DIR / "features_selected_train.csv",
        index_col="date",
        parse_dates=["date"],
    )
    test = pd.read_csv(
        cfg.PROCESSED_DATA_DIR / "features_selected_test.csv",
        index_col="date",
        parse_dates=["date"],
    )
    x_train = train.drop(columns=TARGETS + DROP_COLS)
    y_train = train[TARGETS]
    x_test = test.drop(columns=TARGETS + DROP_COLS)
    y_test = test[TARGETS]
    x_train["season"] = (x_train.index.month % 12) // 3
    x_test["season"] = (x_test.index.month % 12) // 3
    return x_train, y_train, x_test, y_test


def load_saved_model():
    model_path = cfg.get_model_dir("xgboost") / "xgboost_tuned.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Run `python -m src.xgboost.tune_xgboost` first."
        )
    return joblib.load(model_path)


def run_feature_importance() -> pd.DataFrame:
    x_train, y_train, x_test, y_test = load_datasets()
    model = load_saved_model()
    feature_names = x_train.columns.tolist()

    gain_scores = np.zeros(len(feature_names))
    perm_scores = np.zeros(len(feature_names))

    for estimator, target in zip(model.estimators_, y_test.columns):
        booster_scores = estimator.get_booster().get_score(importance_type="gain")
        for feat, score in booster_scores.items():
            gain_scores[feature_names.index(feat)] += score

        result = permutation_importance(
            estimator,
            x_test,
            y_test[target],
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
            scoring="neg_mean_absolute_error",
        )
        perm_scores += result.importances_mean

    gain_scores /= len(model.estimators_)
    perm_scores /= len(model.estimators_)

    combined = build_combined_importance_table(feature_names, gain_scores, perm_scores)
    save_importance_artifacts(
        combined,
        cfg.REPORT_DIR / "xgboost_feature_importance.csv",
        cfg.REPORT_DIR / "xgboost_feature_importance.png",
        "XGBoost feature importance\n(averaged across day1, day2, day3 targets)",
    )
    return combined


def main() -> None:
    combined = run_feature_importance()
    print(combined.to_string(index=False))
    print("\nSaved → reports/xgboost_feature_importance.csv")
    print("Saved → reports/xgboost_feature_importance.png")


if __name__ == "__main__":
    main()
