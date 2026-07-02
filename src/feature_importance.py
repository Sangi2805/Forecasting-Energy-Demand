"""Generate feature-importance tables matching the teammate XGBoost format."""

import argparse

from src.lgbm.feature_importance import run_feature_importance as run_lgbm
from src.prophet.feature_importance import run_feature_importance as run_prophet
from src.sarimax.feature_importance import run_feature_importance as run_sarimax
from src.xgboost.feature_importance import run_feature_importance as run_xgboost

RUNNERS = {
    "xgboost": run_xgboost,
    "lgbm": run_lgbm,
    "prophet": run_prophet,
    "sarimax": run_sarimax,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build gain + permutation feature-importance CSVs."
    )
    parser.add_argument(
        "--model",
        choices=[*RUNNERS.keys(), "all"],
        default="all",
        help="Model family to analyze (default: all).",
    )
    args = parser.parse_args()

    models = RUNNERS if args.model == "all" else {args.model: RUNNERS[args.model]}
    for name, runner in models.items():
        print(f"\n=== {name.upper()} ===")
        runner()


if __name__ == "__main__":
    main()
