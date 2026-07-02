from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


OUTPUT_COLUMNS = [
    "combined_rank",
    "feature",
    "gain_rank",
    "perm_rank",
    "avg_rank",
    "gain_score",
    "contribution_pct",
    "cumulative_pct",
]


def build_combined_importance_table(
    feature_names: list[str],
    gain_scores: np.ndarray,
    perm_scores: np.ndarray,
) -> pd.DataFrame:
    gain_df = (
        pd.DataFrame({"feature": feature_names, "gain_score": gain_scores})
        .sort_values("gain_score", ascending=False)
        .reset_index(drop=True)
    )
    gain_df["gain_rank"] = gain_df.index + 1

    perm_df = (
        pd.DataFrame({"feature": feature_names, "perm_score": perm_scores})
        .sort_values("perm_score", ascending=False)
        .reset_index(drop=True)
    )
    perm_df["perm_rank"] = perm_df.index + 1

    combined = gain_df.merge(perm_df[["feature", "perm_rank"]], on="feature")
    combined["avg_rank"] = (combined["gain_rank"] + combined["perm_rank"]) / 2
    combined = combined.sort_values("avg_rank").reset_index(drop=True)
    combined["combined_rank"] = combined.index + 1

    total_gain = combined["gain_score"].sum()
    if total_gain > 0:
        combined["contribution_pct"] = (combined["gain_score"] / total_gain * 100).round(2)
    else:
        combined["contribution_pct"] = 0.0
    combined["cumulative_pct"] = combined["contribution_pct"].cumsum().round(2)

    return combined[OUTPUT_COLUMNS]


def permutation_importance_scores(
    estimator,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
) -> np.ndarray:
    result = permutation_importance(
        estimator,
        x_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
        scoring="neg_mean_absolute_error",
    )
    return result.importances_mean


def average_horizon_scores(score_lists: list[np.ndarray]) -> np.ndarray:
    if not score_lists:
        raise ValueError("score_lists must not be empty")
    return np.mean(np.vstack(score_lists), axis=0)


def save_importance_artifacts(
    combined: pd.DataFrame,
    csv_path: Path,
    png_path: Path,
    title: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False)

    ranked = combined.sort_values("gain_score")
    colors = cm.RdYlGn(np.linspace(0.2, 0.9, len(ranked)))

    fig, ax = plt.subplots(figsize=(10, max(6, len(ranked) * 0.35)))
    ax.barh(ranked["feature"], ranked["gain_score"], color=colors, edgecolor="none")
    ax.set_xlabel("Native importance score", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
