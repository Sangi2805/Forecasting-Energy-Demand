import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.inspection import permutation_importance

# ── Paths via config ──────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import PROCESSED_DATA_DIR, REPORT_DIR

# ── Config ────────────────────────────────────────────────────────────────────
TARGETS = ["target_day1", "target_day2", "target_day3"]

DROP_COLS = [
    "net_generation", "total_interchange", "ng_nuclear", "ng_hydro",
    "ng_solar", "ng_wind", "ng_natural_gas",
    "is_weekend", "quarter", "snow_depth", "snowfall",
    "wind_gusts_10m", "relative_humidity_2m",
    "relative_humidity_2m_lag_2d", "day_of_month",
]

# ── Load data ─────────────────────────────────────────────────────────────────
train = pd.read_csv(
    PROCESSED_DATA_DIR / "features_selected_train.csv",
    index_col="date",
    parse_dates=["date"]
)
test = pd.read_csv(
    PROCESSED_DATA_DIR / "features_selected_test.csv",
    index_col="date",
    parse_dates=["date"]
)

X_train = train.drop(columns=TARGETS + DROP_COLS)
y_train = train[TARGETS]

X_test = test.drop(columns=TARGETS + DROP_COLS)
y_test = test[TARGETS]

X_train["season"] = (X_train.index.month % 12) // 3
X_test["season"]  = (X_test.index.month  % 12) // 3

feature_names = X_train.columns.tolist()

# ── Train model ───────────────────────────────────────────────────────────────
print("Training model for feature importance analysis...")

model = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        device="cpu",
    )
)
model.fit(X_train, y_train)

# ── Method 1: XGBoost Built-in Gain Importance ────────────────────────────────
print("\n── Method 1: XGBoost Built-in Importance (avg gain across targets) ──")

gain_scores = np.zeros(len(feature_names))

for estimator in model.estimators_:
    scores = estimator.get_booster().get_score(importance_type="gain")
    for feat, score in scores.items():
        idx = feature_names.index(feat)
        gain_scores[idx] += score

gain_scores /= len(model.estimators_)

gain_df = pd.DataFrame({
    "feature":    feature_names,
    "gain_score": gain_scores,
}).sort_values("gain_score", ascending=False).reset_index(drop=True)

gain_df["rank"] = gain_df.index + 1
print(gain_df[["rank", "feature", "gain_score"]].to_string(index=False))

# ── Method 2: Permutation Importance ─────────────────────────────────────────
print("\n── Method 2: Permutation Importance (test set, avg across targets) ──")

perm_scores = np.zeros(len(feature_names))

for i, estimator in enumerate(model.estimators_):
    result = permutation_importance(
        estimator,
        X_test,
        y_test.iloc[:, i],
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    perm_scores += result.importances_mean

perm_scores /= len(model.estimators_)

perm_df = pd.DataFrame({
    "feature":    feature_names,
    "perm_score": perm_scores,
}).sort_values("perm_score", ascending=False).reset_index(drop=True)

perm_df["rank"] = perm_df.index + 1
print(perm_df[["rank", "feature", "perm_score"]].to_string(index=False))

# ── Combined Rank ─────────────────────────────────────────────────────────────
print("\n── Combined Rank (average of both methods) ──")

gain_df_r = gain_df.set_index("feature")[["rank", "gain_score"]].rename(
    columns={"rank": "gain_rank"}
)
perm_df_r = perm_df.set_index("feature")[["rank"]].rename(
    columns={"rank": "perm_rank"}
)

combined = gain_df_r.join(perm_df_r)
combined["avg_rank"] = (combined["gain_rank"] + combined["perm_rank"]) / 2
combined = combined.sort_values("avg_rank").reset_index()
combined["combined_rank"] = combined.index + 1

# ── % Contribution based on gain score ───────────────────────────────────────
total_gain = combined["gain_score"].sum()
combined["contribution_pct"] = (combined["gain_score"] / total_gain * 100).round(2)
combined["cumulative_pct"] = combined["contribution_pct"].cumsum().round(2)

print(
    combined[[
        "combined_rank", "feature", "gain_rank", "perm_rank",
        "avg_rank", "gain_score", "contribution_pct", "cumulative_pct"
    ]].to_string(index=False)
)

# ── Save CSV ──────────────────────────────────────────────────────────────────
combined[[
    "combined_rank", "feature", "gain_rank", "perm_rank",
    "avg_rank", "gain_score", "contribution_pct", "cumulative_pct"
]].to_csv(REPORT_DIR / "xgboost_feature_importance.csv", index=False)
print("\nSaved → reports/xgboost_feature_importance.csv")

# ── Plot All Features ─────────────────────────────────────────────────────────
all_features = gain_df.sort_values("gain_score")
n = len(all_features)
colors = cm.RdYlGn(np.linspace(0.2, 0.9, n))

fig, ax = plt.subplots(figsize=(10, n * 0.35))
ax.barh(all_features["feature"], all_features["gain_score"], color=colors, edgecolor="none")
ax.set_xlabel("Average Gain Score", fontsize=12)
ax.set_title(
    "All Features Ranked by XGBoost Gain Importance\n"
    "(averaged across day1, day2, day3 targets)",
    fontsize=13
)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(REPORT_DIR / "xgboost_feature_importance.png", dpi=150)
plt.show()
print("Saved → reports/xgboost_feature_importance.png")