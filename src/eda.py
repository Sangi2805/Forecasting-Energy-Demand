import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves files, no display needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import config as cfg

sns.set_theme(style="whitegrid", palette="muted")

REPORT_DIR = cfg.REPORT_DIR

##########################################################################
# Helpers
##########################################################################

def _load_full_train() -> pd.DataFrame:
    df = pd.read_parquet(cfg.FULL_TRAIN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

def _save(fig, name: str) -> None:
    path = REPORT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")

##########################################################################
# 1. Demand over time
##########################################################################

def plot_demand_over_time(df: pd.DataFrame = None, rolling_window: int = 30) -> None:
    if df is None:
        df = _load_full_train()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["date"], df["demand"], alpha=0.4, linewidth=0.8, label="Daily demand")
    roll = df.set_index("date")["demand"].rolling(rolling_window).mean()
    ax.plot(roll.index, roll.values, color="tomato", linewidth=1.8,
            label=f"{rolling_window}-day rolling mean")

    ax.set_title("NY Daily Electricity Demand (MWh)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (MWh)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend()
    plt.tight_layout()
    _save(fig, "demand_over_time")
    plt.show()

##########################################################################
# 2. Monthly demand boxplot
##########################################################################

def plot_monthly_boxplot(df: pd.DataFrame = None) -> None:
    if df is None:
        df = _load_full_train()

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(13, 5))
    data = [df[df["month"] == m]["demand"].values for m in range(1, 13)]
    bp = ax.boxplot(data, patch_artist=True, tick_labels=month_labels)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)

    ax.set_title("Daily Demand Distribution by Month", fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Demand (MWh)")
    plt.tight_layout()
    _save(fig, "monthly_demand_boxplot")
    plt.show()

##########################################################################
# 3. Weekday demand boxplot
##########################################################################

def plot_weekday_boxplot(df: pd.DataFrame = None) -> None:
    if df is None:
        df = _load_full_train()

    day_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [df[df["day_of_week"] == d]["demand"].values for d in range(7)]
    bp = ax.boxplot(data, patch_artist=True, tick_labels=day_labels)
    for patch in bp["boxes"]:
        patch.set_facecolor("mediumseagreen")
        patch.set_alpha(0.7)

    ax.set_title("Daily Demand Distribution by Weekday", fontsize=14)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Demand (MWh)")
    plt.tight_layout()
    _save(fig, "weekday_demand_boxplot")
    plt.show()

##########################################################################
# 4. Annual trend
##########################################################################

def plot_annual_demand(df: pd.DataFrame = None) -> None:
    if df is None:
        df = _load_full_train()

    annual = df.groupby("year")["demand"].agg(["mean","std"]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(annual["year"], annual["mean"], yerr=annual["std"],
           color="steelblue", alpha=0.7, capsize=4)
    ax.set_title("Mean Daily Demand by Year (+/- 1 SD)", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Demand (MWh)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    _save(fig, "annual_demand_trend")
    plt.show()

##########################################################################
# 5. Key weather vs demand scatter (2x2 grid)
##########################################################################

def plot_weather_demand_scatter(df: pd.DataFrame = None) -> None:
    if df is None:
        df = _load_full_train()

    features = [
        ("temperature_2m",       "Avg Temp (C)"),
        ("apparent_temperature", "Apparent Temp (C)"),
        ("relative_humidity_2m", "Humidity (%)"),
        ("wind_speed_10m",       "Wind Speed (km/h)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (col, label) in zip(axes.flat, features):
        ax.scatter(df[col], df["demand"], alpha=0.15, s=8, color="steelblue")
        m, b = np.polyfit(df[col], df["demand"], 1)
        xs = np.linspace(df[col].min(), df[col].max(), 100)
        ax.plot(xs, m*xs + b, color="tomato", linewidth=1.5)
        corr = df[col].corr(df["demand"])
        ax.set_title(f"{label}  (r = {corr:.3f})", fontsize=11)
        ax.set_xlabel(label)
        ax.set_ylabel("Demand (MWh)")

    fig.suptitle("Key Weather Features vs Daily Demand", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "weather_demand_scatter")
    plt.show()

##########################################################################
# 6. Feature → target correlation bar chart
##########################################################################

def plot_feature_target_correlation(
    df: pd.DataFrame = None,
    target: str = "target_day1",
    top_n: int = 30,
) -> None:
    if df is None:
        df = _load_full_train()

    # Exclude date, target cols, and year (proxy for trend, not a real feature)
    exclude = {"date", "year"} | set(cfg.TARGET_COLS)
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]

    corrs = df[num_cols].corrwith(df[target]).abs().sort_values(ascending=False)
    corrs_signed = df[num_cols].corrwith(df[target]).loc[corrs.index]
    top = corrs.head(top_n)
    colors = ["tomato" if corrs_signed[c] < 0 else "steelblue" for c in top.index]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    bars = ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], alpha=0.8)
    ax.set_xlabel("Absolute Pearson Correlation")
    ax.set_title(f"Feature Correlations with {target} (top {top_n})", fontsize=13)
    ax.axvline(0.3, color="gray", linestyle="--", linewidth=0.8, label="r=0.3")
    ax.axvline(0.5, color="gray", linestyle="-",  linewidth=0.8, label="r=0.5")
    ax.legend(fontsize=9)

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor="steelblue", label="Positive"), Patch(facecolor="tomato", label="Negative")]
    ax.legend(handles=legend_els + ax.get_legend_handles_labels()[0][1:], fontsize=9)

    plt.tight_layout()
    _save(fig, f"feature_target_correlation_{target}")
    plt.show()

    print(f"\nTop 10 correlations with {target}:")
    print(corrs_signed.head(10).round(3).to_string())

##########################################################################
# 7. Feature–feature correlation heatmap
##########################################################################

def plot_feature_correlation_heatmap(
    df: pd.DataFrame = None,
    feature_group: str = "all",
) -> None:
    """
    feature_group: 'all', 'weather', 'demand_lags', 'temporal'
    """
    if df is None:
        df = _load_full_train()

    exclude = {"date", "year"} | set(cfg.TARGET_COLS)

    if feature_group == "weather":
        cols = [c for c in df.columns if any(c.startswith(p) for p in
                ["temperature","apparent","relative_humidity","dew_point",
                 "precipitation","rain","snowfall","cloud","wind","snow_depth","weather_code"])]
    elif feature_group == "demand_lags":
        cols = [c for c in df.columns if c.startswith("demand_lag") or c.startswith("demand_roll")]
    elif feature_group == "temporal":
        cols = ["month","day_of_month","day_of_week","week_of_year","quarter","is_weekend","is_holiday"]
    else:
        cols = [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]
        # Limit to top 35 by variance to keep heatmap readable
        variances = df[cols].var().sort_values(ascending=False)
        cols = variances.head(35).index.tolist()

    corr_matrix = df[cols].corr()

    size = max(10, len(cols) * 0.4)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, ax=ax,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=(len(cols) <= 20), fmt=".2f", annot_kws={"size": 7},
        linewidths=0.3, square=True,
        cbar_kws={"shrink": 0.6}
    )
    ax.set_title(f"Feature Correlation Heatmap ({feature_group})", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    _save(fig, f"feature_correlation_heatmap_{feature_group}")
    plt.show()

##########################################################################
# 8. Demand lag autocorrelation
##########################################################################

def plot_demand_lag_correlations(df: pd.DataFrame = None) -> None:
    if df is None:
        df = _load_full_train()

    lag_cols = [c for c in df.columns if c.startswith("demand_lag_")]
    corrs = {col: df[col].corr(df["target_day1"]) for col in lag_cols}
    labels = [c.replace("demand_lag_", "").replace("d", " days") for c in corrs]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, list(corrs.values()), color="steelblue", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Demand Lag Features vs target_day1 (Pearson r)", fontsize=13)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Correlation")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    _save(fig, "demand_lag_autocorrelation")
    plt.show()

##########################################################################
# 9. Target distribution
##########################################################################

def plot_target_distributions(df: pd.DataFrame = None) -> None:
    if df is None:
        df = _load_full_train()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, col in zip(axes, cfg.TARGET_COLS):
        ax.hist(df[col], bins=50, color="steelblue", alpha=0.75, edgecolor="white")
        ax.axvline(df[col].mean(), color="tomato", linewidth=1.5, label=f"mean={df[col].mean():,.0f}")
        ax.set_title(col)
        ax.set_xlabel("Demand (MWh)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Count")
    fig.suptitle("Target Variable Distributions", fontsize=13)
    plt.tight_layout()
    _save(fig, "target_distributions")
    plt.show()

##########################################################################
# Run all
##########################################################################

def run_all_eda() -> None:
    df = _load_full_train()
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")

    print("\n[1/8] Demand over time...")
    plot_demand_over_time(df)

    print("\n[2/8] Monthly boxplot...")
    plot_monthly_boxplot(df)

    print("\n[3/8] Weekday boxplot...")
    plot_weekday_boxplot(df)

    print("\n[4/8] Annual trend...")
    plot_annual_demand(df)

    print("\n[5/8] Weather vs demand scatter...")
    plot_weather_demand_scatter(df)

    print("\n[6/8] Feature-target correlation...")
    plot_feature_target_correlation(df, target="target_day1", top_n=30)

    print("\n[7/8] Feature-feature heatmaps...")
    plot_feature_correlation_heatmap(df, feature_group="weather")
    plot_feature_correlation_heatmap(df, feature_group="demand_lags")
    plot_feature_correlation_heatmap(df, feature_group="temporal")

    print("\n[8/8] Demand lag autocorrelation + target distributions...")
    plot_demand_lag_correlations(df)
    plot_target_distributions(df)

    print("\nAll plots saved to reports/")


if __name__ == "__main__":
    run_all_eda()
