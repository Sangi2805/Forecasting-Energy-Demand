from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports"

# TFT first so it is the default. Baselines follow for comparison.
MODELS = {
    "TFT (zonal)": REPORT_DIR / "tft_zonal_predictions.csv",
    "LightGBM": REPORT_DIR / "lgbm_predictions.csv",
    "XGBoost": REPORT_DIR / "xgboost_predictions.csv",
    "Prophet": REPORT_DIR / "prophet_predictions.csv",
    "SARIMAX": REPORT_DIR / "sarimax_predictions.csv",
}

MODEL_COLORS = {
    "Actual": "#1D9E75",
    "TFT (zonal)": "#E8862E",
    "LightGBM": "#4C78A8",
    "XGBoost": "#F58518",
    "Prophet": "#E45756",
    "SARIMAX": "#72B7B2",
}

# The model's headline accuracy is measured hourly (deck figures); this
# dashboard scores on daily totals, which read lower. Shown so the two agree.
TFT_HOURLY_NOTE = "TFT hourly test error: overall 2.95%, Day 1 2.21%."


@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").sort_index()


def available_horizons(df: pd.DataFrame) -> list[int]:
    hs = []
    for n in range(1, 9):
        if f"actual_day{n}" in df.columns and f"pred_day{n}" in df.columns:
            hs.append(n)
    return hs


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    horizons = available_horizons(df)
    metrics = {"n_horizons": len(horizons)}
    for h in horizons:
        actual = df[f"actual_day{h}"]
        pred = df[f"pred_day{h}"]
        errors = actual - pred
        metrics[f"day{h}_mae"] = float(np.mean(np.abs(errors)))
        metrics[f"day{h}_rmse"] = float(np.sqrt(np.mean(errors**2)))
        metrics[f"day{h}_mape"] = float(np.mean(np.abs(errors / actual)) * 100)
    for name in ("mae", "rmse", "mape"):
        metrics[f"avg_{name}"] = float(
            np.mean([metrics[f"day{h}_{name}"] for h in horizons])
        )
    return metrics


@st.cache_data
def load_model_catalog() -> pd.DataFrame:
    rows = []
    for model_name, path in MODELS.items():
        if not path.exists():
            continue
        metrics = compute_metrics(load_predictions(str(path)))
        rows.append({"model": model_name, "path": str(path), **metrics})
    return pd.DataFrame(rows)


def format_model_label(row: pd.Series) -> str:
    days = int(row["n_horizons"])
    return f"{row['model']} — avg RMSE {row['avg_rmse']:,.0f} MW ({days}-day)"


def latest_issue_date(df: pd.DataFrame) -> pd.Timestamp:
    horizons = available_horizons(df)
    cond = pd.Series(True, index=df.index)
    for h in horizons:
        cond &= df[f"actual_day{h}"] > 100_000
    valid = df[cond]
    return valid.index.max() if not valid.empty else df.index.max()


def padded_y_range(values: list[float], padding_ratio: float = 0.1) -> list[float]:
    ymin, ymax = float(np.min(values)), float(np.max(values))
    margin = max((ymax - ymin) * padding_ratio, 2_500)
    return [ymin - margin, ymax + margin]


def build_series(df: pd.DataFrame, issue_date: pd.Timestamp) -> pd.DataFrame:
    row = df.loc[issue_date]
    records = []
    for h in available_horizons(df):
        records.append(
            {
                "horizon": f"Day {h}",
                "horizon_num": h,
                "target_date": issue_date + pd.Timedelta(days=h),
                "actual": float(row[f"actual_day{h}"]),
                "predicted": float(row[f"pred_day{h}"]),
            }
        )
    return pd.DataFrame(records)


def build_table(df: pd.DataFrame, issue_date: pd.Timestamp) -> pd.DataFrame:
    s = build_series(df, issue_date)
    return pd.DataFrame(
        {
            "Horizon": s["horizon"],
            "Target date": s["target_date"].dt.strftime("%Y-%m-%d"),
            "Actual (MW)": s["actual"].round(),
            "Predicted (MW)": s["predicted"].round(),
            "Error (MW)": (s["actual"] - s["predicted"]).round(),
        }
    )


def series_metrics(series: pd.DataFrame) -> tuple[float, float, float]:
    errors = series["actual"] - series["predicted"]
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mape = float(np.mean(np.abs(errors / series["actual"])) * 100)
    return mae, rmse, mape


def plot_single_model_forecast(series: pd.DataFrame, model_name: str) -> go.Figure:
    x_labels = series["target_date"].dt.strftime("%b %d")
    y_values = series["actual"].tolist() + series["predicted"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_labels, y=series["actual"], name="Actual", mode="lines+markers",
        line=dict(color=MODEL_COLORS["Actual"], width=2.5), marker=dict(size=10),
        hovertemplate="%{x}<br>Actual: %{y:,.0f} MW<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x_labels, y=series["predicted"], name=model_name, mode="lines+markers",
        line=dict(color=MODEL_COLORS.get(model_name, "#FF6B35"), width=2.5, dash="dash"),
        marker=dict(size=10),
        hovertemplate="%{x}<br>Predicted: %{y:,.0f} MW<extra></extra>"))
    fig.update_layout(
        xaxis_title="Target date", yaxis_title="Demand (MW)",
        yaxis=dict(range=padded_y_range(y_values), tickformat=","),
        height=460, legend=dict(orientation="h"), margin=dict(t=20, l=10, r=10, b=0))
    return fig


def plot_all_models_forecast(predictions, issue_date, shared_horizons) -> go.Figure:
    def clip(series):
        return series[series["horizon_num"].isin(shared_horizons)]

    reference = clip(build_series(next(iter(predictions.values())), issue_date))
    x_labels = reference["target_date"].dt.strftime("%b %d")
    y_values = reference["actual"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_labels, y=reference["actual"], name="Actual", mode="lines+markers",
        line=dict(color=MODEL_COLORS["Actual"], width=2.5), marker=dict(size=10),
        hovertemplate="%{x}<br>Actual: %{y:,.0f} MW<extra></extra>"))
    for model_name, df in predictions.items():
        s = clip(build_series(df, issue_date))
        y_values += s["predicted"].tolist()
        fig.add_trace(go.Scatter(
            x=x_labels, y=s["predicted"], name=model_name, mode="lines+markers",
            line=dict(color=MODEL_COLORS.get(model_name, "#999999"), width=2.5),
            marker=dict(size=10),
            hovertemplate=f"%{{x}}<br>{model_name}: %{{y:,.0f}} MW<extra></extra>"))
    fig.update_layout(
        xaxis_title="Target date", yaxis_title="Demand (MW)",
        yaxis=dict(range=padded_y_range(y_values), tickformat=","),
        height=460, legend=dict(orientation="h"), margin=dict(t=20, l=10, r=10, b=0))
    return fig


def main() -> None:
    st.set_page_config(page_title="NY Energy Demand Forecast", layout="wide",
                       initial_sidebar_state="collapsed")
    st.title("Energy Demand Forecast — New York State")
    st.caption("Zonal Temporal Fusion Transformer, scored on the held-out test set. "
               + TFT_HOURLY_NOTE)

    catalog = load_model_catalog()
    if catalog.empty:
        st.error("No prediction files found in `reports/`.")
        st.stop()

    catalog["label"] = catalog.apply(format_model_label, axis=1)
    default_idx = (int(catalog.index[catalog["model"] == "TFT (zonal)"][0])
                   if (catalog["model"] == "TFT (zonal)").any()
                   else int(catalog["avg_rmse"].idxmin()))

    model_row = catalog.iloc[
        st.selectbox("Model", range(len(catalog)),
                     format_func=lambda i: catalog.iloc[i]["label"], index=default_idx)]
    df = load_predictions(model_row["path"])
    issue_date = latest_issue_date(df)
    series = build_series(df, issue_date)
    mae, rmse, mape = series_metrics(series)
    ndays = len(available_horizons(df))

    st.subheader("Demand values")
    st.caption(f"Issue date: {issue_date.strftime('%Y-%m-%d')}")
    st.dataframe(build_table(df, issue_date), use_container_width=True, hide_index=True)

    st.subheader(f"Model performance ({ndays}-day window, daily total)")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:,.0f} MW")
    m2.metric("RMSE", f"{rmse:,.0f} MW")
    m3.metric("MAPE", f"{mape:.2f}%")

    st.subheader("Actual vs predicted demand")
    st.plotly_chart(plot_single_model_forecast(series, model_row["model"]),
                    use_container_width=True)

    with st.expander("Compare all models (full test set, daily total)", expanded=True):
        cols = ["model", "avg_mae", "avg_rmse", "avg_mape"] + \
               [f"day{d}_mape" for d in range(1, 6)]
        compare = catalog.reindex(columns=cols).rename(columns={
            "model": "Model", "avg_mae": "Avg MAE (MW)", "avg_rmse": "Avg RMSE (MW)",
            "avg_mape": "Avg MAPE (%)", "day1_mape": "Day 1 (%)", "day2_mape": "Day 2 (%)",
            "day3_mape": "Day 3 (%)", "day4_mape": "Day 4 (%)", "day5_mape": "Day 5 (%)"})
        fmt = {"Avg MAE (MW)": "{:,.0f}", "Avg RMSE (MW)": "{:,.0f}", "Avg MAPE (%)": "{:.2f}"}
        fmt.update({f"Day {d} (%)": "{:.2f}" for d in range(1, 6)})
        st.dataframe(compare.style.format(fmt, na_rep="—"),
                     use_container_width=True, hide_index=True)

        all_predictions = {r["model"]: load_predictions(r["path"])
                           for _, r in catalog.iterrows()}
        shared = sorted(set.intersection(
            *(set(available_horizons(d)) for d in all_predictions.values())))
        st.subheader(f"All models vs actual ({len(shared)}-day overlap)")
        st.caption(f"Issue date: {issue_date.strftime('%Y-%m-%d')}")
        st.plotly_chart(plot_all_models_forecast(all_predictions, issue_date, shared),
                        use_container_width=True)


if __name__ == "__main__":
    main()