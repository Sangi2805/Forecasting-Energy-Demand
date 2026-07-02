from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports"
FORECAST_DAYS = (1, 2, 3)

MODELS = {
    "LightGBM": REPORT_DIR / "lgbm_predictions.csv",
    "XGBoost": REPORT_DIR / "xgboost_predictions.csv",
    "Prophet": REPORT_DIR / "prophet_predictions.csv",
    "SARIMAX": REPORT_DIR / "sarimax_predictions.csv",
}

MODEL_COLORS = {
    "Actual": "#1D9E75",
    "LightGBM": "#4C78A8",
    "XGBoost": "#F58518",
    "Prophet": "#E45756",
    "SARIMAX": "#72B7B2",
}


@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").sort_index()


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    metrics = {}
    for horizon in FORECAST_DAYS:
        actual = df[f"actual_day{horizon}"]
        pred = df[f"pred_day{horizon}"]
        errors = actual - pred
        metrics[f"day{horizon}_mae"] = float(np.mean(np.abs(errors)))
        metrics[f"day{horizon}_rmse"] = float(np.sqrt(np.mean(errors**2)))
        metrics[f"day{horizon}_mape"] = float(np.mean(np.abs(errors / actual)) * 100)

    for name in ("mae", "rmse", "mape"):
        metrics[f"avg_{name}"] = float(
            np.mean([metrics[f"day{h}_{name}"] for h in FORECAST_DAYS])
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
    return f"{row['model']} — avg RMSE {row['avg_rmse']:,.0f} MW"


def latest_issue_date(df: pd.DataFrame) -> pd.Timestamp:
    """Use the latest row with plausible demand on all three horizons."""
    valid = df[
        (df["actual_day1"] > 100_000)
        & (df["actual_day2"] > 100_000)
        & (df["actual_day3"] > 100_000)
    ]
    if valid.empty:
        return df.index.max()
    return valid.index.max()


def padded_y_range(values: list[float], padding_ratio: float = 0.1) -> list[float]:
    ymin = float(np.min(values))
    ymax = float(np.max(values))
    span = ymax - ymin
    margin = max(span * padding_ratio, 2_500)
    return [ymin - margin, ymax + margin]


def axis_values_for_compare(
    actual_series: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    issue_date: pd.Timestamp,
    max_rel_error: float = 0.25,
) -> list[float]:
    """Build a tight y-axis range from actuals and plausible model predictions."""
    values = actual_series["actual"].tolist()
    for df in predictions.values():
        series = build_three_day_series(df, issue_date)
        for pred, actual in zip(series["predicted"], series["actual"]):
            if actual <= 0:
                continue
            if abs(pred - actual) / actual <= max_rel_error:
                values.append(float(pred))
    return values


def build_three_day_table(df: pd.DataFrame, issue_date: pd.Timestamp) -> pd.DataFrame:
    row = df.loc[issue_date]
    records = []
    for horizon in FORECAST_DAYS:
        actual = float(row[f"actual_day{horizon}"])
        predicted = float(row[f"pred_day{horizon}"])
        records.append(
            {
                "Horizon": f"Day {horizon}",
                "Target date": (issue_date + pd.Timedelta(days=horizon)).strftime(
                    "%Y-%m-%d"
                ),
                "Actual (MW)": round(actual),
                "Predicted (MW)": round(predicted),
                "Error (MW)": round(actual - predicted),
            }
        )
    return pd.DataFrame(records)


def build_three_day_series(df: pd.DataFrame, issue_date: pd.Timestamp) -> pd.DataFrame:
    row = df.loc[issue_date]
    records = []
    for horizon in FORECAST_DAYS:
        records.append(
            {
                "horizon": f"Day {horizon}",
                "horizon_num": horizon,
                "target_date": issue_date + pd.Timedelta(days=horizon),
                "actual": float(row[f"actual_day{horizon}"]),
                "predicted": float(row[f"pred_day{horizon}"]),
            }
        )
    return pd.DataFrame(records)


def three_day_metrics(series: pd.DataFrame) -> tuple[float, float, float]:
    errors = series["actual"] - series["predicted"]
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mape = float(np.mean(np.abs(errors / series["actual"])) * 100)
    return mae, rmse, mape


def plot_single_model_forecast(series: pd.DataFrame, model_name: str) -> go.Figure:
    x_labels = series["target_date"].dt.strftime("%b %d")
    y_values = series["actual"].tolist() + series["predicted"].tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=series["actual"],
            name="Actual",
            mode="lines+markers",
            line=dict(color=MODEL_COLORS["Actual"], width=2.5),
            marker=dict(size=10),
            hovertemplate="%{x}<br>Actual: %{y:,.0f} MW<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=series["predicted"],
            name=model_name,
            mode="lines+markers",
            line=dict(color=MODEL_COLORS.get(model_name, "#FF6B35"), width=2.5, dash="dash"),
            marker=dict(size=10),
            hovertemplate="%{x}<br>Predicted: %{y:,.0f} MW<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Target date",
        yaxis_title="Demand (MW)",
        yaxis=dict(range=padded_y_range(y_values), tickformat=","),
        height=460,
        legend=dict(orientation="h"),
        margin=dict(t=20, l=10, r=10, b=0),
    )
    return fig


def plot_all_models_forecast(
    predictions: dict[str, pd.DataFrame],
    issue_date: pd.Timestamp,
) -> go.Figure:
    reference = next(iter(predictions.values()))
    actual_series = build_three_day_series(reference, issue_date)
    x_labels = actual_series["target_date"].dt.strftime("%b %d")
    y_values = axis_values_for_compare(actual_series, predictions, issue_date)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=actual_series["actual"],
            name="Actual",
            mode="lines+markers",
            line=dict(color=MODEL_COLORS["Actual"], width=2.5),
            marker=dict(size=10),
            hovertemplate="%{x}<br>Actual: %{y:,.0f} MW<extra></extra>",
        )
    )

    for model_name, df in predictions.items():
        series = build_three_day_series(df, issue_date)
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=series["predicted"],
                name=model_name,
                mode="lines+markers",
                line=dict(color=MODEL_COLORS.get(model_name, "#999999"), width=2.5),
                marker=dict(size=10),
                hovertemplate=f"%{{x}}<br>{model_name}: %{{y:,.0f}} MW<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Target date",
        yaxis_title="Demand (MW)",
        yaxis=dict(range=padded_y_range(y_values), tickformat=","),
        height=460,
        legend=dict(orientation="h"),
        margin=dict(t=20, l=10, r=10, b=0),
    )
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Energy Demand Forecast",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("Energy Demand Forecast — NY Region")
    st.caption("Latest 3-day ahead forecast window from the held-out test set.")

    catalog = load_model_catalog()
    if catalog.empty:
        st.error(
            "No prediction files found in `reports/`. "
            "Run the training scripts first, e.g. `python -m src.lgbm.train_lgbm`."
        )
        st.stop()

    catalog["label"] = catalog.apply(format_model_label, axis=1)
    best_idx = int(catalog["avg_rmse"].idxmin())

    model_row = catalog.iloc[
        st.selectbox(
            "Model",
            range(len(catalog)),
            format_func=lambda i: catalog.iloc[i]["label"],
            index=best_idx,
        )
    ]
    df = load_predictions(model_row["path"])
    issue_date = latest_issue_date(df)
    three_day_table = build_three_day_table(df, issue_date)
    three_day_series = build_three_day_series(df, issue_date)
    mae, rmse, mape = three_day_metrics(three_day_series)

    st.subheader("Demand values")
    st.caption(f"Issue date: {issue_date.strftime('%Y-%m-%d')}")
    st.dataframe(three_day_table, use_container_width=True, hide_index=True)

    st.subheader("Model performance (3-day window)")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:,.0f} MW")
    m2.metric("RMSE", f"{rmse:,.0f} MW")
    m3.metric("MAPE", f"{mape:.2f}%")

    st.subheader("Actual vs predicted demand")
    st.plotly_chart(
        plot_single_model_forecast(three_day_series, model_row["model"]),
        use_container_width=True,
    )

    with st.expander("Compare all models (full test set)", expanded=True):
        compare = catalog[
            ["model", "avg_mae", "avg_rmse", "avg_mape", "day1_mape", "day2_mape", "day3_mape"]
        ].copy()
        compare = compare.rename(
            columns={
                "model": "Model",
                "avg_mae": "Avg MAE (MW)",
                "avg_rmse": "Avg RMSE (MW)",
                "avg_mape": "Avg MAPE (%)",
                "day1_mape": "Day 1 MAPE (%)",
                "day2_mape": "Day 2 MAPE (%)",
                "day3_mape": "Day 3 MAPE (%)",
            }
        )
        st.dataframe(
            compare.style.format(
                {
                    "Avg MAE (MW)": "{:,.0f}",
                    "Avg RMSE (MW)": "{:,.0f}",
                    "Avg MAPE (%)": "{:.2f}",
                    "Day 1 MAPE (%)": "{:.2f}",
                    "Day 2 MAPE (%)": "{:.2f}",
                    "Day 3 MAPE (%)": "{:.2f}",
                },
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("All models vs actual (3-day window)")
        st.caption(f"Issue date: {issue_date.strftime('%Y-%m-%d')}")
        all_predictions = {
            row["model"]: load_predictions(row["path"])
            for _, row in catalog.iterrows()
        }
        st.plotly_chart(
            plot_all_models_forecast(all_predictions, issue_date),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
