import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np


@st.cache_data
def load_predictions():
    return pd.read_csv(
        "reports/xgboost_predictions.csv",
        index_col="date",
        parse_dates=["date"]
    )


st.set_page_config(page_title="Energy Demand Forecast", layout="wide")
st.title("⚡ Energy Demand Forecast — NY Region")

df = load_predictions()

st.sidebar.header("Filters")
start_date = st.sidebar.date_input("Start date", value=df.index.min(), min_value=df.index.min(), max_value=df.index.max())
end_date = st.sidebar.date_input("End date", value=df.index.max(), min_value=df.index.min(), max_value=df.index.max())

col1, col2 = st.columns([2, 1])
with col1:
    st.selectbox("Model", [
        "XGBoost — avg RMSE 29,106",
        "Prophet — coming soon",
        "SARIMAX — coming soon",
        "Ensemble — coming soon"
    ])
with col2:
    day = st.radio("Forecast horizon", ["Day 1", "Day 2", "Day 3"], horizontal=True)

day_num = int(day.split()[-1])
pred_col = f"pred_day{day_num}"
actual_col = f"actual_day{day_num}"

mask = (df.index.date >= start_date) & (df.index.date <= end_date)
filtered = df[mask]

if filtered.empty:
    st.warning("No data for the selected date range.")
    st.stop()

st.subheader("Demand values")
table = pd.DataFrame({
    "Forecast date": (filtered.index + pd.Timedelta(days=day_num)).strftime("%Y-%m-%d"),
    "Actual (MWh)": filtered[actual_col].astype(int),
    "Predicted (MWh)": filtered[pred_col].round(0).astype(int),
    "Error (MWh)": (filtered[actual_col] - filtered[pred_col]).round(0).astype(int)
})
st.dataframe(table, use_container_width=True, hide_index=True)

st.subheader("Model performance")
errors = filtered[actual_col] - filtered[pred_col]
mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors ** 2))
mape = np.mean(np.abs(errors / filtered[actual_col])) * 100

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:,.0f} MWh")
m2.metric("RMSE", f"{rmse:,.0f} MWh")
m3.metric("MAPE", f"{mape:.2f}%")

st.subheader("Actual vs predicted demand")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=filtered.index + pd.Timedelta(days=day_num),
    y=filtered[actual_col],
    name="Actual",
    line=dict(color="#1D9E75", width=1.5)
))
fig.add_trace(go.Scatter(
    x=filtered.index + pd.Timedelta(days=day_num),
    y=filtered[pred_col],
    name="Predicted",
    line=dict(color="#FF6B35", width=1.5, dash="dash")
))
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Demand (MWh)",
    height=420,
    legend=dict(orientation="h"),
    margin=dict(t=20, l=0, r=0, b=0)
)
st.plotly_chart(fig, use_container_width=True)