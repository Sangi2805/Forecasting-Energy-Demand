import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np


@st.cache_data
def load_forecast_data():
    # Energy data
    energy = pd.read_excel("data/raw/Region_NY.xlsx")
    energy = energy[["UTC time", "Demand", "Demand forecast"]].copy()
    energy.columns = ["timestamp", "actual", "forecast"]
    energy["timestamp"] = pd.to_datetime(energy["timestamp"], utc=True)
    energy = energy.dropna(subset=["actual"])

    # Weather data
    weather = pd.read_csv("data/raw/weather.csv", skiprows=3)
    weather = weather[["time", "temperature_2m (°C)", "wind_speed_10m (km/h)", "precipitation (mm)"]].copy()
    weather.columns = ["timestamp", "temperature", "wind_speed", "precipitation"]
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True)

    # Merge
    df = pd.merge(energy, weather, on="timestamp", how="inner")
    return df


# --- UI ---
st.set_page_config(page_title="Energy Demand Forecast", layout="wide")
st.title("⚡ Energy Demand Forecast — NY Region")

df = load_forecast_data()

# Sidebar filters
st.sidebar.header("Filters")
min_date, max_date = df["timestamp"].dt.date.min(), df["timestamp"].dt.date.max()
start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2015-08-01").date(), min_value=min_date, max_value=max_date)

# Filter
mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
filtered = df[mask]

if filtered.empty:
    st.warning("No data for selected range.")
    st.stop()

# --- Forecast Chart ---
st.subheader("Actual vs Forecast Demand")
fig = go.Figure()
fig.add_trace(go.Scatter(x=filtered["timestamp"], y=filtered["actual"], name="Actual Demand", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=filtered["timestamp"], y=filtered["forecast"], name="Forecast", line=dict(color="#ff7f0e", dash="dash")))
fig.update_layout(xaxis_title="Time", yaxis_title="Demand (MWh)", height=400, legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# --- Metrics ---
st.subheader("Model Performance")
mae = np.mean(np.abs(filtered["actual"] - filtered["forecast"]))
rmse = np.sqrt(np.mean((filtered["actual"] - filtered["forecast"]) ** 2))
mape = np.mean(np.abs((filtered["actual"] - filtered["forecast"]) / filtered["actual"])) * 100

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:,.0f} MWh")
m2.metric("RMSE", f"{rmse:,.0f} MWh")
m3.metric("MAPE", f"{mape:.2f}%")

# --- Weather Context ---
st.subheader("Weather Context")
col1, col2 = st.columns(2)

with col1:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=filtered["timestamp"], y=filtered["temperature"], name="Temperature (°C)", line=dict(color="#d62728")))
    fig2.update_layout(height=300, yaxis_title="°C")
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=filtered["timestamp"], y=filtered["precipitation"], name="Precipitation (mm)", marker_color="#2ca02c"))
    fig3.update_layout(height=300, yaxis_title="mm")
    st.plotly_chart(fig3, use_container_width=True)