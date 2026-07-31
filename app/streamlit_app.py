"""
EnergyAI — NYISO zonal demand dashboard.

Two tabs over the same zonal TFT checkpoint (refit2023+fx, 2.95% held-out MAPE):

  Forecasting     banked predictions over the held-out test period, scored
                  against realised demand.
  Live forecast   the same model run now on NYISO palIntegrated load and
                  Open-Meteo forecast weather, 120 hours ahead, alongside
                  NYISO's own published forecast.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# `streamlit run app/streamlit_app.py` puts app/ on sys.path, not the repo root,
# so the sibling modules are not importable as a package without this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import live_forecast as lfc  # noqa: E402  (needs the path fix above)

# ---------------------------------------------------------------------------
# Paths & zone metadata
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
# The refit2023+fx metrics describe the checkpoint actually being served.
# This previously pointed at metrics_zonal_lr3e4.json -- the superseded 3.196%
# model -- so every quoted accuracy was for a checkpoint no longer in use.
METRICS_PATH = ROOT / "Sangar" / "metrics_zonal_refit2023_fx.json"

# Approximate load-zone centroids (NYISO A–K)
ZONE_META = {
    "WEST": {"code": "A", "label": "West (Buffalo)", "lat": 42.886, "lon": -78.878},
    "GENESE": {"code": "B", "label": "Genesee (Rochester)", "lat": 43.157, "lon": -77.616},
    "CENTRL": {"code": "C", "label": "Central (Syracuse)", "lat": 43.048, "lon": -76.147},
    "NORTH": {"code": "D", "label": "North (Massena)", "lat": 44.928, "lon": -74.892},
    "MHK VL": {"code": "E", "label": "Mohawk Valley (Utica)", "lat": 43.101, "lon": -75.233},
    "CAPITL": {"code": "F", "label": "Capital (Albany)", "lat": 42.653, "lon": -73.757},
    "HUD VL": {"code": "G", "label": "Hudson Valley", "lat": 41.706, "lon": -73.921},
    "MILLWD": {"code": "H", "label": "Millwood", "lat": 41.200, "lon": -73.780},
    "DUNWOD": {"code": "I", "label": "Dunwoodie", "lat": 40.950, "lon": -73.850},
    "N.Y.C.": {"code": "J", "label": "New York City", "lat": 40.714, "lon": -74.006},
    "LONGIL": {"code": "K", "label": "Long Island", "lat": 40.730, "lon": -73.210},
}
ZONE_COLS = list(ZONE_META.keys())

# Professional palette (slate / teal — not purple-default AI look)
COLORS = {
    "bg": "#0f1419",
    "panel": "#1a222c",
    "border": "#2a3544",
    "text": "#e8eef4",
    "muted": "#8b9aab",
    "accent": "#1a9b8e",
    "accent2": "#e8a838",
    "danger": "#d64545",
    "actual": "#3d9be9",
    "forecast": "#e8a838",
    "live": "#1a9b8e",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color=COLORS["text"], size=13),
    margin=dict(l=48, r=24, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    xaxis=dict(gridcolor=COLORS["border"], zeroline=False),
    yaxis=dict(gridcolor=COLORS["border"], zeroline=False),
)


# ---------------------------------------------------------------------------
# Banked model predictions (held-out test period)
#
# One definition per function. This region previously held seasonal-naive mock
# generators that were then redefined twice at import time by real-data
# overrides, so which implementation ran depended on definition order rather
# than anything explicit. The predictions parquet ships with the repo, so the
# placeholders had nothing left to fall back for.
# ---------------------------------------------------------------------------
PRED_PATH = ROOT / "reports" / "tft_hourly_predictions.parquet"


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    """Banked zonal TFT predictions, pivoted wide with a statewide total."""
    if not PRED_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(PRED_PATH)
    df["issue_date"] = pd.to_datetime(df["issue_date"]).dt.date
    ts = pd.to_datetime(df["ts_utc"])
    df["ts_utc"] = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")

    idx = ["issue_date", "lead_day", "ts_utc"]
    pred = df.pivot_table(index=idx, columns="zone", values="pred_mw", aggfunc="mean")
    actual = df.pivot_table(index=idx, columns="zone", values="actual_mw", aggfunc="mean")
    zones = [z for z in ZONE_COLS if z in pred.columns]
    pred, actual = pred[zones].copy(), actual[zones].copy()
    pred["NYISO_TOTAL"] = pred.sum(axis=1)
    actual["NYISO_TOTAL"] = actual.sum(axis=1)
    pred.columns = [f"pred_{c}" for c in pred.columns]
    actual.columns = [f"actual_{c}" for c in actual.columns]

    out = actual.join(pred).reset_index()
    out["actual"] = out["actual_NYISO_TOTAL"]
    out["predicted"] = out["pred_NYISO_TOTAL"]
    return out


@st.cache_data(show_spinner=False)
def model_metrics() -> dict:
    """Held-out scores for the checkpoint actually being served."""
    try:
        raw = json.loads(METRICS_PATH.read_text())
    except (OSError, ValueError):
        return {"day_mape": {}, "overall_mape": float("nan"), "seasonal_mape": {}}
    return {
        "day_mape": {int(k): float(v) for k, v in raw.get("day_mape", {}).items()},
        "overall_mape": raw.get("overall_mape", float("nan")),
        "seasonal_mape": raw.get("seasonal_mape", {}),
        "checkpoint": raw.get("checkpoint", ""),
    }


PREDICTIONS = load_predictions()
if PREDICTIONS.empty:
    st.error(
        f"Prediction archive not found at `{PRED_PATH}`. The Forecasting tab "
        "reads banked zonal TFT output; regenerate it with "
        "`Sangar/09_hourly_predictions.ipynb`."
    )
    st.stop()

ISSUE_DATES = sorted(PREDICTIONS["issue_date"].unique())
ISSUE_SET = set(ISSUE_DATES)
_full = PREDICTIONS.groupby("issue_date")["lead_day"].nunique()
_full = _full[_full >= 5]
LATEST_ISSUE = _full.index.max() if len(_full) else ISSUE_DATES[-1]


def selected_issue() -> date:
    """Issue date chosen in the Forecasting tab, snapped to one that exists."""
    want = st.session_state.get("sel_issue", LATEST_ISSUE)
    if want in ISSUE_SET:
        return want
    earlier = [d for d in ISSUE_DATES if d <= want]
    return earlier[-1] if earlier else ISSUE_DATES[0]


def day_block(day: int, issue: date | None = None) -> pd.DataFrame:
    """The 24 hourly rows of one lead day within one issued forecast."""
    issue = selected_issue() if issue is None else issue
    g = PREDICTIONS[
        (PREDICTIONS["issue_date"] == issue) & (PREDICTIONS["lead_day"] == int(day))
    ]
    if g.empty:
        return pd.DataFrame()
    return (g.set_index("ts_utc")
             .drop(columns=["issue_date", "lead_day"])
             .sort_index())


def forecast_origin() -> pd.Timestamp:
    """First forecast hour of the selected issue."""
    g = PREDICTIONS[
        (PREDICTIONS["issue_date"] == selected_issue()) & (PREDICTIONS["lead_day"] == 1)
    ]
    return g["ts_utc"].min()


@st.cache_data(show_spinner=False)
def mape_by_lead() -> dict[int, float]:
    """
    Statewide MAPE per lead day across every issued forecast.

    Scored on lead_day == n, not lead_day <= n. The cumulative form compared a
    single-day figure against a 1..n average, which flattered later horizons.
    """
    out = {}
    for n in range(1, 6):
        g = PREDICTIONS[PREDICTIONS["lead_day"] == n]
        a = g["actual_NYISO_TOTAL"].to_numpy(dtype=float)
        p = g["pred_NYISO_TOTAL"].to_numpy(dtype=float)
        out[n] = float(np.mean(np.abs(a - p) / np.maximum(a, 1.0)) * 100)
    return out


def day_summary(day_df: pd.DataFrame, actual_col: str = "actual",
                pred_col: str = "predicted") -> dict[str, float]:
    actual, predicted = day_df[actual_col], day_df[pred_col]
    mape = float(np.mean(np.abs(actual - predicted) / actual.clip(lower=1)) * 100)
    return {
        "actual_mw": float(actual.mean()),
        "predicted_mw": float(predicted.mean()),
        "error_pct": mape,
        "actual_mwh": float(actual.sum()),
        "predicted_mwh": float(predicted.sum()),
    }

def forecast_scope_columns(scope: str) -> tuple[str, str, str]:
    """Return (actual_col, pred_col, display_label) for Statewide or a zone."""
    if scope == "Statewide":
        return "actual_NYISO_TOTAL", "pred_NYISO_TOTAL", "Statewide"
    meta = ZONE_META[scope]
    return f"actual_{scope}", f"pred_{scope}", f"{meta['code']} — {meta['label']}"


def zone_day_consumption(day_df: pd.DataFrame) -> pd.DataFrame:
    """Per-zone energy (MWh ≈ sum of hourly MW) and mean MW for the selected day."""
    rows = []
    for zone, meta in ZONE_META.items():
        actual_mwh = float(day_df[f"actual_{zone}"].sum())
        pred_mwh = float(day_df[f"pred_{zone}"].sum())
        rows.append(
            {
                "zone": zone,
                "code": meta["code"],
                "label": meta["label"],
                "lat": meta["lat"],
                "lon": meta["lon"],
                "actual_mwh": actual_mwh,
                "predicted_mwh": pred_mwh,
                "actual_avg_mw": float(day_df[f"actual_{zone}"].mean()),
                "predicted_avg_mw": float(day_df[f"pred_{zone}"].mean()),
                "error_pct": float(
                    np.mean(
                        np.abs(day_df[f"actual_{zone}"] - day_df[f"pred_{zone}"])
                        / day_df[f"actual_{zone}"].clip(lower=1)
                    )
                    * 100
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["share_pct"] = 100 * out["actual_mwh"] / out["actual_mwh"].sum()
    return out


# ---------------------------------------------------------------------------
# Live forecast helpers
# ---------------------------------------------------------------------------
ET = "America/New_York"
PRED_ARCHIVE = ROOT / "reports" / "tft_hourly_predictions.parquet"

# Measured on 22 held-out origins x 120h (scripts/backtest_live_pipeline.py and
# the calibration sweep). The P10-P90 interval is close to nominal statewide but
# materially over-confident per zone, so the UI must not present them alike.
COVERAGE_STATEWIDE = 78.3
COVERAGE_ZONE_RANGE = (47.4, 74.7)


def to_et(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC-naive or UTC-aware -> Eastern. NYISO operates in Eastern; a peak
    reported at 22:00 UTC reads as nonsense to anyone looking at the grid."""
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return idx.tz_convert(ET)


def humanise_age(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    if seconds < 90:
        return "just now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes} min ago"
    hours, mins = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {mins}m ago"
    return f"{hours // 24}d ago"


@st.cache_data(show_spinner=False)
def zone_accuracy() -> pd.DataFrame:
    """Per-zone MAPE from the banked held-out predictions (854 issue dates).

    The 2.95% headline is a *statewide* number; zone errors partly cancel when
    summed. Showing one figure beside a zone selector would overstate zone-level
    reliability, so each zone carries its own.
    """
    if not PRED_ARCHIVE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(PRED_ARCHIVE, columns=["zone", "lead_day", "pred_mw", "actual_mw"])
    df["ape"] = (df["pred_mw"] - df["actual_mw"]).abs() / df["actual_mw"].clip(lower=1) * 100
    overall = df.groupby("zone")["ape"].mean().rename("mape_all")
    day1 = df[df["lead_day"] == 1].groupby("zone")["ape"].mean().rename("mape_day1")
    return pd.concat([overall, day1], axis=1).reset_index()


def live_scope_frame(live: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Collapse the long live frame to one series for Statewide or a single zone.

    Columns are taken from what the cache actually holds; hardcoding the list
    silently dropped nyiso_mw and made a working comparison look unavailable.
    """
    cols = [c for c in ("actual_mw", "pred_mw", "pred_lo", "pred_hi", "nyiso_mw")
            if c in live.columns]
    if scope != "Statewide":
        return live[live["zone"] == scope].set_index("ts_utc").sort_index()[cols]
    return live.groupby("ts_utc")[cols].sum(min_count=1).sort_index()


# ---------------------------------------------------------------------------
# UI chrome
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EnergyAI | NYISO Demand",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');

      /* Soften header; hide Deploy chrome */
      header[data-testid="stHeader"] {{
        background: transparent !important;
        color: {COLORS["text"]} !important;
        z-index: 999990 !important;
      }}
      [data-testid="stDecoration"],
      [data-testid="stStatusWidget"],
      #MainMenu,
      footer,
      .stDeployButton,
      div[data-testid="stAppDeployButton"] {{
        display: none !important;
        visibility: hidden !important;
      }}
      /* No sidebar, so suppress the collapse/expand chevron entirely. Streamlit
         still renders the control in some versions even with no sidebar body. */
      [data-testid="stExpandSidebarButton"],
      [data-testid="stSidebarCollapseButton"],
      [data-testid="collapsedControl"],
      [data-testid="stSidebar"],
      [data-testid="stSidebarNav"],
      button[kind="headerNoPadding"] {{
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        pointer-events: none !important;
      }}
      div[data-testid="stAppViewContainer"] > section > div {{
        padding-top: 1rem;
      }}

      .stApp {{
        background: radial-gradient(1200px 600px at 10% -10%, #1a2a33 0%, {COLORS["bg"]} 55%),
                    linear-gradient(180deg, #121820 0%, {COLORS["bg"]} 100%);
        color: {COLORS["text"]};
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      }}
      .block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1280px; }}
      h1, h2, h3, h4 {{ font-family: "IBM Plex Sans", sans-serif; letter-spacing: -0.02em; }}
      div[data-testid="stMetric"] {{
        background: {COLORS["panel"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 10px;
        padding: 0.65rem 0.85rem;
      }}
      div[data-testid="stMetric"] label {{
        color: {COLORS["muted"]} !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
      }}
      div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-family: "IBM Plex Mono", monospace;
        color: {COLORS["text"]};
        font-size: 1.05rem !important;
        line-height: 1.2 !important;
      }}
      div[data-testid="stMetric"] [data-testid="stMetricDelta"] {{
        font-size: 0.7rem !important;
      }}
      .ea-hero {{
        display: flex; align-items: flex-end; justify-content: space-between;
        gap: 1.5rem; margin-bottom: 0.75rem; padding-bottom: 1rem;
        border-bottom: 1px solid {COLORS["border"]};
      }}
      .ea-brand {{
        font-size: 1.85rem; font-weight: 700; color: {COLORS["text"]};
        letter-spacing: -0.03em; line-height: 1.1;
      }}
      .ea-brand span {{ color: {COLORS["accent"]}; }}
      .ea-sub {{ color: {COLORS["muted"]}; font-size: 0.95rem; margin-top: 0.35rem; max-width: 42rem; }}
      .ea-badge {{
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: rgba(26,155,142,0.15); color: {COLORS["accent"]};
        border: 1px solid rgba(26,155,142,0.35);
        font-size: 0.78rem; font-weight: 600; padding: 0.35rem 0.7rem;
        border-radius: 999px; white-space:nowrap;
      }}
      .ea-badge.mock {{
        background: rgba(232,168,56,0.12); color: {COLORS["accent2"]};
        border-color: rgba(232,168,56,0.35);
      }}
      .ea-dot {{
        width: 7px; height: 7px; border-radius: 50%;
        background: {COLORS["accent"]}; box-shadow: 0 0 0 3px rgba(26,155,142,0.25);
      }}
      .ea-panel {{
        background: {COLORS["panel"]}; border: 1px solid {COLORS["border"]};
        border-radius: 12px; padding: 1rem 1.1rem; margin-bottom: 0.75rem;
      }}
      .ea-caption {{ color: {COLORS["muted"]}; font-size: 0.82rem; }}
      div[data-baseweb="tab-list"] {{
        gap: 0.5rem;
        background: transparent !important;
        border-bottom: none !important;
      }}
      button[data-baseweb="tab"] {{
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 999px !important;
        color: {COLORS["muted"]} !important;
        font-weight: 600 !important;
        padding: 0.45rem 1.15rem !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
      }}
      button[data-baseweb="tab"][aria-selected="true"] {{
        color: {COLORS["text"]} !important;
        background: rgba(26, 155, 142, 0.22) !important;
        border-color: rgba(26, 155, 142, 0.55) !important;
      }}
      button[data-baseweb="tab"] > div {{
        border-bottom: none !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

METRICS = model_metrics()
day_mape = METRICS["day_mape"]

# No sidebar. It held a Deploy link pointing at the wrong repository, plus
# archive metadata now carried by each tab's own caption; dropping it also
# removes the collapse chevron, which has nothing left to toggle.
#
# Coverage describes the prediction archive rather than the raw demand archive:
# it is what the Forecasting tab can actually show, and it saves loading an
# 8.8 MB parquet at startup purely to render a date range.
COVERAGE_NOTE = (
    f"{ISSUE_DATES[0]} → {ISSUE_DATES[-1]}, {len(ISSUE_DATES)} issued forecasts, "
    f"11 NYISO load zones (A–K)"
)

# Badge reflects forecast freshness. It reads the on-disk cache only -- the old
# version fired two NYISO requests on every rerun purely to colour a pill.
_cache_age = lfc.cache_age_seconds()
_badge_fresh = _cache_age is not None and _cache_age < 3600
_badge_cls = "ea-badge" if _badge_fresh else "ea-badge mock"
_badge_text = (
    f"Live forecast · {humanise_age(_cache_age)}" if _cache_age is not None
    else "No live forecast yet"
)

st.markdown(
    f"""
    <div class="ea-hero">
      <div>
        <div class="ea-brand">Energy<span>AI</span></div>
        <div class="ea-sub">New York ISO demand forecasting across 11 load zones — a 120-hour outlook from live NYISO and Open-Meteo feeds.</div>
      </div>
      <div class="{_badge_cls}"><span class="ea-dot" style="background:{COLORS["accent"] if _badge_fresh else COLORS["accent2"]};box-shadow:none;"></span> {_badge_text}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_fc, tab_live = st.tabs(["Forecasting", "Live forecast"])
with tab_fc:
    st.markdown("#### Day-ahead forecasting")
    origin = forecast_origin()

    scope_options = ["Statewide"] + ZONE_COLS
    c_horizon, c_scope = st.columns([2, 1])
    with c_horizon:
        day = st.radio(
            "Forecast horizon",
            options=[1, 2, 3, 4, 5],
            format_func=lambda d: f"Day {d}",
            horizontal=True,
            index=0,
            help="Lead day within the forecast issued on the selected date. "
                 "Day 1 is the first 24 hours after issue, Day 5 the fifth.",
        )
    with c_scope:
        forecast_scope = st.selectbox(
            "Zone",
            options=scope_options,
            index=0,
            format_func=lambda z: (
                "Statewide"
                if z == "Statewide"
                else f"{ZONE_META[z]['code']} — {ZONE_META[z]['label']}"
            ),
            help="Statewide = NYISO total. Otherwise metrics and chart use the selected load zone.",
        )

    if "sel_issue" not in st.session_state:
        st.session_state["sel_issue"] = LATEST_ISSUE
    c_date, _pad = st.columns([1, 3])
    with c_date:
        st.date_input(
            "Forecast date",
            min_value=ISSUE_DATES[0],
            max_value=ISSUE_DATES[-1],
            key="sel_issue",
            help=f"{len(ISSUE_DATES)} forecast dates available, "
                 f"{ISSUE_DATES[0]} to {ISSUE_DATES[-1]}.",
        )
    if st.session_state["sel_issue"] not in ISSUE_SET:
        st.caption("No forecast was issued on that date — showing the nearest earlier one.")

    # Day N means day N alone -- the 24 hours of that lead day. It used to
    # accumulate days 1..N into one series, so every metric, the chart and the
    # map described a window rather than the day actually selected.
    horizon_hours = 24
    actual_col, pred_col, scope_label = forecast_scope_columns(forecast_scope)

    day_df = day_block(day)
    if day_df.empty:
        st.warning(f"No forecast available for Day {day} on that issue date.")
        st.stop()
    day_df = day_df.copy().sort_index()
    day_df["day"] = day
    day_df["actual"] = day_df[actual_col]
    day_df["predicted"] = day_df[pred_col]

    window_start = day_df.index.min()
    window_end = day_df.index.max()
    days_label = f"Day {day}"
    st.caption(
        f"**{days_label}** · **{scope_label}** · "
        f"target date **{window_start.strftime('%a %d %b %Y')}** "
        f"({window_start.strftime('%H:%M')} → {window_end.strftime('%H:%M')} UTC) · "
        f"forecast issued {origin.strftime('%Y-%m-%d')} · "
        f"predictions from the trained zonal TFT model · "
        f"archive {COVERAGE_NOTE}"
    )

    summary = day_summary(day_df, actual_col="actual", pred_col="predicted")
    forecast_peak = float(day_df["predicted"].max())
    forecast_min = float(day_df["predicted"].min())
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Actual", f"{summary['actual_mw']:,.0f} MW",
              help=f"Mean realised demand across the 24 hours of Day {day} "
                   f"({window_start:%d %b}), {scope_label}.")
    m2.metric("Predicted", f"{summary['predicted_mw']:,.0f} MW",
              help=f"Mean forecast demand across the same 24 hours.")
    m3.metric("Forecast Peak", f"{forecast_peak:,.0f} MW",
              help=f"Highest forecast hour on Day {day}.")
    m4.metric("Forecast Min", f"{forecast_min:,.0f} MW",
              help=f"Lowest forecast hour on Day {day}.")
    m5.metric("Error % (MAPE)", f"{summary['error_pct']:.2f}%",
              help=f"Mean absolute percentage error (MAPE) across Day {day}'s "
                   f"24 hours: the average of |predicted − actual| ÷ actual, "
                   f"hour by hour.")
    if forecast_scope == "Statewide":
        st.caption(
            f"This forecast date: **{summary['error_pct']:.2f}%**  ·  "
            f"model average for Day {day} across all {len(ISSUE_DATES)} issued "
            f"forecasts: **{mape_by_lead().get(day, float('nan')):.2f}%**  ·  "
            f"held-out eval: **{day_mape.get(day, float('nan')):.2f}%**"
        )


    hourly = pd.DataFrame(
        {
            "Time (UTC)": day_df.index.strftime("%Y-%m-%d %H:%M"),
            "Forecast (MW)": day_df["predicted"].round(1),
            "Actual (MW)": day_df["actual"].round(1),
            "Error (MW)": (day_df["predicted"] - day_df["actual"]).round(1),
            "Error (%)": (
                100.0
                * (day_df["predicted"] - day_df["actual"]).abs()
                / day_df["actual"].clip(lower=1)
            ).round(2),
        }
    )

    # --- Chart + hour-by-hour table side by side ---
    chart_col, table_col = st.columns([1.15, 1], gap="large")
    panel_height = 460

    with chart_col:
        st.markdown(
            f"##### {scope_label} — actual vs predicted, Day {day} "
            f"({window_start:%d %b})"
        )
        fig_fc = go.Figure()
        fig_fc.add_trace(
            go.Scatter(
                x=day_df.index,
                y=day_df["actual"],
                name="Actual",
                mode="lines",
                line=dict(color=COLORS["actual"], width=2.4),
            )
        )
        fig_fc.add_trace(
            go.Scatter(
                x=day_df.index,
                y=day_df["predicted"],
                name="Predicted",
                mode="lines",
                line=dict(color=COLORS["forecast"], width=2.4, dash="dash"),
            )
        )
        fig_fc.update_layout(
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
            height=panel_height,
            yaxis_title="Demand (MW)",
            xaxis_title="Time (UTC)",
            hovermode="x unified",
            margin=dict(l=48, r=16, t=24, b=40),
        )
        st.plotly_chart(fig_fc, width="stretch")

    with table_col:
        st.markdown(f"##### HOUR-BY-HOUR · Day {day}")
        st.caption(f"{scope_label} · {window_start:%a %d %b %Y} · forecast vs actual")
        st.dataframe(
            hourly,
            width="stretch",
            hide_index=True,
            height=panel_height,
            column_config={
                "Forecast (MW)": st.column_config.NumberColumn(format="%.1f"),
                "Actual (MW)": st.column_config.NumberColumn(format="%.1f"),
                "Error (MW)": st.column_config.NumberColumn(format="%+.1f"),
                "Error (%)": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    # --- Map: zone consumption on the selected day only ---
    zone_df = zone_day_consumption(day_df)
    map_df = zone_df if forecast_scope == "Statewide" else zone_df[zone_df["zone"] == forecast_scope]
    st.markdown(f"##### Zone demand — Day {day} ({window_start:%d %b})")
    st.caption(
        f"Map covers **Day {day}** only, {window_start:%a %d %b %Y}"
        + (
            " (all zones)."
            if forecast_scope == "Statewide"
            else f" · highlighting **{scope_label}**."
        )
        + " Bubble size = predicted energy (MWh), colour = forecast error (%)."
    )

    scatter_fn = getattr(px, "scatter_map", None) or px.scatter_mapbox
    fig_zmap = scatter_fn(
        map_df,
        lat="lat",
        lon="lon",
        size="predicted_mwh",
        color="error_pct",
        hover_name="label",
        hover_data={
            "code": True,
            "actual_mwh": ":.0f",
            "predicted_mwh": ":.0f",
            "error_pct": ":.2f",
            "share_pct": ":.1f",
            "lat": False,
            "lon": False,
        },
        color_continuous_scale=["#1a9b8e", "#e8a838", "#d64545"],
        size_max=52,
        zoom=5.6 if forecast_scope == "Statewide" else 6.4,
        height=480,
    )
    layout_extra = {
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        "coloraxis_colorbar": dict(title="Error %"),
        "margin": dict(l=0, r=0, t=8, b=0),
    }
    if scatter_fn is px.scatter_mapbox:
        layout_extra["mapbox_style"] = "carto-darkmatter"
    else:
        layout_extra["map_style"] = "carto-darkmatter"
    fig_zmap.update_layout(**layout_extra)
    text_trace = getattr(go, "Scattermap", None) or go.Scattermapbox
    fig_zmap.add_trace(
        text_trace(
            lat=map_df["lat"],
            lon=map_df["lon"],
            mode="text",
            text=map_df["code"],
            textfont=dict(size=12, color="white", family="IBM Plex Mono"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    st.plotly_chart(fig_zmap, width="stretch")


# ---------------------------------------------------------------------------
# Live forecast — next 120h from live NYISO + Open-Meteo, laid out like the
# Forecasting tab: pick a lead day and target date, then compare our model
# against NYISO's own published forecast.
# ---------------------------------------------------------------------------
with tab_live:
    live_df, live_meta = lfc.load_cached()

    head_l, head_r = st.columns([3, 1])
    with head_l:
        st.markdown("#### Live 5-day forecast")
        if live_meta:
            st.markdown(
                f'<p class="ea-caption">Last refreshed <b>{humanise_age(_cache_age)}</b>'
                f' · NYISO load through <b>'
                f'{to_et([live_meta["demand_last_hour"]])[0]:%d %b %H:%M} ET</b>'
                f' · weather run through <b>'
                f'{to_et([live_meta["weather_last_hour"]])[0]:%d %b %H:%M} ET</b></p>',
                unsafe_allow_html=True,
            )
    with head_r:
        # A fragment re-runs on its own timer without re-running the whole page,
        # so the cooldown ticks down live instead of only on interaction.
        @st.fragment(run_every=1.0)
        def _refresh_control() -> None:
            wait = lfc.seconds_until_refresh()
            cooling = wait > 0 and live_df is not None
            label = (f"Refresh in {int(wait // 60)}:{int(wait % 60):02d}" if cooling
                     else "Refresh forecast")
            if st.button(label, width="stretch", disabled=cooling,
                         type="primary", key="live_refresh"):
                try:
                    # get_forecast(force=True) re-checks the limit server-side;
                    # the disabled button is a courtesy, not the enforcement point.
                    with st.spinner("Pulling NYISO + Open-Meteo and running the model…"):
                        lfc.get_forecast(force=True)
                    st.cache_data.clear()
                    # scope="app" is required here: a bare st.rerun() inside a
                    # fragment reruns only the fragment, so the charts, tables
                    # and maps would keep showing the previous forecast.
                    st.rerun(scope="app")
                except lfc.RateLimited as exc:
                    st.warning(str(exc))
                except Exception as exc:  # network or upstream outage
                    st.error(f"Refresh failed: {type(exc).__name__}: {exc}")
            st.caption("Limited to one refresh per 10 min." if cooling
                       else "Ready to refresh.")

        _refresh_control()

    if live_df is None:
        # Hosted containers have ephemeral disk, so the cache is empty on every
        # cold start. Generating on first view means a visitor lands on a real
        # forecast rather than an empty panel and a button. Subsequent sessions
        # read the cache, and the 10-minute limit still governs manual refreshes.
        try:
            with st.spinner(
                "First run on this container — pulling NYISO and Open-Meteo "
                "feeds and running the model (about 30 seconds)…"
            ):
                lfc.run_forecast()
            st.rerun()
        except Exception as exc:
            st.error(
                f"Could not build the first forecast: {type(exc).__name__}: {exc}"
            )
            st.caption(
                "Both upstream feeds are public and unauthenticated; this is "
                "usually a transient outage. Try **Refresh forecast**."
            )
            st.stop()

    origin_utc = pd.Timestamp(live_meta["origin"])
    # Lead day N covers the Nth 24h block after the origin. The date picker and
    # the radio are two views of the same choice, so they stay in sync.
    day_dates = [
        (origin_utc + pd.Timedelta(days=d - 1)).date() for d in range(1, 6)
    ]

    # Lead day and target date are two views of one choice, kept in sync both
    # ways. Clamped on every run so a cache refresh (which shifts the origin)
    # cannot leave a stale date selected.
    if st.session_state.get("live_day") not in (1, 2, 3, 4, 5):
        st.session_state["live_day"] = 1
    if st.session_state.get("live_date") not in day_dates:
        st.session_state["live_date"] = day_dates[st.session_state["live_day"] - 1]

    def _sync_from_day() -> None:
        st.session_state["live_date"] = day_dates[st.session_state["live_day"] - 1]

    def _sync_from_date() -> None:
        picked = st.session_state["live_date"]
        if picked in day_dates:
            st.session_state["live_day"] = day_dates.index(picked) + 1
        else:  # outside the forecast window: snap back to the current day
            st.session_state["live_date"] = day_dates[st.session_state["live_day"] - 1]

    c_day, c_date, c_scope = st.columns([2, 1, 1])
    with c_day:
        st.radio(
            "Forecast horizon",
            options=[1, 2, 3, 4, 5],
            format_func=lambda d: f"Day {d}",
            horizontal=True,
            key="live_day",
            on_change=_sync_from_day,
            help="Lead day within the forecast issued now. Day 1 is the next "
                 "24 hours, Day 5 the fifth.",
        )
    with c_date:
        st.date_input(
            "Target date",
            min_value=day_dates[0],
            max_value=day_dates[-1],
            key="live_date",
            on_change=_sync_from_date,
            help="The five dates this forecast covers. Picking one selects the "
                 "matching lead day, and vice versa.",
        )
    live_day = st.session_state["live_day"]
    with c_scope:
        live_scope = st.selectbox(
            "Zone",
            options=["Statewide"] + ZONE_COLS,
            index=0,
            format_func=lambda z: ("Statewide" if z == "Statewide"
                                   else f"{ZONE_META[z]['code']} — {ZONE_META[z]['label']}"),
            key="live_scope",
            help="Statewide = NYISO total. Otherwise everything below uses the "
                 "selected load zone.",
        )

    scope_label_live = ("Statewide" if live_scope == "Statewide"
                        else f"{ZONE_META[live_scope]['code']} — {ZONE_META[live_scope]['label']}")

    series = live_scope_frame(live_df, live_scope)
    win_start = origin_utc + pd.Timedelta(days=live_day - 1)
    win_end = win_start + pd.Timedelta(hours=23)
    block = series.loc[win_start:win_end]
    if block.empty or block["pred_mw"].isna().all():
        st.warning(f"No forecast hours available for Day {live_day}.")
        st.stop()

    has_nyiso = "nyiso_mw" in block.columns and block["nyiso_mw"].notna().any()
    nyiso_issue = live_meta.get("nyiso_issue")

    st.caption(
        f"**Day {live_day}** · **{scope_label_live}** · "
        f"target date **{to_et([win_start])[0]:%a %d %b %Y}** · "
        f"forecast issued **{to_et([origin_utc])[0]:%d %b %H:%M} ET** · "
        f"zonal TFT, held-out MAPE "
        f"{(live_meta.get('model', {}).get('day_mape') or {}).get(live_day, float('nan')):.2f}% "
        f"at this horizon"
    )

    (sub_main,) = st.tabs(["Our prediction vs NYISO prediction"])

    with sub_main:
        x = to_et(block.index)
        ours = block["pred_mw"]
        panel_height = 430

        # --- Our forecast: cards ------------------------------------------
        o_peak_ts, o_min_ts = ours.idxmax(), ours.idxmin()
        c1, c2, c3, _c4, _c5 = st.columns(5)
        c1.metric("Predicted", f"{ours.mean():,.0f} MW",
                  help=f"Our mean forecast demand across Day {live_day}'s 24 hours, "
                       f"{scope_label_live}. These hours have not happened yet, so "
                       f"there is no actual to compare against.")
        c2.metric("Forecast Peak", f"{ours.max():,.0f} MW",
                  help=f"Our highest forecast hour on Day {live_day}, at "
                       f"{to_et([o_peak_ts])[0]:%H:%M} ET.")
        c3.metric("Forecast Min", f"{ours.min():,.0f} MW",
                  help=f"Our lowest forecast hour on Day {live_day}, at "
                       f"{to_et([o_min_ts])[0]:%H:%M} ET.")

        # --- NYISO forecast: cards, same shape, directly below -------------
        if has_nyiso:
            theirs = block["nyiso_mw"]
            n_peak_ts, n_min_ts = theirs.idxmax(), theirs.idxmin()
            n1, n2, n3, _n4, _n5 = st.columns(5)
            n1.metric("NYISO Predicted", f"{theirs.mean():,.0f} MW",
                      help=f"NYISO's own mean forecast for the same 24 hours "
                           f"(isolf, issue {nyiso_issue}).")
            n2.metric("NYISO Peak", f"{theirs.max():,.0f} MW",
                      help=f"NYISO's highest forecast hour on Day {live_day}, at "
                           f"{to_et([n_peak_ts])[0]:%H:%M} ET.")
            n3.metric("NYISO Min", f"{theirs.min():,.0f} MW",
                      help=f"NYISO's lowest forecast hour on Day {live_day}, at "
                           f"{to_et([n_min_ts])[0]:%H:%M} ET.")
        else:
            reason = live_meta.get("nyiso_error")
            if reason is None and "nyiso_mw" not in live_df.columns:
                reason = ("this cached run predates NYISO comparison support — "
                          "press Refresh forecast to pull it")
            st.info(
                "NYISO's published forecast (isolf) is not in this run, so only "
                "ours is shown. Our forecast is unaffected — it does not depend "
                "on it." + (f"\n\nReason: `{reason}`" if reason else "")
            )

        # --- Chart + hourly table -----------------------------------------
        compare = " vs NYISO forecast" if has_nyiso else ""
        st.markdown(
            f"##### {scope_label_live} hourly demand — our forecast{compare}, "
            f"Day {live_day} ({to_et([win_start])[0]:%a %d %b})"
        )

        chart_col, table_col = st.columns([1.15, 1], gap="large")

        with chart_col:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=block["pred_hi"], mode="lines", line=dict(width=0),
                hoverinfo="skip", showlegend=False, name="P90",
            ))
            fig.add_trace(go.Scatter(
                x=x, y=block["pred_lo"], mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(232,168,56,0.18)",
                hoverinfo="skip", name="Our P10–P90 range",
            ))
            fig.add_trace(go.Scatter(
                x=x, y=ours, mode="lines", name="Our forecast",
                line=dict(color=COLORS["forecast"], width=2.4, dash="dash"),
            ))
            if has_nyiso:
                fig.add_trace(go.Scatter(
                    x=x, y=block["nyiso_mw"], mode="lines", name="NYISO forecast",
                    line=dict(color=COLORS["live"], width=2.0, dash="dot"),
                ))
            fig.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
                height=panel_height, hovermode="x unified",
                yaxis_title="Demand (MW)", xaxis_title="Eastern time",
                margin=dict(l=48, r=16, t=24, b=40),
            )
            st.plotly_chart(fig, width="stretch")

        with table_col:
            st.markdown(
                f"##### HOUR-BY-HOUR · Day {live_day}"
                + (" · ours vs NYISO" if has_nyiso else "")
            )
            st.caption(f"{scope_label_live} · {to_et([win_start])[0]:%a %d %b %Y}")
            cols = {
                "Time (ET)": x.strftime("%H:%M"),
                "Ours (MW)": ours.round(0),
                "P10 (MW)": block["pred_lo"].round(0),
                "P90 (MW)": block["pred_hi"].round(0),
            }
            if has_nyiso:
                cols["NYISO (MW)"] = block["nyiso_mw"].round(0)
                cols["Diff (MW)"] = (ours - block["nyiso_mw"]).round(0)
            table = pd.DataFrame(cols)
            st.dataframe(
                table, width="stretch", hide_index=True, height=panel_height,
                column_config={
                    **{c: st.column_config.NumberColumn(format="%.0f")
                       for c in table.columns if c not in ("Time (ET)", "Diff (MW)")},
                    "Diff (MW)": st.column_config.NumberColumn(format="%+.0f"),
                },
            )
            st.download_button(
                "Download (CSV)",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name=f"forecast_day{live_day}_{live_scope}.csv",
                mime="text/csv", width="stretch",
                key=f"dl_{live_day}_{live_scope}",
            )

        cov = (f"{COVERAGE_STATEWIDE:.0f}%" if live_scope == "Statewide"
               else f"{COVERAGE_ZONE_RANGE[0]:.0f}–{COVERAGE_ZONE_RANGE[1]:.0f}%")
        caveat = ("" if live_scope == "Statewide" else
                  " At zone level the band is over-confident — treat it as indicative only.")
        note = (
            f"Shaded band is our model's own P10–P90 quantile output; measured "
            f"coverage on held-out data is **{cov}** against a nominal 80%.{caveat}"
        )
        if has_nyiso:
            gap = float(ours.mean() - block["nyiso_mw"].mean())
            note += (
                f" NYISO's line is their published zonal load forecast "
                f"(isolf, issue **{nyiso_issue}**); over Day {live_day} we sit "
                f"**{gap:+,.0f} MW** ({gap / float(block['nyiso_mw'].mean()) * 100:+.1f}%) "
                f"against it. Neither is truth — both are forecasts of the same hours."
            )
        st.caption(note)

        # --- Zone map: our forecast energy for this day --------------------
        day_rows = live_df[
            live_df["ts_utc"].between(win_start, win_end) & live_df["pred_mw"].notna()
        ]
        zf = day_rows.groupby("zone")["pred_mw"].sum().rename("pred_mwh").reset_index()
        if not zf.empty:
            zf["code"] = zf["zone"].map(lambda z: ZONE_META[z]["code"])
            zf["label"] = zf["zone"].map(lambda z: ZONE_META[z]["label"])
            zf["lat"] = zf["zone"].map(lambda z: ZONE_META[z]["lat"])
            zf["lon"] = zf["zone"].map(lambda z: ZONE_META[z]["lon"])
            zf["share_pct"] = 100 * zf["pred_mwh"] / zf["pred_mwh"].sum()
            acc = zone_accuracy()
            zf = (zf.merge(acc[["zone", "mape_all"]], on="zone", how="left")
                  if not acc.empty else zf.assign(mape_all=np.nan))
            map_df_live = (zf if live_scope == "Statewide"
                           else zf[zf["zone"] == live_scope])

            st.markdown(
                f"##### Our forecast energy by zone — Day {live_day} "
                f"({to_et([win_start])[0]:%a %d %b})"
            )
            st.caption(
                f"Our forecast over **Day {live_day}** only"
                + (" (all zones)." if live_scope == "Statewide"
                   else f" · highlighting **{scope_label_live}**.")
                + " Bubble size = forecast energy (MWh), colour = that zone's held-out MAPE."
            )
            scatter_fn = getattr(px, "scatter_map", None) or px.scatter_mapbox
            fig_map = scatter_fn(
                map_df_live, lat="lat", lon="lon", size="pred_mwh", color="mape_all",
                hover_name="label",
                hover_data={"code": True, "pred_mwh": ":,.0f", "share_pct": ":.1f",
                            "mape_all": ":.2f", "lat": False, "lon": False},
                color_continuous_scale=["#1a9b8e", "#e8a838", "#d64545"],
                size_max=48, zoom=5.4 if live_scope == "Statewide" else 6.4, height=450,
            )
            lay = {**{k: v for k, v in PLOTLY_LAYOUT.items()
                      if k not in ("xaxis", "yaxis")},
                   "coloraxis_colorbar": dict(title="Zone<br>MAPE %"),
                   "margin": dict(l=0, r=0, t=8, b=0)}
            lay["map_style" if scatter_fn is not px.scatter_mapbox else "mapbox_style"] = (
                "carto-darkmatter"
            )
            fig_map.update_layout(**lay)
            st.plotly_chart(fig_map, width="stretch",
                            key=f"map_{live_day}_{live_scope}")

    with st.expander("Run details"):
        t = live_meta.get("timings_s", {})
        m = live_meta.get("model", {})
        st.markdown(
            f"""
- **Forecast origin** {to_et([origin_utc])[0]:%Y-%m-%d %H:%M} ET  ({origin_utc:%Y-%m-%d %H:%M} UTC)
- **Encoder** {live_meta['encoder_hours']}h of realised load ·
  **Horizon** {live_meta['horizon_hours']}h · **Zones** {live_meta['zones']}
- **Checkpoint** `{m.get('checkpoint', 'n/a')}`
- **Held-out MAPE** {m.get('overall_mape', float('nan')):.2f}% overall ·
  {(m.get('day_mape') or {}).get(1, float('nan')):.2f}% at 24h
- **Timings** fetch {t.get('fetch', 0):.1f}s · features {t.get('features', 0):.2f}s ·
  inference {t.get('inference', 0):.2f}s · total {t.get('total', 0):.1f}s
- **Sources** NYISO `palIntegrated` (actuals) · NYISO `isolf` (their forecast) ·
  Open-Meteo forecast API (9 zonal coordinates)
"""
        )
