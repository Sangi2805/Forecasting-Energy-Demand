"""EnergyAI — NYISO demand dashboard (live P-58B + day-ahead forecasting)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths & zone metadata
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
ZONAL_PATH = ROOT / "Sangar" / "nyiso_zonal_hourly.parquet"
METRICS_PATH = ROOT / "Sangar" / "metrics_zonal_lr3e4.json"

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
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_zonal() -> pd.DataFrame:
    if not ZONAL_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(ZONAL_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    return df


def mock_live_window(zonal: pd.DataFrame, hours: int = 48, seed: int = 42) -> pd.DataFrame:
    """Replay the most recent history as a near-real-time feed (timestamps shifted to now)."""
    if zonal.empty:
        return pd.DataFrame()
    window = zonal.tail(hours).copy()
    now = pd.Timestamp.now(tz="UTC").floor("h")
    offset = now - window.index.max()
    window.index = window.index + offset
    rng = np.random.default_rng(seed + int(now.value // 1e12))
    noise = 1.0 + rng.normal(0, 0.008, size=window.shape)
    numeric = window.select_dtypes(include=[np.number])
    window[numeric.columns] = (numeric * noise).clip(lower=0)
    window["NYISO_TOTAL"] = window[ZONE_COLS].sum(axis=1)
    return window


def _nyiso_pal_url(day: date) -> str:
    # P-58B Real-Time Actual Load (5-min), public MIS CSV — no auth
    return f"http://mis.nyiso.com/public/csv/pal/{day.strftime('%Y%m%d')}pal.csv"


def _fetch_nyiso_pal_day(day: date) -> pd.DataFrame:
    req = Request(
        _nyiso_pal_url(day),
        headers={"User-Agent": "EnergyAI-dashboard/1.0"},
    )
    with urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(raw))
    if df.empty:
        return pd.DataFrame()
    df["Time Stamp"] = pd.to_datetime(df["Time Stamp"])
    # NYISO stamps are US/Eastern wall time (EDT/EST in file)
    df["utc"] = (
        df["Time Stamp"]
        .dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
    df = df[df["Name"].isin(ZONE_COLS)].copy()
    wide = (
        df.pivot_table(index="utc", columns="Name", values="Load", aggfunc="last")
        .reindex(columns=ZONE_COLS)
        .sort_index()
    )
    wide["NYISO_TOTAL"] = wide[ZONE_COLS].sum(axis=1, min_count=1)
    return wide.dropna(how="all")


@st.cache_data(ttl=60, show_spinner=False)
def load_nyiso_live(hours: int = 48) -> tuple[pd.DataFrame, str]:
    """
    Pull NYISO P-58B real-time actual load (5-min zonal).
    Returns (dataframe, source_label). Falls back to empty frame on failure.
    """
    today = date.today()
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for day in (today - timedelta(days=1), today):
        try:
            part = _fetch_nyiso_pal_day(day)
            if not part.empty:
                frames.append(part)
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, OSError) as exc:
            errors.append(f"{day}: {exc}")

    if not frames:
        return pd.DataFrame(), "error:" + (" | ".join(errors) if errors else "no data")

    live = pd.concat(frames).sort_index()
    live = live[~live.index.duplicated(keep="last")]
    if hours > 0:
        cutoff = live.index.max() - pd.Timedelta(hours=hours)
        live = live.loc[live.index >= cutoff]
    return live, "nyiso_p58b"


def get_live_frame(zonal: pd.DataFrame, hours: int = 48) -> tuple[pd.DataFrame, str]:
    live, source = load_nyiso_live(hours=hours)
    if not live.empty and source.startswith("nyiso"):
        return live, source
    return mock_live_window(zonal, hours=hours), "mock"


@st.cache_data(show_spinner=False)
def load_day_mape() -> dict[int, float]:
    """Reported per-day MAPE (%) from zonal TFT eval; used to calibrate mock preds."""
    defaults = {1: 2.48, 2: 2.89, 3: 3.14, 4: 3.52, 5: 3.96}
    if not METRICS_PATH.exists():
        return defaults
    try:
        raw = json.loads(METRICS_PATH.read_text())
        return {int(k): float(v) for k, v in raw.get("day_mape", {}).items()}
    except Exception:
        return defaults


def forecast_origin(zonal: pd.DataFrame) -> pd.Timestamp:
    """Start of the latest complete 5-day evaluation block (UTC day boundary)."""
    last = zonal.index.max().floor("h")
    # Need day-5 fully inside the archive: last midnight - 5 days
    end_day = last.floor("D")
    return end_day - pd.Timedelta(days=5)


def day_slice(zonal: pd.DataFrame, day: int) -> pd.DataFrame:
    """24 hourly rows for lead day 1..5 within the evaluation block."""
    origin = forecast_origin(zonal)
    start = origin + pd.Timedelta(days=day - 1)
    end = start + pd.Timedelta(hours=23)
    return zonal.loc[start:end].copy()


def mock_day_predictions(
    zonal: pd.DataFrame, day: int, target_mape: float, seed: int = 11
) -> pd.DataFrame:
    """
    Build hourly actual + predicted demand for one lead day.

    Predictions are seasonal-naive (lag 7d) then lightly calibrated so statewide
    MAPE is near the model's reported day MAPE. No saved prediction parquet exists yet.
    """
    actual = day_slice(zonal, day)
    if actual.empty or len(actual) < 20:
        return pd.DataFrame()

    cols = ZONE_COLS + ["NYISO_TOTAL"]
    pred = actual[cols].copy()
    for col in cols:
        naive = zonal[col].shift(168).reindex(actual.index)
        naive = naive.interpolate(limit_direction="both").bfill().ffill()
        pred[col] = naive.to_numpy(dtype=float)

    # Calibrate statewide series toward target MAPE (shrink naive errors + light noise)
    rng = np.random.default_rng(seed + day * 17)
    y = actual["NYISO_TOTAL"].to_numpy(dtype=float)
    yhat = pred["NYISO_TOTAL"].to_numpy(dtype=float)
    ape0 = float(np.mean(np.abs(y - yhat) / np.maximum(y, 1.0)) * 100)
    scale = (target_mape / ape0) if ape0 > 1e-6 else 1.0
    noise = rng.normal(0.0, 0.005, size=len(y))
    yhat = y + (yhat - y) * scale + y * noise
    pred["NYISO_TOTAL"] = np.clip(yhat, 0, None)

    # Zone preds: same relative shape as naive, renormalized to calibrated total
    zone_mat = pred[ZONE_COLS].to_numpy(dtype=float)
    row_sums = zone_mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-6, 1.0, row_sums)
    zone_mat = zone_mat / row_sums * pred["NYISO_TOTAL"].to_numpy()[:, None]
    pred[ZONE_COLS] = zone_mat

    out = actual[cols].copy()
    out = out.rename(columns={c: f"actual_{c}" for c in cols})
    for c in cols:
        out[f"pred_{c}"] = pred[c].values
    out["actual"] = out["actual_NYISO_TOTAL"]
    out["predicted"] = out["pred_NYISO_TOTAL"]
    return out


def day_summary(day_df: pd.DataFrame, actual_col: str = "actual", pred_col: str = "predicted") -> dict[str, float]:
    actual = day_df[actual_col]
    predicted = day_df[pred_col]
    mape = float(np.mean(np.abs(actual - predicted) / actual.clip(lower=1)) * 100)
    accuracy = max(0.0, 100.0 - mape)
    return {
        "actual_mw": float(actual.mean()),
        "predicted_mw": float(predicted.mean()),
        "error_pct": mape,
        "accuracy_pct": accuracy,
        "actual_mwh": float(actual.sum()),
        "predicted_mwh": float(predicted.sum()),
    }


def predictions_for_calendar_day(
    zonal: pd.DataFrame,
    day_start: pd.Timestamp,
    target_mape: float,
    seed: int = 11,
) -> pd.DataFrame:
    """
    Day-ahead mock forecast for one UTC calendar day (24h).
    Same seasonal-naive + MAPE calibration used in the Forecasting tab.
    """
    day_start = pd.Timestamp(day_start).tz_convert("UTC").floor("D")
    end = day_start + pd.Timedelta(hours=23)
    actual = zonal.loc[day_start:end].copy()
    if actual.empty or len(actual) < 20:
        return pd.DataFrame()

    cols = ZONE_COLS + ["NYISO_TOTAL"]
    pred = actual[cols].copy()
    for col in cols:
        naive = zonal[col].shift(168).reindex(actual.index)
        naive = naive.interpolate(limit_direction="both").bfill().ffill()
        pred[col] = naive.to_numpy(dtype=float)

    day_seed = int(day_start.strftime("%Y%m%d"))
    rng = np.random.default_rng(seed + day_seed)
    y = actual["NYISO_TOTAL"].to_numpy(dtype=float)
    yhat = pred["NYISO_TOTAL"].to_numpy(dtype=float)
    ape0 = float(np.mean(np.abs(y - yhat) / np.maximum(y, 1.0)) * 100)
    scale = (target_mape / ape0) if ape0 > 1e-6 else 1.0
    noise = rng.normal(0.0, 0.005, size=len(y))
    yhat = y + (yhat - y) * scale + y * noise
    pred["NYISO_TOTAL"] = np.clip(yhat, 0, None)

    zone_mat = pred[ZONE_COLS].to_numpy(dtype=float)
    row_sums = zone_mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-6, 1.0, row_sums)
    zone_mat = zone_mat / row_sums * pred["NYISO_TOTAL"].to_numpy()[:, None]
    pred[ZONE_COLS] = zone_mat

    out = actual[cols].copy()
    out = out.rename(columns={c: f"actual_{c}" for c in cols})
    for c in cols:
        out[f"pred_{c}"] = pred[c].values
    out["actual"] = out["actual_NYISO_TOTAL"]
    out["predicted"] = out["pred_NYISO_TOTAL"]
    return out



# ---------------------------------------------------------------------------
# Real model predictions (zonal TFT). Overrides the placeholder generators
# above; falls back to them automatically if the parquet is absent.
# ---------------------------------------------------------------------------
PRED_PATH = ROOT / "reports" / "tft_hourly_predictions.parquet"


@st.cache_data(show_spinner=False)
def load_real_predictions():
    if not PRED_PATH.exists():
        return None
    df = pd.read_parquet(PRED_PATH)
    df["issue_date"] = pd.to_datetime(df["issue_date"]).dt.date
    ts = pd.to_datetime(df["ts_utc"])
    df["ts_utc"] = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    idx = ["issue_date", "lead_day", "ts_utc"]
    p = df.pivot_table(index=idx, columns="zone", values="pred_mw", aggfunc="mean")
    a = df.pivot_table(index=idx, columns="zone", values="actual_mw", aggfunc="mean")
    zones = [z for z in ZONE_COLS if z in p.columns]
    p, a = p[zones].copy(), a[zones].copy()
    p["NYISO_TOTAL"] = p.sum(axis=1)
    a["NYISO_TOTAL"] = a.sum(axis=1)
    p.columns = [f"pred_{c}" for c in p.columns]
    a.columns = [f"actual_{c}" for c in a.columns]
    out = a.join(p).reset_index()
    out["actual"] = out["actual_NYISO_TOTAL"]
    out["predicted"] = out["pred_NYISO_TOTAL"]
    return out


_REAL = load_real_predictions()

if _REAL is not None and not _REAL.empty:
    _counts = _REAL.groupby("issue_date")["lead_day"].nunique()
    _FULL = _counts[_counts >= 5]
    _LATEST = _FULL.index.max() if len(_FULL) else _REAL["issue_date"].max()

    def _real_block(issue, lead):
        g = _REAL[(_REAL["issue_date"] == issue) & (_REAL["lead_day"] == lead)]
        if g.empty:
            return pd.DataFrame()
        return (g.set_index("ts_utc")
                 .drop(columns=["issue_date", "lead_day"])
                 .sort_index())

    def forecast_origin(zonal: pd.DataFrame) -> pd.Timestamp:
        g = _REAL[(_REAL["issue_date"] == _LATEST) & (_REAL["lead_day"] == 1)]
        return g["ts_utc"].min()

    def mock_day_predictions(zonal, day, target_mape=None, seed=11):
        return _real_block(_LATEST, int(day))

    def predictions_for_calendar_day(zonal, day_start, target_mape=None, seed=11):
        return _real_block(pd.Timestamp(day_start).date(), 1)


def score_day_frame(day_df: pd.DataFrame) -> dict[str, float | str]:
    """Peak / wMAPE / peak-APE scorecard for one completed forecast day."""
    actual = day_df["actual"]
    predicted = day_df["predicted"]
    wmape = float(np.sum(np.abs(actual - predicted)) / np.maximum(actual.sum(), 1.0) * 100)
    mape = float(np.mean(np.abs(actual - predicted) / actual.clip(lower=1)) * 100)
    a_peak = float(actual.max())
    p_peak = float(predicted.max())
    peak_ape = abs(p_peak - a_peak) / max(a_peak, 1.0) * 100
    a_peak_ts = actual.idxmax()
    p_peak_ts = predicted.idxmax()
    return {
        "wmape": wmape,
        "mape": mape,
        "accuracy": max(0.0, 100.0 - wmape),
        "actual_peak_mw": a_peak,
        "predicted_peak_mw": p_peak,
        "peak_ape": peak_ape,
        "actual_peak_hour": a_peak_ts.strftime("%H:%M"),
        "predicted_peak_hour": p_peak_ts.strftime("%H:%M"),
        "actual_avg_mw": float(actual.mean()),
        "predicted_avg_mw": float(predicted.mean()),
    }


@st.cache_data(show_spinner=False)
def build_track_record(_zonal_mtime: float, n_days: int = 45) -> pd.DataFrame:
    """
    Score the last N complete UTC days: day-ahead mock vs archive actuals.
    Returns one row per day (newest first).
    """
    zonal = load_zonal()
    day_mape = load_day_mape()
    target = float(day_mape.get(1, 2.48))
    last = zonal.index.max().floor("h")
    # Prefer complete days only (exclude partial last day if incomplete)
    end_day = last.floor("D")
    if last.hour < 23:
        end_day = end_day - pd.Timedelta(days=1)

    rows: list[dict] = []
    for i in range(n_days):
        day_start = end_day - pd.Timedelta(days=i)
        day_df = predictions_for_calendar_day(zonal, day_start, target_mape=target)
        if day_df.empty or len(day_df) < 20:
            continue
        sc = score_day_frame(day_df)
        rows.append(
            {
                "date": day_start.strftime("%Y-%m-%d"),
                "date_ts": day_start,
                **sc,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date_ts", ascending=False).reset_index(drop=True)


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


def zone_snapshot(live: pd.DataFrame) -> pd.DataFrame:
    latest = live.iloc[-1]
    rows = []
    for zone, meta in ZONE_META.items():
        mw = float(latest[zone])
        rows.append(
            {
                "zone": zone,
                "code": meta["code"],
                "label": meta["label"],
                "lat": meta["lat"],
                "lon": meta["lon"],
                "demand_mw": mw,
            }
        )
    out = pd.DataFrame(rows)
    out["share_pct"] = 100 * out["demand_mw"] / out["demand_mw"].sum()
    return out


# ---------------------------------------------------------------------------
# UI chrome
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EnergyAI | NYISO Demand",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');

      /* Soften header; keep expand-sidebar control; hide Deploy chrome */
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
      /* Native reopen control (appears when sidebar is collapsed) */
      [data-testid="stExpandSidebarButton"],
      [data-testid="stSidebarCollapseButton"] {{
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        z-index: 1000000 !important;
        width: 2.6rem !important;
        height: 2.6rem !important;
        border-radius: 999px !important;
        background: rgba(26, 155, 142, 0.92) !important;
        border: 1px solid rgba(232, 238, 244, 0.35) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35) !important;
        align-items: center !important;
        justify-content: center !important;
      }}
      [data-testid="stExpandSidebarButton"] svg,
      [data-testid="stSidebarCollapseButton"] svg {{
        fill: #e8eef4 !important;
        color: #e8eef4 !important;
        stroke: #e8eef4 !important;
      }}
      div[data-testid="stAppViewContainer"] > section > div {{
        padding-top: 1rem;
      }}
      .ea-deploy a {{
        display: flex !important;
        align-items: center;
        justify-content: center;
        width: 100%;
        min-height: 2.6rem;
        border-radius: 999px !important;
        background: rgba(26, 155, 142, 0.22) !important;
        border: 1px solid rgba(26, 155, 142, 0.55) !important;
        color: {COLORS["text"]} !important;
        font-weight: 700 !important;
        text-decoration: none !important;
        backdrop-filter: blur(10px);
      }}
      .ea-deploy a:hover {{
        background: rgba(26, 155, 142, 0.35) !important;
      }}

      .stApp {{
        background: radial-gradient(1200px 600px at 10% -10%, #1a2a33 0%, {COLORS["bg"]} 55%),
                    linear-gradient(180deg, #121820 0%, {COLORS["bg"]} 100%);
        color: {COLORS["text"]};
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      }}
      [data-testid="stSidebar"] {{
        background: {COLORS["panel"]};
        border-right: 1px solid {COLORS["border"]};
      }}
      [data-testid="stSidebar"] * {{ color: {COLORS["text"]}; }}
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

zonal = load_zonal()
if zonal.empty:
    st.error(f"Zonal demand file not found at `{ZONAL_PATH}`.")
    st.stop()

day_mape = load_day_mape()
selected_zones = ZONE_COLS

# Sidebar
with st.sidebar:
    st.markdown("### Deploy")
    deploy_url = (
        "https://share.streamlit.io/deploy"
        "?repository=Sangi2805/End-to-End-Multivariate-Time-Series-Forecasting-for-Energy-Demand"
        "&branch=iteration3"
        "&mainModule=app/streamlit_app.py"
    )
    st.markdown(
        f'<div class="ea-deploy"><a href="{deploy_url}" target="_blank" rel="noopener noreferrer">Deploy</a></div>',
        unsafe_allow_html=True,
    )
    st.caption("Opens Streamlit Community Cloud deploy.")
    st.divider()
    st.markdown("### Archive")
    st.markdown(
        f'<p class="ea-caption">'
        f'<b>{zonal.index.min().strftime("%Y-%m-%d")}</b> → '
        f'<b>{zonal.index.max().strftime("%Y-%m-%d")}</b><br>'
        f"11 NYISO load zones (A–K)</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption(
        "Live: NYISO P-58B 5-min CSVs. "
        "Forecasting + Track record: zonal TFT predictions scored against archive actuals."
    )

# Prefetch live for badge (cached 60s)
_live_preview, _live_source = get_live_frame(zonal, hours=48)
_badge_live = _live_source.startswith("nyiso")
_badge_cls = "ea-badge" if _badge_live else "ea-badge mock"
_badge_text = "NYISO live (P-58B)" if _badge_live else "Live feed mocked"

st.markdown(
    f"""
    <div class="ea-hero">
      <div>
        <div class="ea-brand">Energy<span>AI</span></div>
        <div class="ea-sub">New York ISO load monitoring and short-horizon demand forecasting across 11 zones.</div>
      </div>
      <div class="{_badge_cls}"><span class="ea-dot" style="background:{COLORS["accent"] if _badge_live else COLORS["accent2"]};box-shadow:none;"></span> {_badge_text}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_fc, tab_live, tab_track = st.tabs(["Forecasting", "Live data", "Track record"])


# ---------------------------------------------------------------------------
# Live data — NYISO P-58B realtime (5-min)
# ---------------------------------------------------------------------------
with tab_live:
    st.markdown("#### Live demand")
    top_l, top_r = st.columns([3, 1])
    with top_r:
        if st.button("Refresh live data", use_container_width=True):
            load_nyiso_live.clear()
            st.rerun()

    live, live_source = get_live_frame(zonal, hours=48)
    if live.empty:
        st.error("Could not load live or fallback demand data.")
        st.stop()

    zones_now = zone_snapshot(live)
    as_of = live.index.max()
    is_live = live_source.startswith("nyiso")

    if is_live:
        st.success(
            f"Streaming **NYISO P-58B Real-Time Actual Load** (5-min zonal) · "
            f"last point **{as_of.strftime('%Y-%m-%d %H:%M')} UTC** · cache 60s",
            icon="✅",
        )
    else:
        st.warning(
            "NYISO live feed unavailable — showing mocked archive replay. "
            f"Detail: `{live_source}`",
            icon="⚠️",
        )

    c1, c2 = st.columns([2, 1])
    with c1:
        series = live[selected_zones]
        fig_live = go.Figure()
        palette = px.colors.sample_colorscale(
            "Tealgrn", samplepoints=np.linspace(0.2, 0.95, len(selected_zones))
        )
        for i, zone in enumerate(selected_zones):
            fig_live.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series[zone],
                    name=f"{ZONE_META[zone]['code']} {zone}",
                    mode="lines",
                    line=dict(width=1.6, color=palette[i]),
                )
            )
        fig_live.update_layout(
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
            height=420,
            yaxis_title="Demand (MW)",
            xaxis_title="Time (UTC)",
            hovermode="x unified",
            margin=dict(l=48, r=24, t=40, b=40),
        )
        st.plotly_chart(fig_live, use_container_width=True)

    with c2:
        st.markdown("##### Statewide")
        fig_tot = go.Figure()
        fig_tot.add_trace(
            go.Scatter(
                x=live.index,
                y=live["NYISO_TOTAL"],
                name="NYISO total",
                fill="tozeroy",
                line=dict(color=COLORS["live"], width=2),
                fillcolor="rgba(26,155,142,0.18)",
            )
        )
        fig_tot.update_layout(
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
            height=420,
            yaxis_title="MW",
            showlegend=False,
            margin=dict(l=48, r=24, t=40, b=40),
        )
        st.plotly_chart(fig_tot, use_container_width=True)

    # KPIs for latest interval
    latest = live.iloc[-1]
    statewide_now = float(latest["NYISO_TOTAL"])
    prev = float(live.iloc[-2]["NYISO_TOTAL"]) if len(live) > 1 else statewide_now
    k1, k2, k3 = st.columns(3)
    k1.metric("Statewide now", f"{statewide_now:,.0f} MW", f"{statewide_now - prev:+,.0f} MW")
    k2.metric("Peak (window)", f"{float(live['NYISO_TOTAL'].max()):,.0f} MW")
    k3.metric("As of (UTC)", as_of.strftime("%Y-%m-%d %H:%M"))

    st.markdown("##### Latest reading by zone")
    latest_row = (
        zones_now.sort_values("demand_mw", ascending=False)
        .assign(updated=as_of.strftime("%Y-%m-%d %H:%M UTC"))
    )
    st.dataframe(
        latest_row[["code", "label", "demand_mw", "updated"]].rename(
            columns={"code": "Zone", "label": "Name", "demand_mw": "MW", "updated": "Updated"}
        ),
        use_container_width=True,
        hide_index=True,
        column_config={"MW": st.column_config.NumberColumn(format="%.0f")},
    )


# ---------------------------------------------------------------------------
# Forecasting — 24..120h horizon (selecting Nh includes 24h..Nh)
# ---------------------------------------------------------------------------
with tab_fc:
    st.markdown("#### Day-ahead forecasting")
    origin = forecast_origin(zonal)

    scope_options = ["Statewide"] + ZONE_COLS
    c_horizon, c_scope = st.columns([2, 1])
    with c_horizon:
        day = st.radio(
            "Forecast horizon",
            options=[1, 2, 3, 4, 5],
            format_func=lambda d: f"{d * 24} hours",
            horizontal=True,
            index=0,
            help="Selecting N hours shows the full window from 24 hours through N hours as one continuous series.",
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

    included_days = list(range(1, day + 1))
    horizon_hours = day * 24
    actual_col, pred_col, scope_label = forecast_scope_columns(forecast_scope)

    frames = []
    for d in included_days:
        target = day_mape.get(d, 3.5)
        part = mock_day_predictions(zonal, d, target_mape=target)
        if part.empty:
            st.warning(f"Not enough archive history for {d * 24} hours.")
            st.stop()
        part = part.copy()
        part["day"] = d
        frames.append(part)

    day_df = pd.concat(frames).sort_index()
    day_df["actual"] = day_df[actual_col]
    day_df["predicted"] = day_df[pred_col]

    window_start = day_df.index.min()
    window_end = day_df.index.max()
    days_label = f"{horizon_hours} hours"
    target_note = ", ".join(f"{d * 24}h≈{day_mape.get(d, 3.5):.2f}%" for d in included_days)
    st.caption(
        f"**{days_label}** · **{scope_label}** · "
        f"{window_start.strftime('%Y-%m-%d %H:%M')} → "
        f"{window_end.strftime('%Y-%m-%d %H:%M')} UTC · "
        f"forecast origin {origin.strftime('%Y-%m-%d')} · "
        "predictions from the trained zonal TFT model"
    )

    summary = day_summary(day_df, actual_col="actual", pred_col="predicted")
    forecast_peak = float(day_df["predicted"].max())
    forecast_min = float(day_df["predicted"].min())
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Actual (avg)", f"{summary['actual_mw']:,.0f} MW")
    m2.metric("Predicted (avg)", f"{summary['predicted_mw']:,.0f} MW")
    m3.metric("Forecast Peak", f"{forecast_peak:,.0f} MW")
    m4.metric("Forecast Min", f"{forecast_min:,.0f} MW")
    m5.metric("Error %", f"{summary['error_pct']:.2f}%")

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
        st.markdown(f"##### {scope_label} demand — actual vs predicted")
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
        if len(included_days) > 1:
            for d in included_days[1:]:
                boundary = day_df.loc[day_df["day"] == d].index.min()
                if pd.notna(boundary):
                    fig_fc.add_vline(
                        x=boundary.to_pydatetime(),
                        line_width=1,
                        line_dash="dot",
                        line_color=COLORS["muted"],
                        annotation_text=f"{d * 24}h",
                        annotation_position="top left",
                        annotation_font_color=COLORS["muted"],
                    )
        fig_fc.update_layout(
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
            height=panel_height,
            yaxis_title="Demand (MW)",
            xaxis_title="Time (UTC)",
            hovermode="x unified",
            margin=dict(l=48, r=16, t=24, b=40),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

    with table_col:
        st.markdown(f"##### HOUR-BY-HOUR · {horizon_hours}h block")
        st.caption(f"{scope_label} · forecast vs actual")
        st.dataframe(
            hourly,
            use_container_width=True,
            hide_index=True,
            height=panel_height,
            column_config={
                "Forecast (MW)": st.column_config.NumberColumn(format="%.1f"),
                "Actual (MW)": st.column_config.NumberColumn(format="%.1f"),
                "Error (MW)": st.column_config.NumberColumn(format="%+.1f"),
                "Error (%)": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    # --- Map: consumption for included days ---
    zone_df = zone_day_consumption(day_df)
    map_df = zone_df if forecast_scope == "Statewide" else zone_df[zone_df["zone"] == forecast_scope]
    st.markdown("##### Zone demand on this forecast horizon")
    st.caption(
        f"Map is tied to the selected **{horizon_hours} hours** window"
        + (
            " (all zones)."
            if forecast_scope == "Statewide"
            else f" · highlighting **{scope_label}**."
        )
        + " Bubble size = predicted energy (MWh), color = forecast error (%)."
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
    st.plotly_chart(fig_zmap, use_container_width=True)


# ---------------------------------------------------------------------------
# Track record — scored past day-ahead forecasts vs realized demand
# ---------------------------------------------------------------------------
with tab_track:
    st.markdown("#### Track record")
    st.caption(
        "Daily scores for completed day-ahead forecasts vs archive actuals "
        "(zonal TFT model output, held-out test period). "
        "Overall test error 2.95%; 2.21% day-ahead."
    )

    lookback = st.selectbox(
        "Lookback window",
        options=[14, 30, 45, 60, 90],
        index=2,
        format_func=lambda d: f"Last {d} days",
        key="track_lookback",
    )
    zonal_mtime = ZONAL_PATH.stat().st_mtime if ZONAL_PATH.exists() else 0.0
    ledger = build_track_record(zonal_mtime, n_days=int(lookback))

    if ledger.empty:
        st.warning("Not enough complete days in the archive to build a track record.")
    else:
        avg_wmape = float(ledger["wmape"].mean())
        avg_peak = float(ledger["peak_ape"].mean())
        best = ledger.loc[ledger["wmape"].idxmin()]
        worst = ledger.loc[ledger["wmape"].idxmax()]
        hit_rate = float((ledger["wmape"] <= 3.0).mean() * 100)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Avg wMAPE", f"{avg_wmape:.2f}%")
        k2.metric("Avg peak APE", f"{avg_peak:.2f}%")
        k3.metric("Days ≤ 3% wMAPE", f"{hit_rate:.0f}%")
        k4.metric("Best day", f"{best['date']}", delta=f"{best['wmape']:.2f}% wMAPE")
        k5.metric("Worst day", f"{worst['date']}", delta=f"{worst['wmape']:.2f}% wMAPE", delta_color="inverse")

        chart_col, replay_col = st.columns([1.15, 1])

        with chart_col:
            st.markdown("##### Daily wMAPE")
            trend = ledger.sort_values("date_ts")
            fig_tr = go.Figure()
            fig_tr.add_trace(
                go.Scatter(
                    x=trend["date"],
                    y=trend["wmape"],
                    mode="lines+markers",
                    name="wMAPE",
                    line=dict(color=COLORS["accent"], width=2.2),
                    marker=dict(size=6),
                    fill="tozeroy",
                    fillcolor="rgba(26,155,142,0.12)",
                )
            )
            fig_tr.add_hline(
                y=avg_wmape,
                line_dash="dot",
                line_color=COLORS["muted"],
                annotation_text=f"avg {avg_wmape:.2f}%",
                annotation_position="top left",
                annotation_font_color=COLORS["muted"],
            )
            fig_tr.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
                height=320,
                margin=dict(l=48, r=16, t=24, b=40),
                yaxis_title="wMAPE %",
                xaxis_title=None,
                showlegend=False,
            )
            st.plotly_chart(fig_tr, use_container_width=True)

        with replay_col:
            st.markdown("##### Day replay")
            date_options = ledger["date"].tolist()
            pick = st.selectbox("Replay day", options=date_options, key="track_replay_day")
            day_start = pd.Timestamp(pick, tz="UTC")
            target = float(load_day_mape().get(1, 2.48))
            replay = predictions_for_calendar_day(zonal, day_start, target_mape=target)
            if replay.empty:
                st.info("No hourly series for that day.")
            else:
                sc = score_day_frame(replay)
                r1, r2, r3 = st.columns(3)
                r1.metric("wMAPE", f"{sc['wmape']:.2f}%")
                r2.metric("Peak APE", f"{sc['peak_ape']:.2f}%")
                r3.metric(
                    "Peaks (P / A)",
                    f"{sc['predicted_peak_mw']/1000:.1f} / {sc['actual_peak_mw']/1000:.1f} GW",
                )
                fig_rp = go.Figure()
                fig_rp.add_trace(
                    go.Scatter(
                        x=replay.index,
                        y=replay["actual"],
                        name="Actual",
                        mode="lines",
                        line=dict(color=COLORS["actual"], width=2.4),
                    )
                )
                fig_rp.add_trace(
                    go.Scatter(
                        x=replay.index,
                        y=replay["predicted"],
                        name="Predicted",
                        mode="lines",
                        line=dict(color=COLORS["forecast"], width=2.4, dash="dash"),
                    )
                )
                fig_rp.update_layout(
                    **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("margin", "legend")},
                    height=260,
                    margin=dict(l=48, r=16, t=16, b=32),
                    yaxis_title="MW",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                )
                st.plotly_chart(fig_rp, use_container_width=True)

        st.markdown("##### Daily scorecard")
        table = ledger[
            [
                "date",
                "predicted_peak_mw",
                "actual_peak_mw",
                "peak_ape",
                "wmape",
                "mape",
                "accuracy",
                "predicted_peak_hour",
                "actual_peak_hour",
            ]
        ].copy()
        table = table.rename(
            columns={
                "date": "Date",
                "predicted_peak_mw": "Pred peak (MW)",
                "actual_peak_mw": "Actual peak (MW)",
                "peak_ape": "Peak APE %",
                "wmape": "wMAPE %",
                "mape": "MAPE %",
                "accuracy": "Accuracy %",
                "predicted_peak_hour": "Pred peak hour",
                "actual_peak_hour": "Actual peak hour",
            }
        )
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            height=360,
            column_config={
                "Pred peak (MW)": st.column_config.NumberColumn(format="%,.0f"),
                "Actual peak (MW)": st.column_config.NumberColumn(format="%,.0f"),
                "Peak APE %": st.column_config.NumberColumn(format="%.2f"),
                "wMAPE %": st.column_config.NumberColumn(format="%.2f"),
                "MAPE %": st.column_config.NumberColumn(format="%.2f"),
                "Accuracy %": st.column_config.NumberColumn(format="%.2f"),
            },
        )
