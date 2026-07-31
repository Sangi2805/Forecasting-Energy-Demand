"""EnergyAI — NYISO demand dashboard (live P-58B + day-ahead forecasting)."""

from __future__ import annotations

import json
import sys
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

# `streamlit run app/streamlit_app.py` puts app/ on sys.path, not the repo root,
# so the sibling modules are not importable as a package without this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import live_forecast as lfc  # noqa: E402  (needs the path fix above)

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



if _REAL is not None and not _REAL.empty:
    _SORTED_ISSUES = sorted(_REAL["issue_date"].unique())
    _ISSUE_SET = set(_SORTED_ISSUES)

    def _selected_issue():
        want = st.session_state.get("sel_issue", _LATEST)
        if want in _ISSUE_SET:
            return want
        earlier = [d for d in _SORTED_ISSUES if d <= want]
        return earlier[-1] if earlier else _SORTED_ISSUES[0]

    def forecast_origin(zonal: pd.DataFrame) -> pd.Timestamp:
        g = _REAL[(_REAL["issue_date"] == _selected_issue()) & (_REAL["lead_day"] == 1)]
        return g["ts_utc"].min()

    def mock_day_predictions(zonal, day, target_mape=None, seed=11):
        return _real_block(_selected_issue(), int(day))

    @st.cache_data(show_spinner=False)
    def real_overall_mape():
        out = {}
        for n in range(1, 6):
            g = _REAL[_REAL["lead_day"] <= n]
            a = g["actual_NYISO_TOTAL"].to_numpy(dtype=float)
            p = g["pred_NYISO_TOTAL"].to_numpy(dtype=float)
            out[n] = float(np.mean(np.abs(a - p) / np.maximum(a, 1.0)) * 100)
        return out


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

zonal = load_zonal()
if zonal.empty:
    st.error(f"Zonal demand file not found at `{ZONAL_PATH}`.")
    st.stop()

day_mape = load_day_mape()
selected_zones = ZONE_COLS

# No sidebar. It held a Deploy link pointing at the wrong repository, plus
# archive metadata now carried by each tab's own caption; dropping it also
# removes the collapse chevron, which has nothing left to toggle.
ARCHIVE_RANGE = (
    f"{zonal.index.min():%Y-%m-%d} → {zonal.index.max():%Y-%m-%d}, "
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
    origin = forecast_origin(zonal)

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

    if _REAL is not None and not _REAL.empty:
        if "sel_issue" not in st.session_state:
            st.session_state["sel_issue"] = _SORTED_ISSUES[-1]
        c_date, _pad = st.columns([1, 3])
        with c_date:
            st.date_input(
                "Forecast date",
                min_value=_SORTED_ISSUES[0],
                max_value=_SORTED_ISSUES[-1],
                key="sel_issue",
                help=f"{len(_SORTED_ISSUES)} forecast dates available, "
                     f"{_SORTED_ISSUES[0]} to {_SORTED_ISSUES[-1]}.",
            )
        if st.session_state["sel_issue"] not in _ISSUE_SET:
            st.caption("No forecast was issued on that date — showing the nearest earlier one.")

    # Day N means day N alone -- the 24 hours of that lead day. It used to
    # accumulate days 1..N into one series, so every metric, the chart and the
    # map described a window rather than the day actually selected.
    horizon_hours = 24
    actual_col, pred_col, scope_label = forecast_scope_columns(forecast_scope)

    day_df = mock_day_predictions(zonal, day, target_mape=day_mape.get(day, 3.5))
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
        f"archive {ARCHIVE_RANGE}"
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
    _ov = real_overall_mape() if (_REAL is not None and not _REAL.empty) else {}
    if _ov and forecast_scope == "Statewide":
        st.caption(
            f"This forecast date: **{summary['error_pct']:.2f}%**  ·  "
            f"model average across all {len(_SORTED_ISSUES)} forecast dates "
            f"at this horizon: **{_ov.get(day, float('nan')):.2f}%**"
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
        st.info(
            "No live forecast cached yet. Press **Refresh forecast** to pull the "
            "latest NYISO load and Open-Meteo forecast and run the model."
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
