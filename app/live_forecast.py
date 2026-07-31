"""
Run the zonal TFT over live data and cache the result.

Orchestrates fetch -> feature build -> inference -> disk cache. The cache is
what the dashboard reads, so a cold Streamlit restart shows the last forecast
instantly instead of blocking on 19 HTTP requests.

Output (reports/live_forecast.parquet), 288 hours x 11 zones:
    ts_utc, zone, actual_mw, pred_mw
Encoder rows carry actual_mw with pred_mw null; horizon rows the reverse.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import live_data as ld
from . import tft_core as tc
from .live_features import build_live_frame

CACHE_DIR = tc.ROOT / "reports"
CACHE_PARQUET = CACHE_DIR / "live_forecast.parquet"
CACHE_META = CACHE_DIR / "live_forecast_meta.json"
METRICS_PATH = tc.SANGAR / "metrics_zonal_refit2023_fx.json"

MIN_REFRESH_SECONDS = 600  # 10 minutes

_MODEL = None
_PARAMS = None


class RateLimited(RuntimeError):
    def __init__(self, seconds_remaining: float):
        self.seconds_remaining = seconds_remaining
        super().__init__(
            f"refresh available in {int(seconds_remaining // 60)}m "
            f"{int(seconds_remaining % 60)}s"
        )


def _model_and_params():
    """Lazy singletons -- the checkpoint costs ~1s to load, so load it once."""
    global _MODEL, _PARAMS
    if _MODEL is None:
        # tc.resolve_artifact falls back to the Hub if either file is absent,
        # so no existence check here -- it would defeat that fallback.
        _PARAMS = tc.load_ds_params()
        _MODEL = tc.load_model()
    return _MODEL, _PARAMS


def model_metrics() -> dict:
    """Held-out test scores for the checkpoint actually being served."""
    try:
        raw = json.loads(METRICS_PATH.read_text())
    except (OSError, ValueError):
        return {}
    return {
        "checkpoint": raw.get("checkpoint", tc.CKPT.name),
        "overall_mape": raw.get("overall_mape"),
        "day_mape": {int(k): v for k, v in raw.get("day_mape", {}).items()},
        "seasonal_mape": raw.get("seasonal_mape", {}),
    }


def predict(frame: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Median-quantile forecast for every zone. Returns long (ts_utc, zone, pred_mw)."""
    model, params = _model_and_params()
    t = int((origin - tc.TIME_IDX_EPOCH).total_seconds() // 3600)
    hours = pd.date_range(origin, periods=tc.DECODER_LEN, freq="h")

    ds = tc.dataset_from_params(
        params, frame.sort_values(["zone", "time_idx"]),
        min_prediction_idx=t, stop_randomization=True,
    )
    pred, index = tc.predict_batched(model, ds)

    if len(index) != len(tc.ZONES):
        raise RuntimeError(f"expected {len(tc.ZONES)} windows, got {len(index)}")
    if not (index["time_idx"] == t).all():
        raise RuntimeError(f"windows do not all start at {t}: {index['time_idx'].tolist()}")

    zones = index["zone"].astype(str).tolist()
    if sorted(zones) != sorted(tc.ZONES):
        raise RuntimeError(f"unexpected zone set from dataset: {zones}")

    quantiles = tc.model_quantiles(model)
    i50 = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    i10 = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.1))
    i90 = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.9))

    out = []
    for row, zone in enumerate(zones):
        out.append(pd.DataFrame({
            "ts_utc": hours,
            "zone": zone,
            "pred_mw": np.asarray(pred[row, :, i50], dtype=float),
            "pred_lo": np.asarray(pred[row, :, i10], dtype=float),
            "pred_hi": np.asarray(pred[row, :, i90], dtype=float),
        }))
    return pd.concat(out, ignore_index=True).sort_values(
        ["zone", "ts_utc"]
    ).reset_index(drop=True)


def run_forecast() -> tuple[pd.DataFrame, dict]:
    """Fetch, build, predict, cache. Ignores the rate limit -- callers enforce it."""
    t0 = time.time()

    # Load the model up front so its one-off cost is not charged to inference.
    # Streamlit caches it across reruns, so a refresh only pays the forward pass.
    warm = _MODEL is not None
    _model_and_params()
    t_model = time.time() - t0

    mark = time.time()
    demand, weather, apparent = ld.fetch_all()
    t_fetch = time.time() - mark

    mark = time.time()
    frame, fmeta = build_live_frame(demand, weather, apparent_history=apparent)
    t_build = time.time() - mark

    mark = time.time()
    preds = predict(frame, fmeta["origin"])
    t_pred = time.time() - mark

    history = (
        demand.loc[fmeta["encoder_start"]:fmeta["origin"] - pd.Timedelta(hours=1)]
        .stack()
        .rename("actual_mw")
        .reset_index()
    )
    history.columns = ["ts_utc", "zone", "actual_mw"]

    combined = pd.concat([history, preds], ignore_index=True).sort_values(
        ["zone", "ts_utc"]
    ).reset_index(drop=True)

    # NYISO's own forecast, for side-by-side comparison. Never allowed to fail
    # the run: it is a nice-to-have, our forecast does not depend on it.
    nyiso_issue, nyiso_error = None, None
    try:
        nyiso, nyiso_issue = ld.fetch_nyiso_forecast(fmeta["origin"])
        if nyiso.empty:
            nyiso_error = "NYISO published no usable isolf rows for this origin"
        else:
            combined = combined.merge(nyiso, on=["ts_utc", "zone"], how="left")
    except Exception as exc:
        # Swallowed so an isolf outage cannot fail our forecast -- but the
        # reason is recorded, because a silently missing comparison is
        # indistinguishable from a broken one.
        nyiso_error = f"{type(exc).__name__}: {exc}"
    if "nyiso_mw" not in combined.columns:
        combined["nyiso_mw"] = np.nan

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origin": fmeta["origin"].isoformat(),
        "encoder_start": fmeta["encoder_start"].isoformat(),
        "decoder_end": fmeta["decoder_end"].isoformat(),
        "demand_last_hour": fmeta["demand_last_hour"].isoformat(),
        "weather_last_hour": fmeta["weather_last_hour"].isoformat(),
        "time_idx_origin": fmeta["time_idx_origin"],
        "horizon_hours": tc.DECODER_LEN,
        "encoder_hours": tc.ENCODER_LEN,
        "zones": len(tc.ZONES),
        "timings_s": {
            "model_load": round(t_model, 2),
            "model_was_warm": warm,
            "fetch": round(t_fetch, 2),
            "features": round(t_build, 2),
            "inference": round(t_pred, 2),
            "total": round(time.time() - t0, 2),
        },
        "model": model_metrics(),
        "quantiles": {"lo": 0.1, "hi": 0.9},
        "nyiso_issue": nyiso_issue.isoformat() if nyiso_issue else None,
        "nyiso_error": nyiso_error,
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(CACHE_PARQUET, index=False)
    CACHE_META.write_text(json.dumps(meta, indent=2))
    return combined, meta


def load_cached() -> tuple[pd.DataFrame | None, dict | None]:
    if not (CACHE_PARQUET.exists() and CACHE_META.exists()):
        return None, None
    try:
        df = pd.read_parquet(CACHE_PARQUET)
        df["ts_utc"] = pd.to_datetime(df["ts_utc"])
        return df, json.loads(CACHE_META.read_text())
    except (OSError, ValueError):
        return None, None


def load_meta() -> dict | None:
    """Just the metadata. The countdown polls this every second, so it must not
    drag the forecast parquet off disk each time."""
    if not CACHE_META.exists():
        return None
    try:
        return json.loads(CACHE_META.read_text())
    except (OSError, ValueError):
        return None


def cache_age_seconds() -> float | None:
    """Seconds since the cached forecast was generated, or None if no cache."""
    meta = load_meta()
    if not meta:
        return None
    try:
        gen = datetime.fromisoformat(meta["generated_at"])
    except (KeyError, ValueError):
        return None
    if gen.tzinfo is None:
        gen = gen.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - gen).total_seconds()


def seconds_until_refresh() -> float:
    """How long until a manual refresh is permitted. 0 means now."""
    age = cache_age_seconds()
    if age is None:
        return 0.0
    return max(0.0, MIN_REFRESH_SECONDS - age)


def get_forecast(force: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Cached forecast, refreshing only when allowed.

    force=True still respects the 10-minute limit and raises RateLimited; the
    limit exists to be kind to two free public APIs, not as a UI nicety.
    """
    cached, meta = load_cached()
    if cached is None:
        return run_forecast()
    if not force:
        return cached, meta
    wait = seconds_until_refresh()
    if wait > 0:
        raise RateLimited(wait)
    return run_forecast()


if __name__ == "__main__":
    df, meta = run_forecast()
    print(json.dumps(meta, indent=2))
    print()
    fc = df[df["pred_mw"].notna()]
    total = fc.groupby("ts_utc")["pred_mw"].sum()
    print(f"forecast rows : {len(fc)}  ({fc['zone'].nunique()} zones)")
    print(f"statewide peak: {total.max():,.0f} MW at {total.idxmax()}")
    print(f"statewide min : {total.min():,.0f} MW at {total.idxmin()}")
    print()
    print(total.head(6).round(0).to_string())
