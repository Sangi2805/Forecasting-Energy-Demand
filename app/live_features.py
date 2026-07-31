"""
Assemble the model-ready frame for a live 120-hour forecast.

Reproduces 03_zonal_features.ipynb (calendar, lags, rolling stats, temp_vshape)
and 06_eval_refit2023_fx.py's add_fx, but from freshly fetched data rather than
the banked parquet. The output is 288 hours x 11 zones: a 168-hour encoder of
realised demand plus a 120-hour decoder of forecast weather.

The decoder problem
-------------------
demand and its lag/rolling derivatives are time_varying_unknown_reals. Across the
horizon they are, by definition, unknown -- yet TimeSeriesDataSet still requires
finite values in those columns. The TFT feeds unknown reals to the encoder only,
so decoder values should be inert; scripts/validate_live_features.py proves that
empirically rather than trusting it. They are filled by persistence (last encoder
value carried forward) per zone.

demand_lag168 is the exception: at every decoder hour it reaches exactly back
into the encoder window, so it is genuinely known for all 120 hours.
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from .tft_core import (
    DECODER_LEN, ENCODER_LEN, HOT_DAY_C, KNOWN_CATS, LEAN_VARS, TEMP_VSHAPE_REF,
    TIME_IDX_EPOCH, UNKNOWN_REALS, ZONE_WEATHER, ZONES, add_fx, hot_streak,
)

LAG_WARMUP_H = 168  # demand_lag168 / demand_roll168_mean need a full week behind


class FeatureBuildError(RuntimeError):
    """Raised when the fetched data cannot support a full forecast window."""


def build_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features on the UTC backbone, derived from Eastern local time.

    Verbatim from 03_zonal_features.ipynb:build_calendar, except that the zone
    is named America/New_York rather than US/Eastern. The two are the same zone
    -- verified identical on hour, day-of-week, month, date and UTC offset over
    all 122,712 hours from 2013 to 2026 -- but US/Eastern is a backward-compat
    alias that trimmed tzdata builds may omit, and America/New_York is the
    canonical name that is always present.
    """
    local = index.tz_localize("UTC").tz_convert("America/New_York")
    cal = pd.DataFrame(index=index)
    cal["hour"] = local.hour
    cal["day_of_week"] = local.dayofweek
    cal["month"] = local.month
    cal["is_weekend"] = (local.dayofweek >= 5).astype(int)
    hol = USFederalHolidayCalendar().holidays(
        start=local.min().date(), end=local.max().date()
    )
    cal["is_holiday"] = pd.Series(local.date, index=index).isin(hol.date).astype(int)
    return cal


def _seeded_hot_streak(
    apparent_history: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    """
    fx_hot_streak_day computed over deep history rather than the forecast window.

    The streak is unbounded by construction, so counting it from the edge of a
    short window under-reports it -- measured at 23 days for an August origin,
    which moves the forecast because the streak is a *known* real feeding the
    decoder. Splicing archive history onto the forecast window restores it.

    Returns a (zone, date) -> streak table.
    """
    cols = [c for c in apparent_history.columns if c.startswith("apparent_temperature__")]
    hist = apparent_history[cols]
    fut = weather[[f"apparent_temperature__{k}" for k in {ZONE_WEATHER[z] for z in ZONES}]]

    joined = pd.concat([hist, fut[~fut.index.isin(hist.index)]]).sort_index()
    full = pd.date_range(joined.index.min(), joined.index.max(), freq="h")
    if len(full) != len(joined):
        joined = joined.reindex(full).interpolate(limit=6, limit_direction="both")

    rows = []
    for zone in ZONES:
        col = f"apparent_temperature__{ZONE_WEATHER[zone]}"
        s = joined[col]
        rows.append(pd.DataFrame({"zone": zone, "_d": s.index.date,
                                  "apparent_temperature": s.to_numpy()}))
    long = pd.concat(rows, ignore_index=True)

    daily = (long.groupby(["zone", "_d"])["apparent_temperature"].max()
             .rename("dmax").reset_index())
    daily["hot"] = daily["dmax"] >= HOT_DAY_C
    daily["fx_hot_streak_day"] = (
        daily.groupby("zone")["hot"].transform(hot_streak).astype(float)
    )
    return daily[["zone", "_d", "fx_hot_streak_day"]]


def _fill_decoder_unknowns(df: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Persistence-fill the unknown reals across the horizon, per zone."""
    df = df.sort_values(["zone", "utc"]).reset_index(drop=True)
    for col in UNKNOWN_REALS:
        df[col] = df.groupby("zone")[col].ffill()
    still_null = df[UNKNOWN_REALS].isna().any()
    if still_null.any():
        bad = list(still_null[still_null].index)
        raise FeatureBuildError(f"unknown reals still NaN after fill: {bad}")
    return df


def build_live_frame(
    demand: pd.DataFrame,
    weather: pd.DataFrame,
    apparent_history: pd.DataFrame | None = None,
    origin: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Build the 288-hour x 11-zone frame the model consumes.

    `demand`            UTC-hourly, one column per zone (live_data.fetch_demand)
    `weather`           UTC-hourly, f"{var}__{wkey}"    (live_data.fetch_weather)
    `apparent_history`  deep apparent-temperature history used to seed
                        fx_hot_streak_day (live_data.fetch_apparent_history).
                        Omitting it truncates the streak at the window edge and
                        biases the forecast -- only safe if `weather` itself
                        already reaches back far enough.
    `origin`            first forecast hour; defaults to the hour after the last
                        demand row

    Returns (frame, meta).
    """
    demand = demand.sort_index()
    weather = weather.sort_index()

    if origin is None:
        origin = demand.index.max() + pd.Timedelta(hours=1)
    origin = pd.Timestamp(origin)

    enc_start = origin - pd.Timedelta(hours=ENCODER_LEN)
    dec_end = origin + pd.Timedelta(hours=DECODER_LEN - 1)
    need_demand_from = enc_start - pd.Timedelta(hours=LAG_WARMUP_H)

    if demand.index.min() > need_demand_from:
        have = int((demand.index.max() - demand.index.min()) / pd.Timedelta(hours=1)) + 1
        raise FeatureBuildError(
            f"need demand from {need_demand_from} but earliest is {demand.index.min()} "
            f"({have}h available, {ENCODER_LEN + LAG_WARMUP_H}h required)"
        )
    if demand.index.max() < origin - pd.Timedelta(hours=1):
        raise FeatureBuildError(
            f"demand ends {demand.index.max()}, need through {origin - pd.Timedelta(hours=1)}"
        )
    if weather.index.max() < dec_end:
        raise FeatureBuildError(
            f"weather ends {weather.index.max()}, need through {dec_end}"
        )

    # Backbone runs from the start of the weather window so add_fx sees a deep
    # warmup (fx_hot_streak_day counts consecutive hot days and would otherwise
    # be truncated at the window edge).
    frame_start = max(weather.index.min(), need_demand_from - pd.Timedelta(hours=24 * 21))
    backbone = pd.date_range(frame_start, dec_end, freq="h")
    if not backbone.isin(weather.index).all():
        raise FeatureBuildError("weather has holes across the required backbone")

    wx = weather.reindex(backbone)
    dem = demand.reindex(backbone)
    calendar = build_calendar(backbone)

    frames = []
    for zone in ZONES:
        wkey = ZONE_WEATHER[zone]
        z = pd.DataFrame(index=backbone)
        z["zone"] = zone
        z["demand"] = dem[zone]
        for v in LEAN_VARS:
            z[v] = wx[f"{v}__{wkey}"]
        z["temp_vshape"] = (z["temperature_2m"] - TEMP_VSHAPE_REF).abs()
        for c in calendar.columns:
            z[c] = calendar[c]
        frames.append(z.reset_index().rename(columns={"index": "utc"}))

    long_df = pd.concat(frames, ignore_index=True)

    # fx first: it depends only on apparent_temperature, so it is unaffected by
    # the unknown decoder demand. The 72h and 24h rolling terms are satisfied by
    # the backbone warmup; the unbounded streak is then overwritten from history.
    long_df = add_fx(long_df)
    if apparent_history is not None:
        streaks = _seeded_hot_streak(apparent_history, weather)
        long_df["_d"] = long_df["utc"].dt.date
        long_df = long_df.drop(columns=["fx_hot_streak_day"]).merge(
            streaks, left_on=["zone", "_d"], right_on=["zone", "_d"], how="left"
        )
        if long_df["fx_hot_streak_day"].isna().any():
            raise FeatureBuildError("hot-streak history does not cover the window")
        long_df = long_df.drop(columns=["_d"])

    # Lags and rolling stats, within each zone. Decoder rows land as NaN here
    # because decoder demand is NaN; they are persistence-filled below.
    long_df = long_df.sort_values(["zone", "utc"]).reset_index(drop=True)
    g = long_df.groupby("zone")["demand"]
    long_df["demand_lag24"] = g.shift(24)
    long_df["demand_lag168"] = g.shift(168)
    long_df["demand_roll24_mean"] = g.transform(lambda s: s.rolling(24).mean())
    long_df["demand_roll168_mean"] = g.transform(lambda s: s.rolling(168).mean())
    long_df["demand_roll24_std"] = g.transform(lambda s: s.rolling(24).std())

    window = long_df[long_df["utc"].between(enc_start, dec_end)].copy()
    window = _fill_decoder_unknowns(window, origin)

    # Same absolute clock the checkpoint was fitted on.
    window["time_idx"] = (
        (window["utc"] - TIME_IDX_EPOCH).dt.total_seconds() // 3600
    ).astype(int)
    for c in KNOWN_CATS:
        window[c] = window[c].astype(str).astype("category")
    window["zone"] = window["zone"].astype(str)
    window = window.sort_values(["zone", "time_idx"]).reset_index(drop=True)

    expected = (ENCODER_LEN + DECODER_LEN) * len(ZONES)
    if len(window) != expected:
        raise FeatureBuildError(f"built {len(window)} rows, expected {expected}")
    if window.isna().any().any():
        bad = window.columns[window.isna().any()].tolist()
        raise FeatureBuildError(f"NaNs remain in: {bad}")

    meta = {
        "origin": origin,
        "encoder_start": enc_start,
        "decoder_end": dec_end,
        "demand_last_hour": demand.index.max(),
        "weather_last_hour": weather.index.max(),
        "time_idx_origin": int((origin - TIME_IDX_EPOCH).total_seconds() // 3600),
        "rows": len(window),
    }
    return window, meta


if __name__ == "__main__":
    from . import live_data as ld

    dem, wx, ap = ld.fetch_all()
    frame, meta = build_live_frame(dem, wx, apparent_history=ap)
    print("frame:", frame.shape)
    for k, v in meta.items():
        print(f"  {k:18s} {v}")
    print()
    print(frame.groupby("zone")["demand"].agg(["min", "mean", "max"]).round(0).to_string())
