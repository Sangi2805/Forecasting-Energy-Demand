"""
Live data acquisition for the 120-hour forecast.

Two sources, both public and unauthenticated:

  demand   NYISO MIS palIntegrated -- hourly integrated actual load per zone.
           Daily CSVs at {YYYYMMDD}palIntegrated.csv (note: no "_csv" suffix;
           the _csv.csv spelling used by the 5-min pal feed 404s here).

  weather  Open-Meteo forecast API -- past days and forward forecast in one
           response per coordinate. Units are left at their defaults (degC, %,
           km/h, W/m2) because that is what 02_download_zonal_weather.ipynb
           used; passing unit overrides here would silently shift the model
           off its training distribution.

Only palIntegrated is used for demand. The 5-min `pal` feed is a different
measurement convention (real-time snapshots vs settlement-integrated hourly);
splicing it onto the tail would introduce a discontinuity in the most recent
encoder hours, which are the ones the TFT weights most heavily.
"""

from __future__ import annotations

import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from .tft_core import COORDS, LEAN_VARS, ZONES

NYISO_DAILY = "http://mis.nyiso.com/public/csv/palIntegrated/{ymd}palIntegrated.csv"
NYISO_MONTHLY = "http://mis.nyiso.com/public/csv/palIntegrated/{ym}01palIntegrated_csv.zip"
ISOLF_MONTHLY = "http://mis.nyiso.com/public/csv/isolf/{ym}01isolf_csv.zip"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
UA = {"User-Agent": "EnergyAI-dashboard/1.0"}

# NYISO stamps are Eastern wall time with an explicit EST/EDT column. Using that
# column is DST-exact; inferring from the timestamp alone is ambiguous for the
# repeated 01:00 hour every November.
TZ_OFFSET_H = {"EST": 5, "EDT": 4}

DEMAND_WARMUP_DAYS = 16   # encoder 168h + 168h of lag/roll warmup, plus slack
WEATHER_PAST_DAYS = 35    # covers the 72h/24h fx rolling windows with slack
WEATHER_FCST_DAYS = 7     # 120h horizon + slack

APPARENT_HISTORY_DAYS = 400  # >= one winter, so fx_hot_streak_day starts at a true zero
ARCHIVE_LAG_DAYS = 6         # ERA5 archive trails real time; forecast past_days covers the gap

APPARENT_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "apparent_history.parquet"
APPARENT_CACHE_TTL_H = 12


class LiveDataError(RuntimeError):
    """Raised when a source cannot supply enough data to forecast."""


def _get(url: str, params: dict | None = None, retries: int = 4) -> requests.Response:
    """GET with backoff. 429 backs off harder -- Open-Meteo's free tier throttles
    bursts, and a refresh fires 19 requests at once."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=45)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return r  # caller decides; a missing day is not fatal
            last = RuntimeError(f"HTTP {r.status_code}")
            wait = (5.0 * (attempt + 1)) if r.status_code == 429 else (0.5 * 2 ** attempt)
        except requests.RequestException as exc:
            last = exc
            wait = 0.5 * 2 ** attempt
        if attempt < retries - 1:
            time.sleep(wait)
    raise LiveDataError(f"{url}: {last}")


# ---------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------
def _parse_demand_csv(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    if df.empty or "Integrated Load" not in df.columns:
        return pd.DataFrame()

    ts = pd.to_datetime(df["Time Stamp"])
    offset = df["Time Zone"].map(TZ_OFFSET_H)
    if offset.isna().any():
        bad = sorted(set(df.loc[offset.isna(), "Time Zone"]))
        raise LiveDataError(f"unknown NYISO Time Zone value(s): {bad}")
    df["utc"] = ts + pd.to_timedelta(offset, unit="h")

    df = df[df["Name"].isin(ZONES)]
    return df.pivot_table(index="utc", columns="Name", values="Integrated Load",
                          aggfunc="last")


def _fetch_demand_day(day: date) -> pd.DataFrame:
    """Single-day CSV. Only the most recent ~10 days are retained by NYISO."""
    r = _get(NYISO_DAILY.format(ymd=day.strftime("%Y%m%d")))
    if r.status_code == 404 or not r.text.strip():
        return pd.DataFrame()
    return _parse_demand_csv(r.text)


def _fetch_demand_month(ym: str) -> pd.DataFrame:
    """Whole-month archive ZIP. Carries full history, and is as fresh as the
    daily CSVs -- the current month's ZIP is rebuilt through the latest hour."""
    r = _get(NYISO_MONTHLY.format(ym=ym))
    if r.status_code == 404:
        return pd.DataFrame()
    frames = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith(".csv"):
                continue
            with zf.open(name) as fh:
                part = _parse_demand_csv(fh.read().decode("utf-8", errors="replace"))
            if not part.empty:
                frames.append(part)
    return pd.concat(frames) if frames else pd.DataFrame()


def fetch_demand(days_back: int = DEMAND_WARMUP_DAYS) -> pd.DataFrame:
    """Hourly integrated load per zone, UTC-indexed. Newest complete hour last."""
    today = date.today()
    start = today - timedelta(days=days_back)
    months = sorted({d.strftime("%Y%m") for d in (start, today)})

    with ThreadPoolExecutor(max_workers=4) as pool:
        month_frames = list(pool.map(_fetch_demand_month, months))
        # Belt and braces: today's daily CSV in case the ZIP rebuild lags.
        day_frames = list(pool.map(_fetch_demand_day, [today - timedelta(days=1), today]))

    frames = [f for f in month_frames + day_frames if not f.empty]
    if not frames:
        raise LiveDataError("NYISO returned no demand data for any requested day")

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.reindex(columns=ZONES)

    # An hour is only usable once every zone has reported it.
    complete = out.dropna(how="any")
    if complete.empty:
        raise LiveDataError("no hour has data for all 11 zones")

    # Trim to a contiguous tail: the model cannot span a hole.
    full = pd.date_range(complete.index.min(), complete.index.max(), freq="h")
    reindexed = out.reindex(full)
    gaps = reindexed[ZONES].isna().any(axis=1)
    if gaps.any():
        last_gap = gaps[gaps].index.max()
        reindexed = reindexed.loc[reindexed.index > last_gap]
    return reindexed[ZONES].astype(float)


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
def _fetch_weather_point(item: tuple[str, tuple[float, float]]) -> pd.DataFrame:
    wkey, (lat, lon) = item
    r = _get(OPEN_METEO, {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(LEAN_VARS),
        "past_days": WEATHER_PAST_DAYS,
        "forecast_days": WEATHER_FCST_DAYS,
        "timezone": "UTC",
    })
    if r.status_code != 200:
        raise LiveDataError(f"Open-Meteo HTTP {r.status_code} for {wkey}")
    hourly = r.json().get("hourly")
    if not hourly:
        raise LiveDataError(f"Open-Meteo returned no hourly block for {wkey}")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    df.columns = [f"{c}__{wkey}" for c in df.columns]
    return df


def fetch_weather() -> pd.DataFrame:
    """Hourly weather for all 9 coordinates, UTC-indexed, past + forecast."""
    with ThreadPoolExecutor(max_workers=9) as pool:
        frames = list(pool.map(_fetch_weather_point, COORDS.items()))

    out = pd.concat(frames, axis=1).sort_index()
    out = out[~out.index.duplicated(keep="first")]

    missing = [f"{v}__{k}" for k in COORDS for v in LEAN_VARS if f"{v}__{k}" not in out.columns]
    if missing:
        raise LiveDataError(f"Open-Meteo missing columns: {missing[:5]}")

    # Forward-fill short holes only; a long hole means the fetch is unusable.
    out = out.interpolate(limit=3, limit_direction="both")
    if out.isna().any().any():
        raise LiveDataError("Open-Meteo returned gaps too large to interpolate")
    return out


# ---------------------------------------------------------------------------
# Deep apparent-temperature history, for fx_hot_streak_day only
# ---------------------------------------------------------------------------
def _fetch_apparent_point(item: tuple[str, tuple[float, float]]) -> pd.DataFrame:
    wkey, (lat, lon) = item
    r = _get(OPEN_METEO_ARCHIVE, {
        "latitude": lat, "longitude": lon,
        "start_date": (date.today() - timedelta(days=APPARENT_HISTORY_DAYS)).isoformat(),
        "end_date": (date.today() - timedelta(days=ARCHIVE_LAG_DAYS)).isoformat(),
        "hourly": "apparent_temperature",
        "timezone": "UTC",
    })
    if r.status_code != 200:
        raise LiveDataError(f"Open-Meteo archive HTTP {r.status_code} for {wkey}")
    hourly = r.json().get("hourly")
    if not hourly:
        raise LiveDataError(f"Open-Meteo archive returned no hourly block for {wkey}")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    return df.rename(columns={"apparent_temperature": f"apparent_temperature__{wkey}"})


def fetch_apparent_history(use_cache: bool = True) -> pd.DataFrame:
    """
    Roughly a year of hourly apparent temperature per coordinate.

    fx_hot_streak_day counts consecutive days above 24C without bound. In a New
    York summer that run can exceed the forecast API's 92-day past_days cap, so
    the streak has to be seeded from genuine history or it silently truncates --
    a 23-day error was measured for an August origin. A year always spans a
    winter, which guarantees the count starts from a real zero.

    Only one variable is requested, so the payload stays small.

    Cached to disk: this window ends ARCHIVE_LAG_DAYS in the past, so it cannot
    change more than once a day. Skipping it halves the request count on a
    refresh, which is now entirely network-bound.
    """
    if use_cache and APPARENT_CACHE.exists():
        age_h = (time.time() - APPARENT_CACHE.stat().st_mtime) / 3600
        if age_h < APPARENT_CACHE_TTL_H:
            try:
                cached = pd.read_parquet(APPARENT_CACHE)
                cached.index = pd.to_datetime(cached.index)
                return cached
            except (OSError, ValueError):
                pass  # corrupt cache: fall through and refetch

    with ThreadPoolExecutor(max_workers=9) as pool:
        frames = list(pool.map(_fetch_apparent_point, COORDS.items()))

    out = pd.concat(frames, axis=1).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out = out.interpolate(limit=6, limit_direction="both")
    if out.isna().any().any():
        raise LiveDataError("Open-Meteo archive returned unusable gaps")

    try:
        APPARENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(APPARENT_CACHE)
    except OSError:
        pass  # cache is an optimisation, never a requirement
    return out


# ---------------------------------------------------------------------------
# NYISO's own load forecast (isolf), for side-by-side comparison
# ---------------------------------------------------------------------------
def _parse_isolf_csv(text: str, issue: date) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    cols = {" ".join(str(c).strip().upper().split()): c for c in df.columns}
    have = [z for z in ZONES if z in cols]
    if not have or "TIME STAMP" not in cols:
        return pd.DataFrame()

    ts = pd.to_datetime(df[cols["TIME STAMP"]])
    if "TIME ZONE" in cols:
        offset = df[cols["TIME ZONE"]].map(TZ_OFFSET_H)
    else:
        # isolf omits the tz column on some vintages; fall back to the offset
        # standing on the issue date rather than guessing per row.
        std = date(issue.year, 1, 1)
        offset = pd.Series(4 if 3 < issue.month < 12 or issue > std else 5, index=df.index)
    utc = ts + pd.to_timedelta(offset, unit="h")

    long = pd.DataFrame({"ts_utc": utc})
    for z in have:
        long[z] = pd.to_numeric(df[cols[z]], errors="coerce")
    out = long.melt(id_vars="ts_utc", var_name="zone", value_name="nyiso_mw")
    out["issue"] = issue
    return out.dropna(subset=["nyiso_mw"])


def fetch_nyiso_forecast(origin: pd.Timestamp | None = None) -> tuple[pd.DataFrame, date | None]:
    """
    NYISO's published zonal load forecast, latest issue at or before `origin`.

    Published as monthly ZIPs of per-issue-date CSVs (the daily URL 404s). Used
    purely as an independent comparison -- it is another forecast, not truth.
    """
    when = pd.Timestamp(origin) if origin is not None else pd.Timestamp.utcnow()
    months = sorted({when.strftime("%Y%m"), (when + pd.Timedelta(days=7)).strftime("%Y%m")})

    frames = []
    for ym in months:
        r = _get(ISOLF_MONTHLY.format(ym=ym))
        if r.status_code != 200:
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            for name in sorted(zf.namelist()):
                if not name.lower().endswith(".csv"):
                    continue
                stem = Path(name).name[:8]
                try:
                    issue = pd.to_datetime(stem, format="%Y%m%d").date()
                except ValueError:
                    continue
                part = _parse_isolf_csv(
                    zf.read(name).decode("utf-8", errors="replace"), issue
                )
                if not part.empty:
                    frames.append(part)

    if not frames:
        return pd.DataFrame(columns=["ts_utc", "zone", "nyiso_mw"]), None

    allf = pd.concat(frames, ignore_index=True)
    usable = sorted(d for d in allf["issue"].unique() if d <= when.date())
    if not usable:
        return pd.DataFrame(columns=["ts_utc", "zone", "nyiso_mw"]), None

    issue = usable[-1]
    out = (allf[allf["issue"] == issue]
           .drop(columns=["issue"])
           .drop_duplicates(["ts_utc", "zone"])
           .reset_index(drop=True))
    return out, issue


def fetch_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Core three sources. Returns (demand, weather, apparent_history)."""
    with ThreadPoolExecutor(max_workers=3) as pool:
        dem = pool.submit(fetch_demand)
        wx = pool.submit(fetch_weather)
        ap = pool.submit(fetch_apparent_history)
        return dem.result(), wx.result(), ap.result()


if __name__ == "__main__":
    import time

    t0 = time.time()
    dem = fetch_demand()
    t1 = time.time()
    wx = fetch_weather()
    t2 = time.time()

    print(f"demand  {dem.shape}  {dem.index.min()} -> {dem.index.max()}  ({t1 - t0:.1f}s)")
    print(f"        contiguous: {len(dem) == len(pd.date_range(dem.index.min(), dem.index.max(), freq='h'))}")
    print(f"        nulls: {int(dem.isna().sum().sum())}")
    print(dem.tail(3).to_string())
    print()
    print(f"weather {wx.shape}  {wx.index.min()} -> {wx.index.max()}  ({t2 - t1:.1f}s)")
    print(f"        nulls: {int(wx.isna().sum().sum())}")

    origin = dem.index.max() + pd.Timedelta(hours=1)
    print()
    print(f"forecast origin would be {origin} UTC")
    print(f"  demand history available : {(origin - dem.index.min()) / pd.Timedelta(hours=1):.0f} h")
    print(f"  weather ahead of origin  : {(wx.index.max() - origin) / pd.Timedelta(hours=1) + 1:.0f} h")
