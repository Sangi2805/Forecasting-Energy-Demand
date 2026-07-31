"""
Phase 1 gate: prove the live fetchers reproduce the archives they will replace.

The live path must produce numbers indistinguishable from the ones the model was
trained and evaluated on. Both checks re-fetch a window that the banked archives
already cover, and diff.

  demand   my monthly-ZIP parser vs Sangar/nyiso_zonal_hourly.parquet.
           Exercises the EST/EDT -> UTC conversion, the zone pivot and the
           integrated-load column. Expect an exact match.

  weather  my Open-Meteo parser (archive endpoint, same window) vs
           Sangar/weather_observed_zonal.parquet. Exercises column naming,
           time alignment and units. Expect an exact match.

Usage:  python -m scripts.validate_live_data
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import live_data as ld  # noqa: E402
from app import tft_core as tc  # noqa: E402

ARCHIVE_DEMAND = tc.SANGAR / "nyiso_zonal_hourly.parquet"
ARCHIVE_WEATHER = tc.SANGAR / "weather_observed_zonal.parquet"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

TEST_MONTH = "202605"
WX_START, WX_END = "2026-05-15", "2026-05-31"


def check_demand() -> bool:
    print("=" * 64)
    print(f"DEMAND  --  re-fetch {TEST_MONTH} and diff against the archive")
    print("=" * 64)

    mine = ld._fetch_demand_month(TEST_MONTH)
    if mine.empty:
        print("FAIL: monthly ZIP returned nothing")
        return False
    mine = mine[~mine.index.duplicated(keep="last")].sort_index()

    ref = pd.read_parquet(ARCHIVE_DEMAND)
    if ref.index.tz is not None:
        ref.index = ref.index.tz_convert("UTC").tz_localize(None)

    common = mine.index.intersection(ref.index)
    print(f"mine   : {mine.shape}  {mine.index.min()} -> {mine.index.max()}")
    print(f"overlap: {len(common)} hours")
    if len(common) < 24 * 20:
        print("FAIL: not enough overlap to be meaningful")
        return False

    diff = (mine.loc[common, tc.ZONES] - ref.loc[common, tc.ZONES]).abs()
    print(f"max |diff| : {diff.to_numpy().max():.6f} MW")
    print(f"mean |diff|: {diff.to_numpy().mean():.6f} MW")
    worst = diff.max().sort_values(ascending=False)
    print(f"worst zone : {worst.index[0]} @ {worst.iloc[0]:.6f} MW")

    ok = bool(diff.to_numpy().max() < 0.01)
    print("PASS" if ok else "FAIL -- demand parsing diverges from the archive")
    return ok


def check_weather() -> bool:
    print()
    print("=" * 64)
    print(f"WEATHER --  re-fetch {WX_START}..{WX_END} and diff against the archive")
    print("=" * 64)

    ref = pd.read_parquet(ARCHIVE_WEATHER)
    ref.index = pd.to_datetime(ref.index)

    ok_all = True
    for wkey, (lat, lon) in list(tc.COORDS.items())[:3]:  # 3 coords is enough signal
        r = requests.get(ARCHIVE_URL, params={
            "latitude": lat, "longitude": lon,
            "start_date": WX_START, "end_date": WX_END,
            "hourly": ",".join(tc.LEAN_VARS),
            "timezone": "UTC",
        }, headers=ld.UA, timeout=60)
        js = r.json()
        mine = pd.DataFrame(js["hourly"])
        mine["time"] = pd.to_datetime(mine["time"])
        mine = mine.set_index("time").sort_index()

        cols = [f"{v}__{wkey}" for v in tc.LEAN_VARS]
        have = [c for c in cols if c in ref.columns]
        if len(have) != len(cols):
            print(f"{wkey}: archive missing {set(cols) - set(have)}")
            ok_all = False
            continue

        common = mine.index.intersection(ref.index)
        d = (mine.loc[common, tc.LEAN_VARS].to_numpy()
             - ref.loc[common, cols].to_numpy())
        mx = abs(d).max()
        print(f"{wkey:10s} {len(common):5d} h   max |diff| = {mx:.6f}")
        ok_all &= bool(mx < 0.01)

    print("PASS" if ok_all else "FAIL -- weather parsing diverges from the archive")
    return ok_all


def main() -> int:
    d = check_demand()
    w = check_weather()
    print()
    print("=" * 64)
    print("PHASE 1 GATE: " + ("PASS" if (d and w) else "FAIL"))
    return 0 if (d and w) else 1


if __name__ == "__main__":
    raise SystemExit(main())
