"""
Score the *live* pipeline against realised demand.

validate_live_forecast.py compares today's forecast to NYISO's, but neither is
ground truth. This runs the identical live code path -- build_live_frame then
live_forecast.predict -- at a historical origin where realised demand is known,
and scores it.

Weather comes from the observed archive, so this isolates the pipeline and the
model from weather-forecast error. The gap between this number and the
lead-matched 2.95% is precisely the cost of forecasting the weather, which the
held-out eval already quantified.

Usage:  python -m scripts.backtest_live_pipeline [ORIGIN]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import tft_core as tc  # noqa: E402
from app.live_features import build_live_frame  # noqa: E402
from app.live_forecast import predict  # noqa: E402

DEFAULT_ORIGIN = "2026-05-20 00:00"


def main() -> int:
    origin = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ORIGIN)

    dem = pd.read_parquet(tc.SANGAR / "nyiso_zonal_hourly.parquet")
    if dem.index.tz is not None:
        dem.index = dem.index.tz_convert("UTC").tz_localize(None)
    dem = dem[tc.ZONES]

    wx = pd.read_parquet(tc.SANGAR / "weather_observed_zonal.parquet")
    wx.index = pd.to_datetime(wx.index)
    wx = wx[[f"{v}__{k}" for k in tc.COORDS for v in tc.LEAN_VARS]]

    dec_end = origin + pd.Timedelta(hours=tc.DECODER_LEN - 1)
    if dem.index.max() < dec_end:
        print(f"archive ends {dem.index.max()}, need realised demand through {dec_end}")
        return 2

    ap = wx[[c for c in wx.columns if c.startswith("apparent_temperature__")]]
    ap = ap[ap.index < origin]

    frame, meta = build_live_frame(
        dem[dem.index < origin], wx, apparent_history=ap, origin=origin
    )
    preds = predict(frame, meta["origin"])

    actual = (dem.loc[origin:dec_end].stack().rename("actual_mw").reset_index())
    actual.columns = ["ts_utc", "zone", "actual_mw"]
    cmp = preds.merge(actual, on=["ts_utc", "zone"])
    if len(cmp) != tc.DECODER_LEN * len(tc.ZONES):
        print(f"joined {len(cmp)} rows, expected {tc.DECODER_LEN * len(tc.ZONES)}")
        return 2

    st = cmp.groupby("ts_utc")[["pred_mw", "actual_mw"]].sum()
    st["lead_day"] = ((st.index - origin) // pd.Timedelta(hours=24)).astype(int) + 1
    st["ape"] = (st["pred_mw"] - st["actual_mw"]).abs() / st["actual_mw"] * 100

    print(f"origin {origin}  ->  {dec_end}   (observed weather)")
    print("=" * 60)
    ref = meta_day_mape()
    for d in range(1, 6):
        g = st[st["lead_day"] == d]
        print(f"  day {d}  statewide MAPE {g['ape'].mean():5.2f} %"
              f"   (held-out eval: {ref.get(d, float('nan')):.2f} %)")
    overall = st["ape"].mean()
    print("-" * 60)
    print(f"  overall statewide MAPE {overall:5.2f} %")
    print(f"  mean signed bias       {(st['pred_mw'] - st['actual_mw']).mean():+,.0f} MW "
          f"({(st['pred_mw'] - st['actual_mw']).mean() / st['actual_mw'].mean() * 100:+.2f} %)")
    print(f"  peak  pred / actual    {st['pred_mw'].max():,.0f} / {st['actual_mw'].max():,.0f} MW")

    per_zone = (cmp.assign(ape=lambda d: (d["pred_mw"] - d["actual_mw"]).abs()
                           / d["actual_mw"].clip(lower=1) * 100)
                .groupby("zone")["ape"].mean().sort_values())
    print()
    print("per-zone MAPE:")
    for z, v in per_zone.items():
        print(f"  {z:10s} {v:5.2f} %")

    ok = bool(overall < 5.0)
    print()
    print("=" * 60)
    print("PASS -- live pipeline reproduces held-out accuracy" if ok
          else "FAIL -- live pipeline is materially worse than the eval")
    return 0 if ok else 1


def meta_day_mape() -> dict[int, float]:
    from app.live_forecast import model_metrics
    return model_metrics().get("day_mape", {})


if __name__ == "__main__":
    raise SystemExit(main())
