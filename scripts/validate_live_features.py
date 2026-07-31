"""
Phase 2 gate. Two questions, neither taken on trust.

TEST A -- does the live feature builder reproduce the training features?
    Replay a historical origin through build_live_frame using the banked
    archives as if they were live feeds, then diff every column against
    zonal_features_fx.parquet. Catches drift in the calendar, the lags, the
    rolling stats, temp_vshape, the fx recipe and the time_idx epoch.

TEST B -- are decoder values of the unknown reals actually inert?
    The horizon has no realised demand, so those columns are persistence-filled.
    That is only safe if the TFT ignores them. Run the same window twice --
    once with true decoder demand, once persistence-filled -- and compare
    predictions. Identical output means the fill cannot affect a forecast.
    Anything else means the fill strategy is load-bearing and must be solved.

Usage:  python -m scripts.validate_live_features
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import tft_core as tc  # noqa: E402
from app.live_features import build_live_frame  # noqa: E402

ORIGIN = pd.Timestamp(__import__("os").environ.get("VLF_ORIGIN", "2026-05-20 00:00"))

KNOWN_COLS = tc.LEAN_VARS + ["temp_vshape"] + tc.FX + ["time_idx"]


def load_archives() -> tuple[pd.DataFrame, pd.DataFrame]:
    dem = pd.read_parquet(tc.SANGAR / "nyiso_zonal_hourly.parquet")
    if dem.index.tz is not None:
        dem.index = dem.index.tz_convert("UTC").tz_localize(None)
    dem = dem[tc.ZONES]

    wx = pd.read_parquet(tc.SANGAR / "weather_observed_zonal.parquet")
    wx.index = pd.to_datetime(wx.index)
    keep = [f"{v}__{k}" for k in tc.COORDS for v in tc.LEAN_VARS]
    return dem, wx[keep]


def reference_window(enc_start, dec_end) -> pd.DataFrame:
    """The same window straight out of the training feature table."""
    ref = pd.read_parquet(tc.FEATURES)
    ref["utc"] = pd.to_datetime(ref["utc"])
    ref = ref[ref["utc"].between(enc_start, dec_end)].copy()
    ref["time_idx"] = (
        (ref["utc"] - tc.TIME_IDX_EPOCH).dt.total_seconds() // 3600
    ).astype(int)
    for c in tc.KNOWN_CATS:
        ref[c] = ref[c].astype(str).astype("category")
    ref["zone"] = ref["zone"].astype(str)
    return ref.sort_values(["zone", "time_idx"]).reset_index(drop=True)


def test_a(mine: pd.DataFrame, ref: pd.DataFrame, origin) -> bool:
    print("=" * 66)
    print("TEST A -- live feature builder vs training features")
    print("=" * 66)
    ok = True

    key = ["zone", "time_idx"]
    j = mine.merge(ref, on=key, suffixes=("_mine", "_ref"))
    print(f"rows joined : {len(j)} (expect {len(mine)})")
    if len(j) != len(mine):
        print("FAIL: join lost rows")
        return False

    print("\nknown reals (all 288h -- these drive the decoder):")
    for c in KNOWN_COLS:
        if c == "time_idx":
            continue
        d = (j[f"{c}_mine"] - j[f"{c}_ref"]).abs().max()
        flag = "ok" if d < 1e-6 else "MISMATCH"
        print(f"  {c:24s} max|diff| = {d:.8f}   {flag}")
        ok &= d < 1e-6

    print("\ncalendar categoricals:")
    for c in tc.KNOWN_CATS:
        same = (j[f"{c}_mine"].astype(str) == j[f"{c}_ref"].astype(str)).all()
        print(f"  {c:24s} {'ok' if same else 'MISMATCH'}")
        ok &= bool(same)

    enc = j[j["utc_mine"] < origin]
    print(f"\nunknown reals (encoder only, {len(enc)} rows):")
    for c in tc.UNKNOWN_REALS:
        d = (enc[f"{c}_mine"] - enc[f"{c}_ref"]).abs().max()
        flag = "ok" if d < 1e-6 else "MISMATCH"
        print(f"  {c:24s} max|diff| = {d:.8f}   {flag}")
        ok &= d < 1e-6

    print("\n" + ("TEST A PASS" if ok else "TEST A FAIL"))
    return ok


def test_b(mine: pd.DataFrame, ref: pd.DataFrame, origin) -> bool:
    print()
    print("=" * 66)
    print("TEST B -- are persistence-filled decoder unknowns inert?")
    print("=" * 66)

    params = tc.load_ds_params()
    model = tc.load_model()
    t = int((origin - tc.TIME_IDX_EPOCH).total_seconds() // 3600)

    filled = mine.copy()
    truth = ref.copy()

    # sanity: the two frames must differ ONLY in decoder unknown reals
    dec_m = filled[filled["time_idx"] >= t]
    dec_r = truth[truth["time_idx"] >= t]
    delta = float(
        (dec_m.set_index(["zone", "time_idx"])["demand"]
         - dec_r.set_index(["zone", "time_idx"])["demand"]).abs().max()
    )
    print(f"decoder demand differs by up to {delta:,.1f} MW between the two frames")
    if delta < 1.0:
        print("FAIL: frames are not actually different, test proves nothing")
        return False

    max_diff = 0.0
    for zone in tc.ZONES:
        preds = []
        for frame in (filled, truth):
            dz = frame[frame["zone"] == zone].sort_values("time_idx")
            ds = tc.dataset_from_params(
                params, dz, min_prediction_idx=t, stop_randomization=True
            )
            p, _, starts = tc.predict_zone(model, ds)
            assert len(starts) == 1, f"{zone}: {len(starts)} windows"
            preds.append(p[0])
        d = float(np.abs(preds[0] - preds[1]).max())
        max_diff = max(max_diff, d)
        print(f"  {zone:10s} max|pred diff| = {d:.8f} MW")

    print(f"\nworst across all zones and all 120 hours: {max_diff:.8f} MW")
    ok = max_diff < 1e-4
    print("TEST B PASS -- decoder unknowns are inert, fill is safe" if ok
          else "TEST B FAIL -- fill strategy affects the forecast")
    return ok


def main() -> int:
    dem, wx = load_archives()
    # Simulate live conditions: at the origin, demand beyond it does not exist
    # yet. Without this the replay hands the builder the very future it is
    # supposed to be forecasting, and nothing gets persistence-filled.
    dem = dem[dem.index < ORIGIN]
    # Deep apparent-temperature history to seed fx_hot_streak_day, standing in
    # for live_data.fetch_apparent_history. Cut at the origin so the replay
    # cannot see past it.
    ap = wx[[c for c in wx.columns if c.startswith("apparent_temperature__")]]
    ap = ap[ap.index < ORIGIN]
    mine, meta = build_live_frame(dem, wx, apparent_history=ap, origin=ORIGIN)
    print(f"origin {meta['origin']}   encoder {meta['encoder_start']} "
          f"-> decoder end {meta['decoder_end']}\n")

    ref = reference_window(meta["encoder_start"], meta["decoder_end"])

    a = test_a(mine, ref, ORIGIN)
    b = test_b(mine, ref, ORIGIN)

    print()
    print("=" * 66)
    print("PHASE 2 GATE: " + ("PASS" if (a and b) else "FAIL"))
    return 0 if (a and b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
