"""
Phase 0 gate: prove the local inference path reproduces the banked predictions.

Picks one issue_date out of reports/tft_hourly_predictions.parquet, re-runs the
checkpoint over that window through app/tft_core, and compares MW against the
numbers notebook 09 produced on the pod. An exact match proves the checkpoint,
the GroupNormalizer state, and the feature recipe all survived the move off CAIR.

Usage:  python scripts/validate_inference.py [ISSUE_DATE]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import tft_core as tc  # noqa: E402

PRED_PATH = tc.ROOT / "reports" / "tft_hourly_predictions.parquet"
LEAD = 1


def main() -> int:
    t0 = time.time()

    banked = pd.read_parquet(PRED_PATH)
    banked["issue_date"] = pd.to_datetime(banked["issue_date"]).dt.date
    issues = sorted(banked["issue_date"].unique())

    if len(sys.argv) > 1:
        target = pd.Timestamp(sys.argv[1]).date()
        if target not in set(issues):
            print(f"issue_date {target} not banked; pick from {issues[0]}..{issues[-1]}")
            return 2
    else:
        target = issues[len(issues) // 2]  # mid-range, well clear of edges
    print(f"target issue_date : {target}   (lead {LEAD})")

    print("loading features ...", flush=True)
    df = tc.load_base()
    leads = pd.read_parquet(tc.LEADS)
    leads.index = pd.to_datetime(leads.index)
    if getattr(leads.index, "tz", None) is not None:
        leads.index = leads.index.tz_localize(None)

    tsi = int(df.loc[df["utc"] >= tc.TEST_START, "time_idx"].min())

    if tc.DS_PARAMS.exists():
        print("loading banked dataset params ...", flush=True)
        params = tc.load_ds_params()
    else:
        print("building training dataset (one-off, slow) ...", flush=True)
        params = tc.save_ds_params()
        print(f"  wrote {tc.DS_PARAMS.name}")

    print("loading checkpoint ...", flush=True)
    model = tc.load_model()

    # decoder must start at local midnight -- that is what defines an issue_date
    one = df[df["zone"] == df["zone"].iloc[0]][["time_idx", "utc"]]
    local = pd.DatetimeIndex(one["utc"]).tz_localize("UTC").tz_convert("America/New_York")
    hit = one[(local.date == target) & (local.hour == 0)]
    if hit.empty:
        print(f"no local-midnight row for {target}")
        return 2
    t = int(hit["time_idx"].iloc[0])
    utc0 = pd.Timestamp(hit["utc"].iloc[0])
    print(f"decoder start     : time_idx={t}  utc={utc0}")

    # swap on the full trimmed frame so rolling fx windows match the pod run
    df = df[df["time_idx"] >= tsi - tc.ENCODER_LEN - 1].copy()
    print("applying lead-matched weather swap ...", flush=True)
    dfx = tc.swap_lead(df, leads, LEAD)

    lo, hi = t - tc.ENCODER_LEN, t + tc.DECODER_LEN - 1
    rows = []
    for zi, zone in enumerate(tc.ZONES, 1):
        dz = dfx[(dfx["zone"] == zone) & dfx["time_idx"].between(lo, hi)]
        if len(dz) != tc.ENCODER_LEN + tc.DECODER_LEN:
            print(f"  {zone}: expected {tc.ENCODER_LEN + tc.DECODER_LEN} rows, got {len(dz)}")
            return 2
        ds = tc.dataset_from_params(
            params, dz, min_prediction_idx=t, stop_randomization=True
        )
        pred, actual, starts = tc.predict_zone(model, ds)
        assert len(starts) == 1 and int(starts[0]) == t, f"{zone}: got {len(starts)} windows"
        rows.append(
            pd.DataFrame({
                "ts_utc": pd.date_range(utc0, periods=24, freq="h"),
                "zone": zone,
                "mine_mw": pred[0][:24],
                "mine_actual": actual[0][:24],
            })
        )
        print(f"    {zone:10s} ({zi}/11)", flush=True)

    mine = pd.concat(rows, ignore_index=True)

    ref = banked[(banked["issue_date"] == target) & (banked["lead_day"] == LEAD)].copy()
    ref["ts_utc"] = pd.to_datetime(ref["ts_utc"])
    if getattr(ref["ts_utc"].dt, "tz", None) is not None:
        ref["ts_utc"] = ref["ts_utc"].dt.tz_localize(None)

    cmp = mine.merge(ref[["ts_utc", "zone", "pred_mw", "actual_mw"]], on=["ts_utc", "zone"])
    if len(cmp) != 24 * 11:
        print(f"join produced {len(cmp)} rows, expected {24 * 11}")
        return 2

    d_pred = (cmp["mine_mw"] - cmp["pred_mw"]).abs()
    d_act = (cmp["mine_actual"] - cmp["actual_mw"]).abs()
    rel = 100 * d_pred / cmp["pred_mw"].abs().clip(lower=1)

    print("\n" + "=" * 60)
    print(f"rows compared        : {len(cmp)}")
    print(f"actual  max |diff|   : {d_act.max():.4f} MW")
    print(f"pred    max |diff|   : {d_pred.max():.4f} MW")
    print(f"pred    mean |diff|  : {d_pred.mean():.4f} MW")
    print(f"pred    max rel diff : {rel.max():.4f} %")

    st = cmp.groupby("ts_utc")[["mine_mw", "pred_mw", "actual_mw"]].sum()
    mape_mine = float((st["mine_mw"] - st["actual_mw"]).abs().div(st["actual_mw"]).mean() * 100)
    mape_ref = float((st["pred_mw"] - st["actual_mw"]).abs().div(st["actual_mw"]).mean() * 100)
    print(f"statewide MAPE mine  : {mape_mine:.3f} %")
    print(f"statewide MAPE ref   : {mape_ref:.3f} %")

    ok = d_pred.max() < 1.0  # float32 round-trip + CPU/GPU kernel drift
    print("=" * 60)
    print(("PASS - inference path reproduces the pod" if ok
           else "FAIL - predictions diverge, do not build on this"))
    print(f"({time.time() - t0:.0f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
