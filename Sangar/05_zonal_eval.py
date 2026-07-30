"""
Lead-matched evaluation of the zonal TFT, aggregated to statewide.
v3: RESUMABLE. Each lead's APE matrix is saved to zonal_lead{d}.npz on
completion; finished leads are skipped on relaunch. Survives pod recycles
by never risking more than one ~13-minute pass.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from ckpt_utils import load_tft_checkpoint, pick_device

ROOT      = Path(__file__).resolve().parent
FEATURES  = ROOT / "zonal_features.parquet"
LEADS     = ROOT / "weather_leads_zonal.parquet"
CKPT      = ROOT / "checkpoints_zonal/zonal_tft_best.ckpt"
BASE_JSON = ROOT / "metrics_lead_matched.json"
OUT_JSON  = ROOT / "metrics_zonal_lead_matched.json"

TRAIN_START = "2015-07-01 04:00"
VAL_START   = "2023-01-01 05:00"
TEST_START  = "2024-01-23 12:00"

ENCODER_LEN, DECODER_LEN = 168, 120
BATCH = 128
SEED = 42

ZONE_WEATHER = {
    "WEST": "A_WEST", "GENESE": "B_GENESE", "CENTRL": "C_CENTRL",
    "NORTH": "D_NORTH", "MHK VL": "E_MHKVL", "CAPITL": "F_CAPITL",
    "HUD VL": "G_HUDVL", "MILLWD": "G_HUDVL", "DUNWOD": "J_NYC",
    "N.Y.C.": "J_NYC", "LONGIL": "K_LONGIL",
}
LEAN_VARS = ["temperature_2m", "apparent_temperature", "relative_humidity_2m",
             "wind_speed_10m", "shortwave_radiation", "cloud_cover"]
UNKNOWN_REALS = ["demand", "demand_lag24", "demand_lag168",
                 "demand_roll24_mean", "demand_roll168_mean",
                 "demand_roll24_std"]
KNOWN_REALS = LEAN_VARS + ["temp_vshape", "time_idx"]
KNOWN_CATS = ["hour", "day_of_week", "month", "is_weekend", "is_holiday"]


def lead_file(d):
    return ROOT / f"zonal_lead{d}.npz"


def load_base():
    df = pd.read_parquet(FEATURES)
    df["utc"] = pd.to_datetime(df["utc"])
    for c in KNOWN_CATS:
        df[c] = df[c].astype(str).astype("category")
    df["zone"] = df["zone"].astype(str)
    df = df[df["utc"] >= TRAIN_START].copy()
    df["time_idx"] = df["time_idx"] - df["time_idx"].min()
    return df.sort_values(["zone", "time_idx"]).reset_index(drop=True)


def swap_lead(base, leads, d):
    out = base.copy()
    boundary = pd.Timestamp(TEST_START)
    leak_fallbacks = 0
    for zone, wkey in ZONE_WEATHER.items():
        mask = out["zone"] == zone
        utc = out.loc[mask, "utc"]
        post = utc >= boundary
        for v in LEAN_VARS:
            s = utc.map(leads[f"{v}_prev_day{d}__{wkey}"]).interpolate(limit=6)
            s5 = utc.map(leads[f"{v}_prev_day5__{wkey}"]).interpolate(limit=6)
            s = s.fillna(s5)
            leak_fallbacks += int((s.isna() & post).sum())
            s = s.fillna(out.loc[mask, v])
            out.loc[mask, v] = s.values
    out["temp_vshape"] = (out["temperature_2m"] - 14.0).abs()
    print(f"  post-boundary observed-fallback cells: {leak_fallbacks}"
          f"{'  <-- INVESTIGATE' if leak_fallbacks > 0 else ''}", flush=True)
    return out


def build_training_ds(df):
    val_idx = int(df.loc[df["utc"] >= VAL_START, "time_idx"].min())
    train_df = df[df["time_idx"] < val_idx]
    return TimeSeriesDataSet(
        train_df, time_idx="time_idx", target="demand", group_ids=["zone"],
        max_encoder_length=ENCODER_LEN, max_prediction_length=DECODER_LEN,
        time_varying_unknown_reals=UNKNOWN_REALS,
        time_varying_known_reals=KNOWN_REALS,
        time_varying_known_categoricals=KNOWN_CATS,
        static_categoricals=["zone"],
        target_normalizer=GroupNormalizer(groups=["zone"]),
        add_relative_time_idx=True, add_target_scales=True,
        allow_missing_timesteps=False,
    )


def predict_pass(model, training_ds, df, test_start_idx):
    test = TimeSeriesDataSet.from_dataset(
        training_ds, df, min_prediction_idx=test_start_idx,
        stop_randomization=True)
    dl = test.to_dataloader(train=False, batch_size=BATCH, num_workers=0)
    preds = model.predict(dl, mode="prediction", return_y=True,
                          return_index=True,
                          trainer_kwargs={"accelerator": "auto",
                                          "enable_progress_bar": False})
    return (preds.output.cpu().numpy(), preds.y[0].cpu().numpy(),
            preds.index["time_idx"].values)


def aggregate_statewide(y_hat, y_true, tidx):
    uniq, inv, counts = np.unique(tidx, return_inverse=True, return_counts=True)
    sum_hat = np.zeros((len(uniq), y_hat.shape[1]))
    sum_true = np.zeros_like(sum_hat)
    np.add.at(sum_hat, inv, y_hat)
    np.add.at(sum_true, inv, y_true)
    complete = counts == 11
    if (~complete).any():
        print(f"  dropped {(~complete).sum()} incomplete windows", flush=True)
    return sum_hat[complete], sum_true[complete], uniq[complete]


def summarize(baseline):
    """Build final summary from the five saved npz files."""
    day_mape, apes, mons = {}, [], []
    for d in range(1, 6):
        z = np.load(lead_file(d))
        day_mape[d] = float(z["ape"].mean() * 100)
        apes.append(z["ape"].ravel())
        mons.append(z["months"].ravel())
    all_ape, all_mon = np.concatenate(apes), np.concatenate(mons)
    overall = float(all_ape.mean() * 100)

    season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring",
                  4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer",
                  8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}
    seasons = {}
    for sn in ["Winter", "Spring", "Summer", "Fall"]:
        mask = np.isin(all_mon, [m for m, x in season_map.items() if x == sn])
        seasons[sn] = float(all_ape[mask].mean() * 100) if mask.any() else None

    print("\n============ ZONAL LEAD-MATCHED RESULTS ============", flush=True)
    for d in range(1, 6):
        base_str = f"   (statewide baseline {baseline[d]:.2f}%)" if baseline else ""
        print(f"Day {d}: {day_mape[d]:.2f}%{base_str}")
    print(f"Overall: {overall:.2f}%   (statewide baseline 3.96%)")
    print("\nSeasonal breakdown:")
    for sn, v in seasons.items():
        print(f"  {sn}: {v:.2f}%")

    out = {"day_mape": {str(k): round(v, 3) for k, v in day_mape.items()},
           "overall_mape": round(overall, 3),
           "seasonal_mape": {k: (round(v, 3) if v else None)
                             for k, v in seasons.items()},
           "checkpoint": str(CKPT),
           "note": "epoch-0 checkpoint, LR=1e-3 run; retrain at 3e-4 pending"}
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {OUT_JSON}", flush=True)


def main():
    pl.seed_everything(SEED)

    baseline = None
    if BASE_JSON.exists():
        b = json.loads(BASE_JSON.read_text())
        baseline = {int(k): v for k, v in b["day_mape"].items()}

    todo = [d for d in range(1, 6) if not lead_file(d).exists()]
    done = [d for d in range(1, 6) if lead_file(d).exists()]
    print(f"leads done: {done or 'none'}, todo: {todo or 'none'}", flush=True)

    if todo:
        df = load_base()
        leads = pd.read_parquet(LEADS)
        leads.index = pd.to_datetime(leads.index)
        if getattr(leads.index, "tz", None) is not None:
            leads.index = leads.index.tz_localize(None)

        test_start_idx = int(df.loc[df["utc"] >= TEST_START, "time_idx"].min())
        training_ds = build_training_ds(df)
        device = pick_device()
        model = load_tft_checkpoint(str(CKPT), device=device)

        one = df[df["zone"] == df["zone"].iloc[0]]
        month_arr = np.full(int(df["time_idx"].max()) + DECODER_LEN + 2, -1)
        month_arr[one["time_idx"].values] = one["utc"].dt.month.values

        df = df[df["time_idx"] >= test_start_idx - ENCODER_LEN - 1].copy()
        print(f"trimmed to test region: {len(df):,} rows", flush=True)

        for d in todo:
            print(f"\n=== Lead {d}: swap -> predict -> aggregate ===", flush=True)
            df_d = swap_lead(df, leads, d)
            y_hat, y_true, tidx = predict_pass(model, training_ds, df_d,
                                               test_start_idx)
            del df_d
            sh, st, ut = aggregate_statewide(y_hat, y_true, tidx)
            s, e = (d - 1) * 24, d * 24
            ape = np.abs(st[:, s:e] - sh[:, s:e]) / np.clip(st[:, s:e],
                                                            1e-6, None)
            months = month_arr[ut[:, None] + np.arange(s, e)[None, :]]
            np.savez(lead_file(d), ape=ape, months=months)
            base_str = f" (baseline {baseline[d]:.2f}%)" if baseline else ""
            print(f"Day {d} zonal-aggregated MAPE: "
                  f"{float(ape.mean() * 100):.2f}%{base_str}  [saved]",
                  flush=True)

    if all(lead_file(d).exists() for d in range(1, 6)):
        summarize(baseline)
    else:
        print("\nnot all leads complete yet - relaunch to continue", flush=True)


if __name__ == "__main__":
    main()
