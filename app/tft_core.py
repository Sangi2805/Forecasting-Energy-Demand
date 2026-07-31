"""
Shared zonal-TFT inference core.

Mirrors Sangar/06_eval_refit2023_fx.py exactly (constants, fx recipe, dataset
construction) but with repo-relative paths so it runs off the pod. Both the
validation harness and the live forecasting path import from here, so there is
one definition of the feature contract instead of two that can drift.

Heavy imports (torch, pytorch_forecasting) are deferred into the functions that
need them; importing this module for its constants alone stays cheap.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SANGAR = ROOT / "Sangar"

FEATURES = SANGAR / "zonal_features_fx.parquet"
LEADS = SANGAR / "weather_leads_zonal.parquet"
CKPT = SANGAR / "zonal_tft_refit2023_fx_best.ckpt"
DS_PARAMS = SANGAR / "tft_ds_params.pkl"

# Both artifacts are committed to the repo, so nothing here normally reaches the
# network. The Hub is a fallback for deployments that would rather clone a slim
# repo and pull the 6 MB checkpoint at runtime; it is never required.
HF_MODEL_REPO = "shahriarrashid54/energyai-nyiso-tft"


def resolve_artifact(local: Path) -> Path:
    """Local file if present, otherwise the same filename off the Hub."""
    if local.exists():
        return local
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FileNotFoundError(
            f"{local} is missing and huggingface_hub is not installed, so it "
            f"cannot be fetched from {HF_MODEL_REPO}."
        ) from exc
    return Path(hf_hub_download(repo_id=HF_MODEL_REPO, filename=local.name))

# ---------------------------------------------------------------------------
# Model / dataset contract -- must match 05_train_refit2023_fx.py
# ---------------------------------------------------------------------------
TRAIN_START = "2015-07-01 04:00"
VAL_START = "2024-01-01 05:00"
TEST_START = "2024-01-23 12:00"
ENCODER_LEN, DECODER_LEN = 168, 120
BATCH, SEED = 128, 42
HOT_DAY_C, CDH_BASE_C = 24.0, 22.0
TEMP_VSHAPE_REF = 14.0

# time_idx is a known real the model consumes, so live rows must sit on the same
# absolute clock the checkpoint was fitted against. Verified against
# zonal_features_fx.parquet: time_idx 0 == 2015-07-01 04:00 UTC, zero drift over
# all 95,708 hours to 2026-05-31.
TIME_IDX_EPOCH = pd.Timestamp(TRAIN_START)

ZONE_WEATHER = {
    "WEST": "A_WEST", "GENESE": "B_GENESE", "CENTRL": "C_CENTRL",
    "NORTH": "D_NORTH", "MHK VL": "E_MHKVL", "CAPITL": "F_CAPITL",
    "HUD VL": "G_HUDVL", "MILLWD": "G_HUDVL", "DUNWOD": "J_NYC",
    "N.Y.C.": "J_NYC", "LONGIL": "K_LONGIL",
}

# Open-Meteo coordinates behind each weather key (from 02_download_zonal_weather).
COORDS = {
    "A_WEST": (42.886, -78.878),
    "B_GENESE": (43.157, -77.616),
    "C_CENTRL": (43.048, -76.147),
    "D_NORTH": (44.928, -74.892),
    "E_MHKVL": (43.101, -75.233),
    "F_CAPITL": (42.653, -73.757),
    "G_HUDVL": (41.706, -73.921),
    "J_NYC": (40.714, -74.006),
    "K_LONGIL": (40.730, -73.210),
}

LEAN_VARS = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "wind_speed_10m", "shortwave_radiation", "cloud_cover",
]
FX = ["fx_app_roll72", "fx_cdh24", "fx_hot_streak_day"]
UNKNOWN_REALS = [
    "demand", "demand_lag24", "demand_lag168",
    "demand_roll24_mean", "demand_roll168_mean", "demand_roll24_std",
]
KNOWN_REALS = LEAN_VARS + ["temp_vshape"] + FX + ["time_idx"]
KNOWN_CATS = ["hour", "day_of_week", "month", "is_weekend", "is_holiday"]

ZONES = sorted(ZONE_WEATHER)


# ---------------------------------------------------------------------------
# Feature recipe (verbatim from 06_eval_refit2023_fx.py:36-55)
# ---------------------------------------------------------------------------
def hot_streak(flags) -> pd.Series:
    """Running count of consecutive True days, reset by any False.

    Note this is unbounded: a streak carries across the whole series it is
    computed over. Any live window must therefore seed it from real history
    rather than starting the count at the window edge.
    """
    out, run = [], 0
    for h in flags:
        run = run + 1 if h else 0
        out.append(run)
    return pd.Series(out, index=flags.index)


def add_fx(df: pd.DataFrame) -> pd.DataFrame:
    """Heat-memory features. Recomputed after any weather swap, never carried over."""
    df = df.sort_values(["zone", "utc"]).reset_index(drop=True)
    df["fx_app_roll72"] = (
        df.groupby("zone")["apparent_temperature"]
        .transform(lambda s: s.rolling(72, min_periods=1).mean())
    )
    cdh = (df["apparent_temperature"] - CDH_BASE_C).clip(lower=0)
    df["fx_cdh24"] = cdh.groupby(df["zone"]).transform(
        lambda s: s.rolling(24, min_periods=1).sum()
    )
    d = df["utc"].dt.date
    daily = (
        df.assign(_d=d)
        .groupby(["zone", "_d"])["apparent_temperature"]
        .max()
        .rename("dmax")
        .reset_index()
    )
    daily["hot"] = daily["dmax"] >= HOT_DAY_C
    daily["fx_hot_streak_day"] = (
        daily.groupby("zone")["hot"].transform(hot_streak).astype(float)
    )
    df = df.assign(_d=d).merge(
        daily[["zone", "_d", "fx_hot_streak_day"]],
        on=["zone", "_d"], how="left", suffixes=("_old", ""),
    )
    df = df.drop(columns=[c for c in df.columns if c.endswith("_old") or c == "_d"])
    return df


def swap_lead(base: pd.DataFrame, leads: pd.DataFrame, d: int) -> pd.DataFrame:
    """
    Replace observed weather with the lead-`d` forecast weather (verbatim from
    06_eval_refit2023_fx.py:65-77). Used only to replay historical forecasts;
    the live path pulls a real forward forecast instead.
    """
    out = base.copy()
    boundary = pd.Timestamp(TEST_START)
    fallback = 0
    for zone, wkey in ZONE_WEATHER.items():
        mask = out["zone"] == zone
        utc = out.loc[mask, "utc"]
        post = utc >= boundary
        for v in LEAN_VARS:
            s = utc.map(leads[f"{v}_prev_day{d}__{wkey}"]).interpolate(limit=6)
            s5 = utc.map(leads[f"{v}_prev_day5__{wkey}"]).interpolate(limit=6)
            s = s.fillna(s5)
            fallback += int((s.isna() & post).sum())
            s = s.fillna(out.loc[mask, v])
            out.loc[mask, v] = s.values
    out["temp_vshape"] = (out["temperature_2m"] - TEMP_VSHAPE_REF).abs()
    out = add_fx(out)
    if fallback:
        print(f"  swap_lead fallback cells: {fallback}  <-- INVESTIGATE")
    return out


def load_base() -> pd.DataFrame:
    """Historical feature table, prepared exactly as the eval script does."""
    df = pd.read_parquet(FEATURES)
    df["utc"] = pd.to_datetime(df["utc"])
    for c in KNOWN_CATS:
        df[c] = df[c].astype(str).astype("category")
    df["zone"] = df["zone"].astype(str)
    df = df[df["utc"] >= TRAIN_START].copy()
    df["time_idx"] = df["time_idx"] - df["time_idx"].min()
    return df.sort_values(["zone", "time_idx"]).reset_index(drop=True)


def build_training_ds(df: pd.DataFrame):
    """The exact TimeSeriesDataSet the checkpoint was fitted against."""
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    val_idx = int(df.loc[df["utc"] >= VAL_START, "time_idx"].min())
    return TimeSeriesDataSet(
        df[df["time_idx"] < val_idx],
        time_idx="time_idx", target="demand", group_ids=["zone"],
        max_encoder_length=ENCODER_LEN, max_prediction_length=DECODER_LEN,
        time_varying_unknown_reals=UNKNOWN_REALS,
        time_varying_known_reals=KNOWN_REALS,
        time_varying_known_categoricals=KNOWN_CATS,
        static_categoricals=["zone"],
        target_normalizer=GroupNormalizer(groups=["zone"]),
        add_relative_time_idx=True, add_target_scales=True,
        allow_missing_timesteps=False,
    )


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
def save_ds_params(path: Path = DS_PARAMS) -> dict:
    """Build the training dataset once and bank its parameters (normalizer state)."""
    params = build_training_ds(load_base()).get_parameters()
    with open(path, "wb") as fh:
        pickle.dump(params, fh)
    return params


def load_ds_params(path: Path = DS_PARAMS) -> dict:
    with open(resolve_artifact(path), "rb") as fh:
        return pickle.load(fh)


def load_model(ckpt: Path = CKPT):
    """
    Load the refit2023+fx checkpoint on CPU.

    Two CAIR-isms have to be neutralised to load this on a CPU-only box:

    1. torch >= 2.6 flipped torch.load's weights_only default to True, which
       cannot unpickle the GroupNormalizer stored in the checkpoint.
    2. The QuantileLoss / logging_metrics objects were pickled on a CUDA box and
       carry device='cuda'. Lightning's post-load .to(device) walk asks each
       torchmetrics Metric to allocate on its recorded device, which raises
       "Torch not compiled with CUDA enabled". Pin that recorded device to CPU.
    """
    import torch
    import torchmetrics
    from pytorch_forecasting import TemporalFusionTransformer

    _orig_load = torch.load
    _orig_apply = torchmetrics.Metric._apply

    def _load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    def _apply(self, *args, **kwargs):
        self._device = torch.device("cpu")
        return _orig_apply(self, *args, **kwargs)

    torch.load = _load
    torchmetrics.Metric._apply = _apply
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(
            str(resolve_artifact(ckpt)), map_location="cpu"
        )
    finally:
        torch.load = _orig_load
        torchmetrics.Metric._apply = _orig_apply
    return model.eval()


def dataset_from_params(params: dict, df: pd.DataFrame, **kwargs):
    from pytorch_forecasting import TimeSeriesDataSet

    return TimeSeriesDataSet.from_parameters(params, df, **kwargs)


def predict_batched(model, dataset, device: str = "cpu"):
    """
    Median-quantile prediction for every window, with its identity attached.

    One dataset spanning all zones instead of one per zone: the model call is
    cheap, but TimeSeriesDataSet construction is not, so building it eleven
    times dominated the runtime. x_to_index maps each row of the output back to
    its (zone, decoder-start time_idx) rather than relying on batch ordering.

    All quantiles are returned, not just the median. The checkpoint was trained
    with QuantileLoss, so the spread is a real predictive interval the model
    learned -- discarding it would mean inventing an error band later.

    Returns (pred[n_windows, DECODER_LEN, n_quantiles], index_df with zone + time_idx).
    """
    import numpy as np
    import pandas as pd
    import torch

    dl = dataset.to_dataloader(train=False, batch_size=BATCH, num_workers=0)
    preds, index = [], []
    with torch.no_grad():
        for x, _ in dl:
            xd = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
            raw = model(xd)
            preds.append(raw["prediction"].cpu().numpy())
            index.append(dataset.x_to_index(x))
    return np.concatenate(preds), pd.concat(index, ignore_index=True)


def model_quantiles(model) -> list[float]:
    """Quantile levels the checkpoint predicts, in output order."""
    q = getattr(getattr(model, "loss", None), "quantiles", None)
    return list(q) if q else [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]


def predict_zone(model, dataset, device: str = "cpu"):
    """Median-quantile prediction for every window in `dataset`. Returns (pred, actual, start_idx)."""
    import numpy as np
    import torch

    dl = dataset.to_dataloader(train=False, batch_size=BATCH, num_workers=0)
    yh, yt, tt = [], [], []
    with torch.no_grad():
        for x, (y, _) in dl:
            xd = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
            raw = model(xd)
            p = raw["prediction"][..., raw["prediction"].shape[-1] // 2]
            yh.append(p.cpu().numpy())
            yt.append(y.cpu().numpy())
            tt.append(x["decoder_time_idx"][:, 0].cpu().numpy())
    return np.concatenate(yh), np.concatenate(yt), np.concatenate(tt)
