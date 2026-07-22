"""
Zonal TFT retrain — lr 3e-4 + plateau scheduler, speed-optimized and resumable.
Base to beat: floor checkpoint (lr=1e-3) = 3.37% zonal MAPE.
"""
from pathlib import Path
import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

FEATURES = Path("zonal_features.parquet")
CKPT_DIR = Path("checkpoints_zonal")
TRAIN_START = "2015-07-01 04:00"
VAL_START   = "2023-01-01 05:00"
TEST_START  = "2024-01-23 12:00"
ENCODER_LEN, DECODER_LEN = 168, 120
BATCH = 128; MAX_EPOCHS = 40; SEED = 42
NUM_WORKERS = 4

WEATHER = ["temperature_2m","apparent_temperature","relative_humidity_2m",
           "wind_speed_10m","shortwave_radiation","cloud_cover","temp_vshape"]
UNKNOWN_REALS = ["demand","demand_lag24","demand_lag168",
                 "demand_roll24_mean","demand_roll168_mean","demand_roll24_std"]
KNOWN_REALS = WEATHER + ["time_idx"]
KNOWN_CATS = ["hour","day_of_week","month","is_weekend","is_holiday"]

def main():
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision("high")

    df = pd.read_parquet(FEATURES)
    df["utc"] = pd.to_datetime(df["utc"])
    for c in KNOWN_CATS: df[c] = df[c].astype(str).astype("category")
    df["zone"] = df["zone"].astype(str)
    df = df[df["utc"] >= TRAIN_START].copy()
    df["time_idx"] = df["time_idx"] - df["time_idx"].min()
    val_idx = int(df.loc[df["utc"] >= VAL_START, "time_idx"].min())
    test_idx = int(df.loc[df["utc"] >= TEST_START, "time_idx"].min())
    max_idx = int(df["time_idx"].max())
    print(f"boundaries - val:{val_idx} test:{test_idx} max:{max_idx}", flush=True)

    train_df = df[df["time_idx"] < val_idx]
    training_ds = TimeSeriesDataSet(
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
    validation_ds = TimeSeriesDataSet.from_dataset(
        training_ds, df[df["time_idx"] < test_idx],
        min_prediction_idx=val_idx, stop_randomization=True)

    train_dl = training_ds.to_dataloader(train=True, batch_size=BATCH,
                                         num_workers=NUM_WORKERS, persistent_workers=True)
    val_dl = validation_ds.to_dataloader(train=False, batch_size=BATCH,
                                         num_workers=NUM_WORKERS, persistent_workers=True)

    model = TemporalFusionTransformer.from_dataset(
        training_ds,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        loss=QuantileLoss(),
        learning_rate=3e-4,
        reduce_on_plateau_patience=2,
        log_interval=100,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=CKPT_DIR, filename="zonal_tft_lr3e4_best",
                              monitor="val_loss", mode="min", save_top_k=1,
                              save_last=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, mode="min"),
        ckpt_cb,
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS, accelerator="auto",
        gradient_clip_val=0.1, callbacks=callbacks,
        enable_progress_bar=False,
        log_every_n_steps=200,
    )

    resume = CKPT_DIR / "last.ckpt"
    resume_path = str(resume) if resume.exists() else None
    print(f"resuming from: {resume_path}", flush=True)

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl,
                ckpt_path=resume_path, weights_only=False)

    print("Best checkpoint:", ckpt_cb.best_model_path, flush=True)
    print("Best val_loss:  ", float(ckpt_cb.best_model_score), flush=True)

if __name__ == "__main__":
    main()
