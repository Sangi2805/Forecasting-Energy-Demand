"""
Zonal TFT lead-matched evaluation (NYISO 11-zone -> statewide aggregation).
Result: overall 3.37% MAPE vs statewide baseline 3.96% (epoch-0 floor, lr=1e-3).
Manual batched inference: pf 1.8 forward() returns predictions already in MW.
Resumable: each lead banks to zonal_lead{d}.npz.
"""
import os, json, gc, glob
import numpy as np, pandas as pd, torch
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# --- paths: override FOLDER via env var for Colab vs local ---
FOLDER    = os.environ.get("ZONAL_DIR", ".")
FEATURES  = os.path.join(FOLDER, "zonal_features.parquet")
LEADS     = os.path.join(FOLDER, "weather_leads_zonal.parquet")
CKPT      = os.path.join(FOLDER, "zonal_tft_best.ckpt")
BASE_JSON = os.path.join(FOLDER, "metrics_lead_matched.json")
OUT_JSON  = os.path.join(FOLDER, "metrics_zonal_lead_matched.json")

TRAIN_START="2015-07-01 04:00"; VAL_START="2023-01-01 05:00"; TEST_START="2024-01-23 12:00"
ENCODER_LEN, DECODER_LEN = 168, 120
BATCH, SEED = 128, 42

ZONE_WEATHER = {"WEST":"A_WEST","GENESE":"B_GENESE","CENTRL":"C_CENTRL","NORTH":"D_NORTH",
    "MHK VL":"E_MHKVL","CAPITL":"F_CAPITL","HUD VL":"G_HUDVL","MILLWD":"G_HUDVL",
    "DUNWOD":"J_NYC","N.Y.C.":"J_NYC","LONGIL":"K_LONGIL"}
LEAN_VARS=["temperature_2m","apparent_temperature","relative_humidity_2m","wind_speed_10m","shortwave_radiation","cloud_cover"]
UNKNOWN_REALS=["demand","demand_lag24","demand_lag168","demand_roll24_mean","demand_roll168_mean","demand_roll24_std"]
KNOWN_REALS=LEAN_VARS+["temp_vshape","time_idx"]
KNOWN_CATS=["hour","day_of_week","month","is_weekend","is_holiday"]

def lead_file(d): return os.path.join(FOLDER, f"zonal_lead{d}.npz")

def load_base():
    df=pd.read_parquet(FEATURES); df["utc"]=pd.to_datetime(df["utc"])
    for c in KNOWN_CATS: df[c]=df[c].astype(str).astype("category")
    df["zone"]=df["zone"].astype(str)
    df=df[df["utc"]>=TRAIN_START].copy()
    df["time_idx"]=df["time_idx"]-df["time_idx"].min()
    return df.sort_values(["zone","time_idx"]).reset_index(drop=True)

def swap_lead(base, leads, d):
    out=base.copy(); boundary=pd.Timestamp(TEST_START); fb=0
    for zone,wkey in ZONE_WEATHER.items():
        mask=out["zone"]==zone; utc=out.loc[mask,"utc"]; post=utc>=boundary
        for v in LEAN_VARS:
            s=utc.map(leads[f"{v}_prev_day{d}__{wkey}"]).interpolate(limit=6)
            s5=utc.map(leads[f"{v}_prev_day5__{wkey}"]).interpolate(limit=6)
            s=s.fillna(s5); fb+=int((s.isna()&post).sum())
            s=s.fillna(out.loc[mask,v]); out.loc[mask,v]=s.values
    out["temp_vshape"]=(out["temperature_2m"]-14.0).abs()
    print(f"  fallback cells: {fb}{'  <-- INVESTIGATE' if fb else ''}", flush=True)
    return out

def build_training_ds(df):
    val_idx=int(df.loc[df["utc"]>=VAL_START,"time_idx"].min())
    return TimeSeriesDataSet(df[df["time_idx"]<val_idx],
        time_idx="time_idx",target="demand",group_ids=["zone"],
        max_encoder_length=ENCODER_LEN,max_prediction_length=DECODER_LEN,
        time_varying_unknown_reals=UNKNOWN_REALS,time_varying_known_reals=KNOWN_REALS,
        time_varying_known_categoricals=KNOWN_CATS,static_categoricals=["zone"],
        target_normalizer=GroupNormalizer(groups=["zone"]),
        add_relative_time_idx=True,add_target_scales=True,allow_missing_timesteps=False)

@torch.no_grad()
def predict_one_zone(model, training_ds, df_zone, tsi, device, debug=False):
    ds = TimeSeriesDataSet.from_dataset(training_ds, df_zone,
        min_prediction_idx=tsi, stop_randomization=True)
    dl = ds.to_dataloader(train=False, batch_size=BATCH, num_workers=0)
    yhat_parts, ytrue_parts, tidx_parts = [], [], []
    for bi, (x, (y, _)) in enumerate(dl):
        xd = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
        raw = model(xd)
        yhat = raw["prediction"][..., raw["prediction"].shape[-1] // 2].cpu().numpy()  # already MW
        if debug and bi == 0:
            print(f"      [debug] pred_MW~{yhat[0,0]:.0f} true~{float(y[0,0]):.0f}", flush=True)
        yhat_parts.append(yhat); ytrue_parts.append(y.cpu().numpy())
        tidx_parts.append(x["decoder_time_idx"][:, 0].cpu().numpy())
        del xd, raw, yhat
    del dl, ds; gc.collect()
    if device == "cuda": torch.cuda.empty_cache()
    return (np.concatenate(yhat_parts), np.concatenate(ytrue_parts), np.concatenate(tidx_parts))

def main():
    pl.seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline = {int(k):v for k,v in json.load(open(BASE_JSON))["day_mape"].items()} if os.path.exists(BASE_JSON) else None

    df=load_base()
    leads=pd.read_parquet(LEADS); leads.index=pd.to_datetime(leads.index)
    if getattr(leads.index,"tz",None) is not None: leads.index=leads.index.tz_localize(None)

    tsi=int(df.loc[df["utc"]>=TEST_START,"time_idx"].min())
    training_ds=build_training_ds(df)
    model=TemporalFusionTransformer.load_from_checkpoint(CKPT).to(device).eval()

    one=df[df["zone"]==df["zone"].iloc[0]]
    month_arr=np.full(int(df["time_idx"].max())+DECODER_LEN+2,-1)
    month_arr[one["time_idx"].values]=one["utc"].dt.month.values

    df=df[df["time_idx"]>=tsi-ENCODER_LEN-1].copy()
    print(f"trimmed to {len(df):,} rows, test_start_idx={tsi}, device={device}", flush=True)
    ZONES=sorted(df["zone"].unique())

    day_mape, composite = {}, []
    for d in range(1,6):
        print(f"\n=== Lead {d} ===", flush=True)
        if os.path.exists(lead_file(d)):
            z=np.load(lead_file(d)); ape,months=z["ape"],z["months"]; print("  cached",flush=True)
        else:
            df_d=swap_lead(df,leads,d)
            hat_by_t,true_by_t={},{}
            for zi,zone in enumerate(ZONES):
                yh,yt,tt=predict_one_zone(model,training_ds,df_d[df_d["zone"]==zone],tsi,device,
                                          debug=(d==1 and zi==0))
                for i,t in enumerate(tt):
                    hat_by_t.setdefault(int(t),[]).append(yh[i])
                    true_by_t.setdefault(int(t),[]).append(yt[i])
                print(f"    {zone:10s} ({zi+1}/11)", flush=True)
            del df_d; gc.collect()
            good=sorted(t for t,v in hat_by_t.items() if len(v)==11)
            drop=len(hat_by_t)-len(good)
            if drop: print(f"  dropped {drop} incomplete windows",flush=True)
            sh=np.stack([np.sum(hat_by_t[t],axis=0) for t in good])
            st=np.stack([np.sum(true_by_t[t],axis=0) for t in good])
            ut=np.array(good); s,e=(d-1)*24,d*24
            ape=np.abs(st[:,s:e]-sh[:,s:e])/np.clip(st[:,s:e],1e-6,None)
            months=month_arr[ut[:,None]+np.arange(s,e)[None,:]]
            np.savez(lead_file(d),ape=ape,months=months)
            del hat_by_t,true_by_t; gc.collect()
        day_mape[d]=float(ape.mean()*100)
        b=f" (baseline {baseline[d]:.2f}%)" if baseline else ""
        print(f"Day {d}: {day_mape[d]:.2f}%{b}  [saved]",flush=True)
        composite.append((ape.ravel(),months.ravel()))

    all_ape=np.concatenate([a for a,_ in composite]); all_mon=np.concatenate([m for _,m in composite])
    overall=float(all_ape.mean()*100)
    smap={12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
    seasons={sn:(float(all_ape[np.isin(all_mon,[m for m,x in smap.items() if x==sn])].mean()*100)
        if np.isin(all_mon,[m for m,x in smap.items() if x==sn]).any() else None)
        for sn in ["Winter","Spring","Summer","Fall"]}

    print("\n============ ZONAL LEAD-MATCHED RESULTS ============")
    for d in range(1,6):
        b=f"   (statewide baseline {baseline[d]:.2f}%)" if baseline else ""
        print(f"Day {d}: {day_mape[d]:.2f}%{b}")
    print(f"Overall: {overall:.2f}%   (statewide baseline 3.96%)")
    print("\nSeasonal:")
    for sn,v in seasons.items(): print(f"  {sn}: {v:.2f}%" if v else f"  {sn}: n/a")

    json.dump({"day_mape":{str(k):round(v,3) for k,v in day_mape.items()},
        "overall_mape":round(overall,3),
        "seasonal_mape":{k:(round(v,3) if v else None) for k,v in seasons.items()},
        "checkpoint":"zonal_tft_best.ckpt (epoch-0, lr=1e-3)"},open(OUT_JSON,"w"),indent=2)
    print(f"\nSaved -> {OUT_JSON}")

if __name__ == "__main__":
    main()