"""
Lead-matched eval for the refit2023 + fx model.
fx features are recomputed AFTER the lead swap (no observed-future leak).
Compare against: 3.20% (tuned), 3.37% (floor), 3.96% (statewide).
"""
import os, json, gc, glob
import numpy as np, pandas as pd, torch
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

FOLDER    = os.path.dirname(os.path.abspath(__file__))
FEATURES  = os.path.join(FOLDER, "zonal_features_fx.parquet")
LEADS     = os.path.join(FOLDER, "weather_leads_zonal.parquet")
CKPT      = os.path.join(FOLDER, "checkpoints_zonal", "zonal_tft_refit2023_fx_best.ckpt")
OUT_JSON  = os.path.join(FOLDER, "metrics_zonal_refit2023_fx.json")

TRAIN_START="2015-07-01 04:00"; VAL_START="2024-01-01 05:00"; TEST_START="2024-01-23 12:00"
ENCODER_LEN, DECODER_LEN = 168, 120
BATCH, SEED = 128, 42
HOT_DAY_C, CDH_BASE_C = 24.0, 22.0

ZONE_WEATHER = {"WEST":"A_WEST","GENESE":"B_GENESE","CENTRL":"C_CENTRL","NORTH":"D_NORTH",
    "MHK VL":"E_MHKVL","CAPITL":"F_CAPITL","HUD VL":"G_HUDVL","MILLWD":"G_HUDVL",
    "DUNWOD":"J_NYC","N.Y.C.":"J_NYC","LONGIL":"K_LONGIL"}
LEAN_VARS=["temperature_2m","apparent_temperature","relative_humidity_2m",
           "wind_speed_10m","shortwave_radiation","cloud_cover"]
FX=["fx_app_roll72","fx_cdh24","fx_hot_streak_day"]
UNKNOWN_REALS=["demand","demand_lag24","demand_lag168",
               "demand_roll24_mean","demand_roll168_mean","demand_roll24_std"]
KNOWN_REALS=LEAN_VARS+["temp_vshape"]+FX+["time_idx"]
KNOWN_CATS=["hour","day_of_week","month","is_weekend","is_holiday"]

def lead_file(d): return os.path.join(FOLDER, "fx_lead%d.npz" % d)

def add_fx(df):
    df = df.sort_values(["zone","utc"]).reset_index(drop=True)
    df["fx_app_roll72"] = (df.groupby("zone")["apparent_temperature"]
                             .transform(lambda s: s.rolling(72, min_periods=1).mean()))
    cdh = (df["apparent_temperature"] - CDH_BASE_C).clip(lower=0)
    df["fx_cdh24"] = cdh.groupby(df["zone"]).transform(lambda s: s.rolling(24, min_periods=1).sum())
    d = df["utc"].dt.date
    daily = df.assign(_d=d).groupby(["zone","_d"])["apparent_temperature"].max().rename("dmax").reset_index()
    daily["hot"] = daily["dmax"] >= HOT_DAY_C
    def streak(s):
        out, run = [], 0
        for h in s:
            run = run + 1 if h else 0
            out.append(run)
        return pd.Series(out, index=s.index)
    daily["fx_hot_streak_day"] = daily.groupby("zone")["hot"].transform(streak).astype(float)
    df = df.assign(_d=d).merge(daily[["zone","_d","fx_hot_streak_day"]],
                               on=["zone","_d"], how="left", suffixes=("_old",""))
    df = df.drop(columns=[c for c in df.columns if c.endswith("_old") or c == "_d"])
    return df

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
            s=utc.map(leads["%s_prev_day%d__%s" % (v,d,wkey)]).interpolate(limit=6)
            s5=utc.map(leads["%s_prev_day5__%s" % (v,wkey)]).interpolate(limit=6)
            s=s.fillna(s5); fb+=int((s.isna()&post).sum())
            s=s.fillna(out.loc[mask,v]); out.loc[mask,v]=s.values
    out["temp_vshape"]=(out["temperature_2m"]-14.0).abs()
    out=add_fx(out)
    print("  fallback cells: %d%s" % (fb, "  <-- INVESTIGATE" if fb else ""), flush=True)
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
    yh, yt, tt = [], [], []
    for bi, (x, (y, _)) in enumerate(dl):
        xd = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in x.items()}
        raw = model(xd)
        p = raw["prediction"][..., raw["prediction"].shape[-1]//2].cpu().numpy()
        if debug and bi == 0:
            print("      [debug] pred_MW~%.0f true~%.0f" % (p[0,0], float(y[0,0])), flush=True)
        yh.append(p); yt.append(y.cpu().numpy())
        tt.append(x["decoder_time_idx"][:,0].cpu().numpy())
        del xd, raw, p
    del dl, ds; gc.collect()
    if device=="cuda": torch.cuda.empty_cache()
    return np.concatenate(yh), np.concatenate(yt), np.concatenate(tt)

def main():
    pl.seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df=load_base()
    missing=[c for c in KNOWN_REALS if c not in df.columns]
    if missing: raise SystemExit("missing columns: " + str(missing))

    leads=pd.read_parquet(LEADS); leads.index=pd.to_datetime(leads.index)
    if getattr(leads.index,"tz",None) is not None: leads.index=leads.index.tz_localize(None)

    tsi=int(df.loc[df["utc"]>=TEST_START,"time_idx"].min())
    training_ds=build_training_ds(df)
    model=TemporalFusionTransformer.load_from_checkpoint(CKPT).to(device).eval()

    one=df[df["zone"]==df["zone"].iloc[0]]
    month_arr=np.full(int(df["time_idx"].max())+DECODER_LEN+2,-1)
    month_arr[one["time_idx"].values]=one["utc"].dt.month.values

    df=df[df["time_idx"]>=tsi-ENCODER_LEN-1].copy()
    print("trimmed to %d rows, test_start_idx=%d, device=%s" % (len(df), tsi, device), flush=True)
    ZONES=sorted(df["zone"].unique())

    FLOOR=[2.48,2.89,3.14,3.52,3.96]
    day_mape, composite = {}, []
    for d in range(1,6):
        print("\n=== Lead %d ===" % d, flush=True)
        if os.path.exists(lead_file(d)):
            z=np.load(lead_file(d)); ape,months=z["ape"],z["months"]; print("  cached",flush=True)
        else:
            df_d=swap_lead(df,leads,d)
            hat,true={},{}
            for zi,zone in enumerate(ZONES):
                yh,yt,tt=predict_one_zone(model,training_ds,df_d[df_d["zone"]==zone],tsi,device,
                                          debug=(d==1 and zi==0))
                for i,t in enumerate(tt):
                    hat.setdefault(int(t),[]).append(yh[i])
                    true.setdefault(int(t),[]).append(yt[i])
                print("    %-10s (%d/11)" % (zone, zi+1), flush=True)
            del df_d; gc.collect()
            good=sorted(t for t,v in hat.items() if len(v)==11)
            drop=len(hat)-len(good)
            if drop: print("  dropped %d incomplete windows" % drop, flush=True)
            sh=np.stack([np.sum(hat[t],axis=0) for t in good])
            st=np.stack([np.sum(true[t],axis=0) for t in good])
            ut=np.array(good); s,e=(d-1)*24,d*24
            ape=np.abs(st[:,s:e]-sh[:,s:e])/np.clip(st[:,s:e],1e-6,None)
            months=month_arr[ut[:,None]+np.arange(s,e)[None,:]]
            np.savez(lead_file(d),ape=ape,months=months)
            del hat,true; gc.collect()
        day_mape[d]=float(ape.mean()*100)
        print("Day %d: %.2f%%   (tuned %.2f%%)  [saved]" % (d, day_mape[d], FLOOR[d-1]), flush=True)
        composite.append((ape.ravel(),months.ravel()))

    all_ape=np.concatenate([a for a,_ in composite]); all_mon=np.concatenate([m for _,m in composite])
    overall=float(all_ape.mean()*100)
    smap={12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
          6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
    seasons={}
    for sn in ["Winter","Spring","Summer","Fall"]:
        m=np.isin(all_mon,[k for k,v in smap.items() if v==sn])
        seasons[sn]=float(all_ape[m].mean()*100) if m.any() else None

    print("\n===== REFIT2023 + FX RESULTS =====")
    for d in range(1,6):
        print("Day %d: %.2f%%   (tuned %.2f%%)" % (d, day_mape[d], FLOOR[d-1]))
    print("Overall: %.2f%%   (tuned 3.20%%, floor 3.37%%, statewide 3.96%%)" % overall)
    print("\nSeasonal (tuned: Winter 3.35, Spring 3.37, Summer 3.60, Fall 2.35):")
    for sn,v in seasons.items():
        print("  %s: %.2f%%" % (sn, v) if v else "  %s: n/a" % sn)

    json.dump({"day_mape":{str(k):round(v,3) for k,v in day_mape.items()},
        "overall_mape":round(overall,3),
        "seasonal_mape":{k:(round(v,3) if v else None) for k,v in seasons.items()},
        "checkpoint":"zonal_tft_refit2023_fx_best.ckpt (2 epochs, lr=3e-4, +2023, +fx)"},
        open(OUT_JSON,"w"),indent=2)
    print("\nSaved -> " + OUT_JSON)

if __name__ == "__main__":
    main()
