"""
Permutation importance for the 2.95% zonal TFT.
Measures, in MAPE points, how much accuracy is lost when each feature is
replaced by noise. Individual features plus correlated groups.
Usage:  python 07_permutation_importance.py day1
        python 07_permutation_importance.py all5
"""
import os, sys, gc, json, time
import numpy as np, pandas as pd, torch
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

MODE   = sys.argv[1] if len(sys.argv) > 1 else "day1"
LEADS  = [1] if MODE == "day1" else [1, 2, 3, 4, 5]
STRIDE = 6 if MODE == "day1" else 15

FOLDER   = os.path.dirname(os.path.abspath(__file__))
FEATURES = os.path.join(FOLDER, "zonal_features_fx.parquet")
LEADFILE = os.path.join(FOLDER, "weather_leads_zonal.parquet")
CKPT     = os.path.join(FOLDER, "checkpoints_zonal", "zonal_tft_refit2023_fx_best.ckpt")
OUT_JSON = os.path.join(FOLDER, "perm_importance_%s.json" % MODE)
OUT_CSV  = os.path.join(FOLDER, "perm_importance_%s.csv" % MODE)

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

LAGS = ["demand_lag24","demand_lag168","demand_roll24_mean",
        "demand_roll168_mean","demand_roll24_std"]
INDIVIDUAL = LEAN_VARS + ["temp_vshape"] + FX + LAGS + KNOWN_CATS
GROUPS = {
  "GROUP temperature family": ["temperature_2m","apparent_temperature","temp_vshape"],
  "GROUP solar":              ["shortwave_radiation","cloud_cover"],
  "GROUP heat memory":        FX,
  "GROUP all raw weather":    LEAN_VARS,
  "GROUP all weather-derived": LEAN_VARS + ["temp_vshape"] + FX,
  "GROUP demand history":     LAGS,
  "GROUP calendar":           KNOWN_CATS,
}

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
    return df.drop(columns=[c for c in df.columns if c.endswith("_old") or c == "_d"])

def load_base():
    df=pd.read_parquet(FEATURES); df["utc"]=pd.to_datetime(df["utc"])
    for c in KNOWN_CATS: df[c]=df[c].astype(str).astype("category")
    df["zone"]=df["zone"].astype(str)
    df=df[df["utc"]>=TRAIN_START].copy()
    df["time_idx"]=df["time_idx"]-df["time_idx"].min()
    return df.sort_values(["zone","time_idx"]).reset_index(drop=True)

def swap_lead(base, leads, d):
    out=base.copy(); fb=0
    for zone,wkey in ZONE_WEATHER.items():
        mask=out["zone"]==zone; utc=out.loc[mask,"utc"]
        for v in LEAN_VARS:
            s=utc.map(leads["%s_prev_day%d__%s" % (v,d,wkey)]).interpolate(limit=6)
            s5=utc.map(leads["%s_prev_day5__%s" % (v,wkey)]).interpolate(limit=6)
            s=s.fillna(s5).fillna(out.loc[mask,v]); out.loc[mask,v]=s.values
    out["temp_vshape"]=(out["temperature_2m"]-14.0).abs()
    return add_fx(out)

def permute(df, cols, seed):
    """Shuffle cols within each zone. One shared permutation per group so
    correlated members stay internally consistent."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    zv = out["zone"].values
    dtypes = {c: df[c].dtype for c in cols}
    arrs = {c: out[c].to_numpy(copy=True) for c in cols}
    for z in np.unique(zv):
        m = zv == z
        p = rng.permutation(int(m.sum()))
        for c in cols:
            a = arrs[c]; a[m] = a[m][p]
    for c in cols:
        out[c] = pd.Series(arrs[c], index=out.index).astype(dtypes[c])
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
def predict_zone(model, tds, df_zone, tsi, device):
    ds = TimeSeriesDataSet.from_dataset(tds, df_zone, min_prediction_idx=tsi,
                                        stop_randomization=True)
    dl = ds.to_dataloader(train=False, batch_size=BATCH, num_workers=0)
    yh, yt, tt = [], [], []
    for bi, (x, (y, _)) in enumerate(dl):
        if bi % STRIDE: continue
        xd = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in x.items()}
        raw = model(xd)
        yh.append(raw["prediction"][..., raw["prediction"].shape[-1]//2].cpu().numpy())
        yt.append(y.cpu().numpy())
        tt.append(x["decoder_time_idx"][:,0].cpu().numpy())
        del xd, raw
    del dl, ds; gc.collect()
    if device=="cuda": torch.cuda.empty_cache()
    if not yh: return None, None, None
    return np.concatenate(yh), np.concatenate(yt), np.concatenate(tt)

def score(model, tds, df_d, tsi, device, zones, lead):
    hat, true = {}, {}
    for z in zones:
        yh, yt, tt = predict_zone(model, tds, df_d[df_d["zone"]==z], tsi, device)
        if yh is None: continue
        for i,t in enumerate(tt):
            hat.setdefault(int(t),[]).append(yh[i])
            true.setdefault(int(t),[]).append(yt[i])
    good=sorted(t for t,v in hat.items() if len(v)==11)
    if not good: return float("nan"), 0
    sh=np.stack([np.sum(hat[t],axis=0) for t in good])
    st=np.stack([np.sum(true[t],axis=0) for t in good])
    s,e=(lead-1)*24, lead*24
    ape=np.abs(st[:,s:e]-sh[:,s:e])/np.clip(st[:,s:e],1e-6,None)
    return float(ape.mean()*100), len(good)

def main():
    pl.seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_base()
    leads = pd.read_parquet(LEADFILE); leads.index = pd.to_datetime(leads.index)
    if getattr(leads.index,"tz",None) is not None: leads.index = leads.index.tz_localize(None)

    tsi = int(df.loc[df["utc"]>=TEST_START,"time_idx"].min())
    tds = build_training_ds(df)
    model = TemporalFusionTransformer.load_from_checkpoint(CKPT).to(device).eval()
    df = df[df["time_idx"] >= tsi-ENCODER_LEN-1].copy()
    zones = sorted(df["zone"].unique())

    results = json.load(open(OUT_JSON))["results"] if os.path.exists(OUT_JSON) else {}
    runs = [("BASELINE", None)] + [(f, [f]) for f in INDIVIDUAL] + list(GROUPS.items())
    print("mode=%s leads=%s stride=%d runs=%d device=%s\n"
          % (MODE, LEADS, STRIDE, len(runs), device), flush=True)

    for lead in LEADS:
        print("### building lead %d weather ###" % lead, flush=True)
        df_lead = swap_lead(df, leads, lead)
        for name, cols in runs:
            key = "lead%d::%s" % (lead, name)
            if key in results:
                print("  cached  %s" % key, flush=True); continue
            t0 = time.time()
            d_use = df_lead if cols is None else permute(df_lead, cols, SEED + abs(hash(name)) % 9999)
            mape, n = score(model, tds, d_use, tsi, device, zones, lead)
            del d_use; gc.collect()
            base = results.get("lead%d::BASELINE" % lead, {}).get("mape")
            delta = None if base is None or name=="BASELINE" else round(mape-base, 4)
            results[key] = {"lead":lead, "run":name, "mape":round(mape,4),
                            "delta":delta, "windows":n, "secs":round(time.time()-t0,1)}
            json.dump({"mode":MODE,"stride":STRIDE,
                       "checkpoint":os.path.basename(CKPT),"results":results},
                      open(OUT_JSON,"w"), indent=2)
            print("  %-32s MAPE %6.3f%%  delta %s  (%.0fs)"
                  % (name, mape, "  base" if delta is None else "%+.3f"%delta,
                     time.time()-t0), flush=True)
            if name == "BASELINE":
                print("     -> est. total for this lead: %.0f min"
                      % ((time.time()-t0)*len(runs)/60), flush=True)
        del df_lead; gc.collect()

    rows = sorted(results.values(), key=lambda r: (r["lead"], -(r["delta"] or -99)))
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print("\n===== RANKED BY ACCURACY COST =====")
    for lead in LEADS:
        print("\n--- Day %d (baseline %.3f%%) ---"
              % (lead, results["lead%d::BASELINE" % lead]["mape"]))
        for r in [x for x in rows if x["lead"]==lead and x["delta"] is not None]:
            print("  %+6.3f pts   %s" % (r["delta"], r["run"]))
    print("\nsaved:", OUT_CSV)

if __name__ == "__main__":
    main()
