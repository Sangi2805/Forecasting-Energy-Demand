import numpy as np
import pandas as pd
import config as cfg
import data_collection as dc


##########################################################################
# Read raw data
##########################################################################

def read_elec_demand_data() -> pd.DataFrame:
    df = dc.read_excel_file(cfg.RAW_DATA_DIR / "Power.xlsx", cols=cfg.POWER_COLS)
    df = df.rename(columns=cfg.POWER_RENAME)

    for col in cfg.NUMERIC_POWER_COLS:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )

    # Drop last rows where demand is NaN (forecast-only rows with no actuals)
    df = df[df["demand"].notna()].copy()

    # Generation column NaN = structural zero (generation didn't exist yet)
    gen_cols = ["ng_nuclear", "ng_hydro", "ng_solar", "ng_wind", "ng_natural_gas"]
    df[gen_cols] = df[gen_cols].fillna(0)

    return df


def read_weather_data() -> pd.DataFrame:
    # skiprows=3: skip metadata row, values row, and blank row before header
    df = pd.read_csv(
        cfg.RAW_DATA_DIR / "Weather.csv",
        skiprows=3,
        header=0,
    )
    df = df.rename(columns=cfg.WEATHER_RENAME)
    return df


def read_gdp_data() -> pd.DataFrame:
    return dc.read_csv_file(cfg.RAW_DATA_DIR / "NYNGSP.csv")


def read_population_data() -> pd.DataFrame:
    return dc.read_csv_file(cfg.RAW_DATA_DIR / "Population.csv")


def read_holiday_data(start_date: str, end_date: str) -> pd.DataFrame:
    return dc.read_holiday_data(start_date, end_date)


##########################################################################
# Hourly → daily aggregation
##########################################################################

def aggregate_power_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    # Parse UTC time and convert to America/New_York to get unambiguous local date.
    # Using utc_time avoids the DST fall-back hour (25h day) and spring-forward
    # gap (23h day) that naive local_time parsing produces.
    df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True)
    df["date"] = pd.to_datetime(
        df["utc_time"].dt.tz_convert("America/New_York").dt.date
    )

    daily = df.groupby("date").agg(
        demand          =("demand",           "sum"),
        demand_forecast =("demand_forecast",  "sum"),
        net_generation  =("net_generation",   "sum"),
        total_interchange=("total_interchange","sum"),
        ng_nuclear      =("ng_nuclear",       "mean"),
        ng_hydro        =("ng_hydro",         "mean"),
        ng_solar        =("ng_solar",         "mean"),
        ng_wind         =("ng_wind",          "mean"),
        ng_natural_gas  =("ng_natural_gas",   "mean"),
    ).reset_index()

    return daily


def aggregate_weather_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    # Weather.csv timestamps are America/New_York local (naive). Localize properly:
    #   - Fall-back: two identical "01:00" rows exist. Mark first=DST, second=non-DST
    #     via boolean ambiguous array (~duplicated keeps first=True, rest=False).
    #   - Spring-forward: "02:00" doesn't exist; shift_forward maps it to 03:00.
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%dT%H:%M")
    ambiguous = ~df["time"].duplicated(keep="first")  # first occurrence → DST (True)
    df["time"] = df["time"].dt.tz_localize(
        "America/New_York", ambiguous=ambiguous, nonexistent="shift_forward"
    )
    df["date"] = pd.to_datetime(df["time"].dt.date)

    agg_dict = {
        "temperature_2m":       [("temperature_2m", "mean"), ("temp_max", "max"), ("temp_min", "min")],
        "apparent_temperature": [("apparent_temperature", "mean")],
        "relative_humidity_2m": [("relative_humidity_2m", "mean")],
        "dew_point_2m":         [("dew_point_2m", "mean")],
        "precipitation":        [("precipitation", "sum")],
        "rain":                 [("rain", "sum")],
        "snowfall":             [("snowfall", "sum")],
        "snow_depth":           [("snow_depth", "max")],
        "cloud_cover":          [("cloud_cover", "mean")],
        "cloud_cover_low":      [("cloud_cover_low", "mean")],
        "cloud_cover_mid":      [("cloud_cover_mid", "mean")],
        "cloud_cover_high":     [("cloud_cover_high", "mean")],
        "wind_speed_10m":       [("wind_speed_10m", "mean")],
        "wind_speed_100m":      [("wind_speed_100m", "mean")],
        "wind_direction_10m":   [("wind_direction_10m", "mean")],
        "wind_direction_100m":  [("wind_direction_100m", "mean")],
        "wind_gusts_10m":       [("wind_gusts_10m", "max")],
        "weather_code":         [("weather_code", "max")],
        "pressure_msl":                    [("pressure_msl", "mean")],
        "surface_pressure":                [("surface_pressure", "mean")],
        "et0_fao_evapotranspiration":      [("et0_fao_evapotranspiration", "sum")],
        "vapour_pressure_deficit":         [("vapour_pressure_deficit", "mean")],
        "soil_temperature_0_to_7cm":       [("soil_temperature_0_to_7cm", "mean")],
        "soil_temperature_7_to_28cm":      [("soil_temperature_7_to_28cm", "mean")],
        "soil_temperature_28_to_100cm":    [("soil_temperature_28_to_100cm", "mean")],
        "soil_temperature_100_to_255cm":   [("soil_temperature_100_to_255cm", "mean")],
        "soil_moisture_0_to_7cm":          [("soil_moisture_0_to_7cm", "mean")],
        "soil_moisture_7_to_28cm":         [("soil_moisture_7_to_28cm", "mean")],
        "soil_moisture_28_to_100cm":       [("soil_moisture_28_to_100cm", "mean")],
        "soil_moisture_100_to_255cm":      [("soil_moisture_100_to_255cm", "mean")],
    }

    frames = []
    for src_col, named_aggs in agg_dict.items():
        if src_col not in df.columns:
            continue
        for out_col, func in named_aggs:
            s = df.groupby("date")[src_col].agg(func).rename(out_col)
            frames.append(s)

    daily = pd.concat(frames, axis=1).reset_index()
    return daily


##########################################################################
# Gap audit — must run before any lag/shift computation
##########################################################################

def audit_and_fix_gaps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)

    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_range).rename_axis("date").reset_index()

    missing = df[df["demand"].isna()]["date"]
    if len(missing) == 0:
        print("Gap audit: no missing days found.")
    else:
        print(f"Gap audit: {len(missing)} missing day(s):")
        # Identify consecutive runs
        gaps = []
        run = [missing.iloc[0]]
        for d in missing.iloc[1:]:
            if (d - run[-1]).days == 1:
                run.append(d)
            else:
                gaps.append(run)
                run = [d]
        gaps.append(run)

        for run in gaps:
            print(f"  {run[0].date()} → {run[-1].date()} ({len(run)} day(s))")
            if len(run) <= 2:
                # Interpolate short gaps linearly
                df = df.set_index("date")
                df = df.interpolate(method="time")
                df = df.reset_index()
                print(f"    → interpolated.")
            else:
                print(f"    → gap > 2 days: rows retained as NaN, will be dropped with lag NaNs.")

    return df


##########################################################################
# Merge all sources
##########################################################################

def merge_all_sources(
    daily_power: pd.DataFrame,
    daily_weather: pd.DataFrame,
    gdp_df: pd.DataFrame,
    pop_df: pd.DataFrame,
    holiday_df: pd.DataFrame,
) -> pd.DataFrame:

    df = daily_power.merge(daily_weather, on="date", how="left")

    # Parse annual data
    gdp_df["year"] = pd.to_datetime(gdp_df["observation_date"]).dt.year
    pop_df["year"] = pd.to_datetime(pop_df["observation_date"], dayfirst=False).dt.year

    df["year"] = pd.to_datetime(df["date"]).dt.year

    # Publication-lag fix: at year Y, use year Y-1 published data
    gdp_df["join_year"] = gdp_df["year"] + 1
    pop_df["join_year"]  = pop_df["year"]  + 1

    df = df.merge(
        gdp_df[["join_year", "NYNGSP"]].rename(columns={"join_year": "year"}),
        on="year", how="left"
    )
    df = df.merge(
        pop_df[["join_year", "NYPOP"]].rename(columns={"join_year": "year"}),
        on="year", how="left"
    )

    # Forward-fill to cover 2026 (no GSP/Population data yet)
    df[["NYNGSP", "NYPOP"]] = df[["NYNGSP", "NYPOP"]].ffill()

    # Merge holidays and convert to binary flag
    holiday_df["date"] = pd.to_datetime(holiday_df["date"])
    df = df.merge(holiday_df, on="date", how="left")
    df["is_holiday"] = df["holiday"].notna().astype(int)
    df.drop(columns=["holiday"], inplace=True)

    return df


##########################################################################
# Feature engineering
##########################################################################

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["date"])
    df["year"]         = dt.dt.year
    df["month"]        = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["day_of_week"]  = dt.dt.dayofweek        # 0=Monday integer
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["quarter"]      = dt.dt.quarter
    df["is_weekend"]   = (dt.dt.dayofweek >= 5).astype(int)
    return df


def add_demand_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in [3, 4, 5, 6, 7, 14, 21, 28]:
        df[f"demand_lag_{lag}d"] = df["demand"].shift(lag)
    return df


def add_rolling_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    # Shift by 3 first so rolling window never touches future demand
    d = df["demand"].shift(3)
    df["demand_roll_mean_7d"]  = d.rolling(7,  min_periods=4).mean()
    df["demand_roll_std_7d"]   = d.rolling(7,  min_periods=4).std()
    df["demand_roll_mean_30d"] = d.rolling(30, min_periods=15).mean()
    df["demand_roll_min_30d"]  = d.rolling(30, min_periods=15).min()
    df["demand_roll_max_30d"]  = d.rolling(30, min_periods=15).max()
    return df


def add_lagged_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in cfg.LAG_WEATHER_FEATURES:
        if col not in df.columns:
            continue
        df[f"{col}_lag_1d"] = df[col].shift(1)
        df[f"{col}_lag_2d"] = df[col].shift(2)
    return df


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    for h in [1, 2, 3]:
        df[f"target_day{h}"] = df["demand"].shift(-h)
    return df


##########################################################################
# Output datasets
##########################################################################

def create_output_datasets(df: pd.DataFrame) -> None:
    df_full     = df.copy()
    df_selected = df.drop(
        columns=[c for c in cfg.COLS_TO_DROP_FOR_SELECTED if c in df.columns]
    )

    train_full     = df_full[df_full["date"] < cfg.TRAIN_CUTOFF]
    test_full      = df_full[df_full["date"] >= cfg.TRAIN_CUTOFF]
    train_selected = df_selected[df_selected["date"] < cfg.TRAIN_CUTOFF]
    test_selected  = df_selected[df_selected["date"] >= cfg.TRAIN_CUTOFF]

    train_full.to_parquet(cfg.FULL_TRAIN_PATH,     index=False)
    test_full.to_parquet(cfg.FULL_TEST_PATH,       index=False)
    train_selected.to_parquet(cfg.SELECTED_TRAIN_PATH, index=False)
    test_selected.to_parquet(cfg.SELECTED_TEST_PATH,   index=False)

    train_full.to_csv(cfg.PROCESSED_DATA_DIR / "features_full_train.csv",         index=False)
    test_full.to_csv(cfg.PROCESSED_DATA_DIR / "features_full_test.csv",           index=False)
    train_selected.to_csv(cfg.PROCESSED_DATA_DIR / "features_selected_train.csv", index=False)
    test_selected.to_csv(cfg.PROCESSED_DATA_DIR / "features_selected_test.csv",   index=False)

    print(f"Saved: features_full_train     {train_full.shape}")
    print(f"Saved: features_full_test      {test_full.shape}")
    print(f"Saved: features_selected_train {train_selected.shape}")
    print(f"Saved: features_selected_test  {test_selected.shape}")


##########################################################################
# Main pipeline
##########################################################################

def process_data() -> pd.DataFrame:

    # 1. Read raw hourly data
    print("Reading raw data...")
    power_df   = read_elec_demand_data()
    weather_df = read_weather_data()
    gdp_df     = read_gdp_data()
    pop_df     = read_population_data()

    # 2. Aggregate hourly → daily
    print("Aggregating to daily...")
    daily_power   = aggregate_power_to_daily(power_df)
    daily_weather = aggregate_weather_to_daily(weather_df)

    # 3. Read holidays using date range from power data
    start_date = str(daily_power["date"].min().date())
    end_date   = str(daily_power["date"].max().date())
    holiday_df = read_holiday_data(start_date, end_date)

    # 4. Merge all sources
    print("Merging sources...")
    df = merge_all_sources(daily_power, daily_weather, gdp_df, pop_df, holiday_df)

    # 5. Gap audit — reindex to full date range before any lags
    print("Auditing gaps...")
    df = audit_and_fix_gaps(df)

    # 6. Temporal features (raw integers, no cyclical encoding)
    df = add_date_features(df)

    # 7. Demand lags and rolling stats (all shift-based, order matters)
    df = add_demand_lag_features(df)
    df = add_rolling_demand_features(df)

    # 8. Lagged weather features
    df = add_lagged_weather_features(df)

    # 9. Target columns
    df = add_target_columns(df)

    # 10. Drop warmup rows (28-day lag) and tail rows (3-day target)
    df = df.iloc[28:].copy()
    df = df.dropna(subset=cfg.TARGET_COLS)

    # 11. Drop any rows with NaN in lag features (from gap blocks)
    lag_cols = [c for c in df.columns if c.startswith("demand_lag_") or c.startswith("demand_roll_")]
    df = df.dropna(subset=lag_cols)

    print(f"Final dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"NaN count: {df.isna().sum().sum()}")

    # 12. Save train/test splits
    create_output_datasets(df)

    return df


if __name__ == "__main__":
    process_data()
