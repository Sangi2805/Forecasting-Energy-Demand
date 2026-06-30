import pandas as pd
import numpy as np
import holidays

RAW_DIR = "./data/raw"
OUT_DIR = "./data/processed"

DEMAND_FILE = f"{RAW_DIR}/Region_NY.xlsx"
WEATHER_FILE = f"{RAW_DIR}/weather.csv"
GDP_FILE = f"{RAW_DIR}/NYGDP.csv"
POP_FILE = f"{RAW_DIR}/NYPopulation.csv"

TRAIN_END = "2023-12-31 23:00:00"
VAL_END = "2025-05-31 23:00:00"


def load_demand():
    df = pd.read_excel(DEMAND_FILE, sheet_name="Published Hourly Data")
    df = df[df["Region"] == "NY"].copy()
    df = df[["UTC time", "Local time", "Demand", "Demand forecast"]]
    df = df.rename(columns={
        "UTC time": "utc_time", "Local time": "local_time",
        "Demand": "demand", "Demand forecast": "demand_forecast"
    })
    df["utc_time"] = pd.to_datetime(df["utc_time"])
    df["local_time"] = pd.to_datetime(df["local_time"])
    df = df.sort_values("utc_time").drop_duplicates("utc_time")
    df.loc[df["demand"] < 1000, "demand"] = np.nan
    df["demand"] = df["demand"].interpolate(method="linear")
    return df


def load_weather():
    df = pd.read_csv(WEATHER_FILE, skiprows=3)
    df["utc_time"] = pd.to_datetime(df["time"]) + pd.Timedelta(hours=4)
    df = df.drop(columns=["time"])
    keep = [
        "utc_time", "temperature_2m (°C)", "relative_humidity_2m (%)",
        "apparent_temperature (°C)", "precipitation (mm)", "snowfall (cm)",
        "cloud_cover (%)", "wind_speed_10m (km/h)", "wind_gusts_10m (km/h)",
        "pressure_msl (hPa)",
    ]
    df = df[keep]
    rename = {
        "temperature_2m (°C)": "temp_c",
        "relative_humidity_2m (%)": "humidity_pct",
        "apparent_temperature (°C)": "feels_like_c",
        "precipitation (mm)": "precip_mm",
        "snowfall (cm)": "snowfall_cm",
        "cloud_cover (%)": "cloud_cover_pct",
        "wind_speed_10m (km/h)": "wind_speed_kmh",
        "wind_gusts_10m (km/h)": "wind_gusts_kmh",
        "pressure_msl (hPa)": "pressure_hpa",
    }
    df = df.rename(columns=rename)
    df = df.sort_values("utc_time").drop_duplicates("utc_time")
    return df


def load_macro():
    gdp = pd.read_csv(GDP_FILE)
    gdp["year"] = pd.to_datetime(gdp["observation_date"]).dt.year
    gdp = gdp[["year", "NYNGSP"]].rename(columns={"NYNGSP": "ny_gdp"})

    pop = pd.read_csv(POP_FILE)
    pop["year"] = pd.to_datetime(pop["observation_date"]).dt.year
    pop = pop[["year", "NYPOP"]].rename(columns={"NYPOP": "ny_population"})

    macro = pd.merge(gdp, pop, on="year", how="outer").sort_values("year")
    macro = macro.set_index("year").reindex(range(macro["year"].min(), 2027)).ffill().reset_index()
    return macro


def add_calendar_features(df):
    t = df["local_time"]
    df["hour"] = t.dt.hour
    df["dayofweek"] = t.dt.dayofweek
    df["month"] = t.dt.month
    df["dayofyear"] = t.dt.dayofyear
    df["year"] = t.dt.year

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    us_holidays = holidays.UnitedStates(years=range(2014, 2027))
    df["is_holiday"] = t.dt.date.astype(str).map(lambda d: d in us_holidays).astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def add_weather_derived_features(df):
    df["temp_dev_comfort"] = (df["temp_c"] - 14).abs()
    df["feels_like_dev_comfort"] = (df["feels_like_c"] - 14).abs()
    df["snow_flag"] = (df["snowfall_cm"] > 0).astype(int)
    return df


def add_lag_rolling_features(df):
    df["demand_lag_24"] = df["demand"].shift(24)
    df["demand_lag_168"] = df["demand"].shift(168)
    df["demand_roll_mean_24"] = df["demand"].rolling(24).mean()
    df["demand_roll_std_24"] = df["demand"].rolling(24).std()
    df["demand_roll_mean_168"] = df["demand"].rolling(168).mean()
    df["demand_roll_std_168"] = df["demand"].rolling(168).std()
    return df


def add_extreme_flags(df, train_mask):
    for col, low_pct, high_pct in [("demand", 0.05, 0.95), ("temp_c", 0.05, 0.95)]:
        low = df.loc[train_mask, col].quantile(low_pct)
        high = df.loc[train_mask, col].quantile(high_pct)
        df[f"{col}_extreme_low"] = (df[col] <= low).astype(int)
        df[f"{col}_extreme_high"] = (df[col] >= high).astype(int)
    return df


def build():
    demand = load_demand()
    weather = load_weather()
    macro = load_macro()

    df = pd.merge(demand, weather, on="utc_time", how="inner")
    df = df.sort_values("utc_time").reset_index(drop=True)

    df = add_calendar_features(df)
    df = pd.merge(df, macro, on="year", how="left")

    df = add_weather_derived_features(df)
    df = add_lag_rolling_features(df)

    train_mask = df["utc_time"] <= TRAIN_END
    df = add_extreme_flags(df, train_mask)

    df["group"] = "NY"
    df["time_idx"] = np.arange(len(df))

    df["split"] = np.where(
        df["utc_time"] <= TRAIN_END, "train",
        np.where(df["utc_time"] <= VAL_END, "val", "test")
    )

    df = df.dropna().reset_index(drop=True)
    df["time_idx"] = np.arange(len(df))

    return df


if __name__ == "__main__":
    df = build()
    print(df["split"].value_counts())
    print(df.shape)
    print(df.columns.tolist())
    df.to_parquet(f"{OUT_DIR}/features.parquet", index=False)