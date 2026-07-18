import pandas as pd
from streamlit import columns
import config as cfg
import numpy as np

import data_collection as dc
import os
import re

BASE_TEMP = 18.0
HOT_TEMP_THRESHOLD = 27.0


def read_elec_demand_data():
    df = dc.read_csv_file(cfg.RAW_DATA_DIR / "electricity_demand_data.csv",cols=cfg.ELEC_DEMAND_FEATURES)   # all columns are needed for electricity demand data
    df = remove_units_from_column_names(df)
    return df[cfg.ELEC_DEMAND_FEATURES]   # all columns are needed for electricity demand data
    
def read_weather_data():
    df = dc.read_csv_file(cfg.RAW_DATA_DIR / "weather_data.csv")
    df = remove_units_from_column_names(df)
    return df[cfg.SEL_WEATHER_FEATURES]   # all columns are needed for weather data
    #return dc.read_csv_file(cfg.RAW_DATA_DIR / "weather_data.csv", cols=cfg.SEL_WEATHER_FEATURES)   # all columns are needed for weather data

def read_GDP_data():
    return dc.read_csv_file(cfg.RAW_DATA_DIR / "gdp_data.csv")   # all columns are needed for GDP data

def read_population_data():
    return dc.read_csv_file(cfg.RAW_DATA_DIR / "population_data.csv")   # all columns are needed for population data  

def read_holiday_data(start_date, end_date):
    
    return dc.read_holiday_data(start_date, end_date)
# test: ok
def get_date_range_from_elec_demand(elec_demand_df):
    utc_times = pd.to_datetime(elec_demand_df["UTC time"], format="%m/%d/%Y %H:%M")
    start_date = utc_times.min().strftime("%Y-%m-%d")
    end_date = utc_times.max().strftime("%Y-%m-%d")
    return start_date, end_date

def add_date_features(df, datetime_col):
    df["date"] = df[datetime_col].dt.normalize()
    df["year"] = df[datetime_col].dt.year
    df["month"] = df[datetime_col].dt.month
    return df

def add_weather_utc_time_column(weather_df):
    weather_df = weather_df.copy()
    local_time = pd.to_datetime(
        weather_df["time"],
        format="%Y-%m-%dT%H:%M"
    )

    localized_time = local_time.dt.tz_localize(
        "America/New_York",
        ambiguous=False,
        nonexistent="shift_forward"
    )

    weather_df["UTC time"] = (
        localized_time
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    return weather_df

def add_date_time_standardized_columns(elec_demand_df, weather_df, gdp_df, population_df, holiday_df):
    elec_demand_df["UTC time"] = pd.to_datetime(
        elec_demand_df["UTC time"],
        format="%m/%d/%Y %H:%M"
    )
    weather_df = add_weather_utc_time_column(weather_df)
    gdp_df["observation_date"] = pd.to_datetime(
        gdp_df["observation_date"],
        format="%Y-%m-%d"
    )
    population_df["observation_date"] = pd.to_datetime(
        population_df["observation_date"],
        format="%Y-%m-%d"
    )
    holiday_df["date"] = pd.to_datetime(holiday_df["date"])

    #add_date_features(elec_demand_df, "UTC time")
    elec_demand_df["date"] = elec_demand_df["UTC time"].dt.normalize()
    elec_demand_df["day_name"] = elec_demand_df["date"].dt.day_name()
    elec_demand_df["Hour"] = elec_demand_df["UTC time"].dt.hour
    elec_demand_df["month"] = elec_demand_df["UTC time"].dt.month
    elec_demand_df["month_name"] = elec_demand_df["UTC time"].dt.month_name()
    elec_demand_df["year"] = elec_demand_df["UTC time"].dt.year


    #add_date_features(weather_df, "time")

    #add_date_features(gdp_df, "observation_date")
    gdp_df["year"] = gdp_df["observation_date"].dt.year
    
    #add_date_features(population_df, "observation_date")
    population_df["year"] = population_df["observation_date"].dt.year

    #add_date_features(holiday_df, "date")
    
    return elec_demand_df, weather_df, gdp_df, population_df, holiday_df

def Convert_Demand_To_Numeric(df):
    # Remove commas and convert to numeric
    df['Demand'] = pd.to_numeric(
        df['Demand'].astype(str).str.replace(',', '', regex=False)
    )
    return df

def encode_holiday_to_numeric(series):
    # Any non-empty holiday label is encoded as 1, otherwise 0.
    cleaned = series.fillna(0).astype(str).str.strip().str.lower()
    return (~cleaned.isin(["0", "", "none", "nan", "null"])).astype(int)


def add_hour_weekday_month_encoded_features(df_input):
    df_output = df_input.copy()

    hour_series = pd.to_numeric(df_output["Hour"], errors="coerce")
    month_series = pd.to_numeric(df_output["month"], errors="coerce")

    if "UTC time" in df_output.columns:
        utc_time_dt = pd.to_datetime(df_output["UTC time"], errors="coerce")
        weekday_series = utc_time_dt.dt.dayofweek
    elif "Local time" in df_output.columns:
        local_time_dt = pd.to_datetime(df_output["Local time"], errors="coerce")
        weekday_series = local_time_dt.dt.dayofweek
    elif "day_of_week" in df_output.columns:
        day_map = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6
        }
        weekday_series = (
            df_output["day_of_week"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(day_map)
        )
    else:
        raise KeyError("Missing both 'Local time' and 'day_of_week' columns.")

    df_output["hour_sin"] = np.sin(2 * np.pi * hour_series / 24.0)
    df_output["hour_cos"] = np.cos(2 * np.pi * hour_series / 24.0)

    df_output["weekday_sin"] = np.sin(2 * np.pi * weekday_series / 7.0)
    df_output["weekday_cos"] = np.cos(2 * np.pi * weekday_series / 7.0)

    df_output["month_sin"] = np.sin(2 * np.pi * (month_series - 1) / 12.0)
    df_output["month_cos"] = np.cos(2 * np.pi * (month_series - 1) / 12.0)

    return df_output

def add_previous_day_avg_weather_features(merged_df):
    daily_weather_avg_df = (
        merged_df
        .groupby("date", as_index=False)[cfg.LAG_WEATHER_FEATURES]
        .mean()
    )

    for days in [1, 2]:
        day_label = "day" if days == 1 else "days"
        previous_date_col = f"date_prev_{days}_{day_label}"
        previous_weather_df = daily_weather_avg_df.rename(
            columns={
                "date": previous_date_col,
                **{
                    feature: f"{feature}_prev_{days}_{day_label}_avg"
                    for feature in cfg.LAG_WEATHER_FEATURES
                }
            }
        )

        merged_df[previous_date_col] = merged_df["date"] - pd.Timedelta(days=days)
        merged_df = pd.merge(
            merged_df,
            previous_weather_df,
            on=previous_date_col,
            how="left"
        )
        merged_df.drop(columns=[previous_date_col], inplace=True)

    return merged_df

def remove_units_from_column_names(df):
    df = df.copy()
    df.rename(
        columns=lambda col: re.sub(r"\s*\([^)]*\)", "", col).strip(),
        inplace=True
    )
    return df

def add_model_features(df):
    df = df.copy()

    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)

    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    df["time_index"] = np.arange(len(df))

    max_time_index = df["time_index"].max()
    if max_time_index > 0:
        df["time_index"] = df["time_index"] / max_time_index
    else:
        df["time_index"] = 0.0

    df["temp_squared"] = df["temperature_2m"] ** 2
    df["cooling_degree"] = np.maximum(df["temperature_2m"] - BASE_TEMP, 0)
    df["heating_degree"] = np.maximum(BASE_TEMP - df["temperature_2m"], 0)
    df["rolling_max_temperature_24h"] = (
        df["temperature_2m"]
        .rolling(window=24, min_periods=1)
        .max()
    )
    df["rolling_mean_temperature_24h"] = (
        df["temperature_2m"]
        .rolling(window=24, min_periods=1)
        .mean()
    )
    df["temperature_anomaly"] = (
        df["temperature_2m"] - df["rolling_mean_temperature_24h"]
    )

    is_hot_hour = df["temperature_2m"] >= HOT_TEMP_THRESHOLD
    hot_spell_group = is_hot_hour.ne(is_hot_hour.shift(fill_value=False)).cumsum()
    df["consecutive_hot_hours"] = (
        is_hot_hour
        .groupby(hot_spell_group)
        .cumcount()
        .add(1)
        .where(is_hot_hour, 0)
    )

    return df

#process data

def process_data():

    #step 1: read data
    elec_demand_df = read_elec_demand_data()
    weather_df = read_weather_data()
    gdp_df = read_GDP_data()
    population_df = read_population_data()
    holiday_df = read_holiday_data( *get_date_range_from_elec_demand(elec_demand_df))

    #step 2: convert date columns to datetime format and create date features
    elec_demand_df, weather_df, gdp_df, population_df, holiday_df = add_date_time_standardized_columns(
        elec_demand_df,
        weather_df,
        gdp_df,
        population_df,
        holiday_df
    )
    # Convert 'Demand' to numeric
    elec_demand_df = Convert_Demand_To_Numeric(elec_demand_df)
    # step 4: create lag features
    elec_demand_df["demand_lag_1h"] = elec_demand_df["Demand"].shift(1)
    elec_demand_df["demand_lag_2h"] = elec_demand_df["Demand"].shift(2)

    elec_demand_df["demand_lag_3h"] = elec_demand_df["Demand"].shift(3)
    elec_demand_df["demand_lag_4h"] = elec_demand_df["Demand"].shift(4)

    elec_demand_df["demand_lag_24h"] = elec_demand_df["Demand"].shift(24)
    elec_demand_df["demand_lag_48h"] = elec_demand_df["Demand"].shift(48)
    elec_demand_df["demand_lag_72h"] = elec_demand_df["Demand"].shift(72)
    elec_demand_df["demand_lag_168h"] = elec_demand_df["Demand"].shift(168)

    elec_demand_df["demand_rolling_24h_mean"] = elec_demand_df["Demand"].shift(1).rolling(24).mean()
    elec_demand_df["demand_rolling_48h_mean"] = elec_demand_df["Demand"].shift(1).rolling(48).mean()  
    elec_demand_df["demand_rolling_72h_mean"] = elec_demand_df["Demand"].shift(1).rolling(72).mean()  
    elec_demand_df["demand_rolling_168h_mean"] = elec_demand_df["Demand"].shift(1).rolling(168).mean()  

    historical_demand = elec_demand_df["Demand"].shift(1)

    # Std
    elec_demand_df["demand_std_24h"] = historical_demand.rolling(24).std()
    elec_demand_df["demand_std_48h"] = historical_demand.rolling(48).std()
    elec_demand_df["demand_std_72h"] = historical_demand.rolling(72).std()
    elec_demand_df["demand_std_168h"] = historical_demand.rolling(168).std()

    # Min
    elec_demand_df["demand_min_24h"] = historical_demand.rolling(24).min()
    elec_demand_df["demand_min_48h"] = historical_demand.rolling(48).min()
    elec_demand_df["demand_min_72h"] = historical_demand.rolling(72).min()
    elec_demand_df["demand_min_168h"] = historical_demand.rolling(168).min()

    # Max
    elec_demand_df["demand_max_24h"] = historical_demand.rolling(24).max()
    elec_demand_df["demand_max_48h"] = historical_demand.rolling(48).max()
    elec_demand_df["demand_max_72h"] = historical_demand.rolling(72).max()
    elec_demand_df["demand_max_168h"] = historical_demand.rolling(168).max()


    ########################
    # Encode weekday to numeric
    merged_df = pd.merge(elec_demand_df, holiday_df, left_on='date', right_on='date', how='left')


    merged_df = add_hour_weekday_month_encoded_features(merged_df)
    
    merged_df["holiday_encoded"] = encode_holiday_to_numeric(merged_df["holiday"])

    merged_df = pd.merge(merged_df, weather_df, on='UTC time', how='left')
    merged_df = pd.merge(merged_df, gdp_df, left_on='year', right_on='year', how='left')
    merged_df = pd.merge(merged_df, population_df, left_on='year', right_on='year', how='left')
       

    #merged_df = add_previous_day_avg_weather_features(merged_df)

    merged_df = remove_units_from_column_names(merged_df)
    
    # Convert 'Demand' to numeric
    merged_df = Convert_Demand_To_Numeric(merged_df)

    #drop unnecessary columns
    merged_df.drop(columns=["time", "observation_date_x", "observation_date_y"], inplace=True)

    


    # step 5 zero imputation for missing values in merged_df
    merged_df.fillna(0, inplace=True)


    # step 6: drop duplicates based on 'UTC time'
    merged_df.drop_duplicates(subset='UTC time', inplace=True)  

    # step 7: Cut the dataset to the desired date range 
    # (2016-04-30 to 2026-04-30) to match the electricity demand data range
    # to avoid any potential issues with demand_lag features =0 and 
    # data is of 10 years.
    merged_df = merged_df[(merged_df['date'] >= '2016-04-30') & (merged_df['date'] <= '2026-04-30')]    

    # rename column "UTC time" to "datetime" for consistency
    merged_df.rename(columns={"UTC time": "datetime"}, inplace=True)  

    merged_df = add_model_features(merged_df)

    return merged_df



def test():
    #dc.read_col_names(cfg.RAW_DATA_DIR / "weather_data.csv")
    # export df to csv file with 5 head and 5 tail rows
    dataset_df=process_data()
    
    #dataset_df.head(5).to_csv(cfg.PROCESSED_DATA_DIR / "head_rows.csv", index=False)
    #dataset_df.tail(5).to_csv(cfg.PROCESSED_DATA_DIR / "tail_rows.csv", index=False)
    #test_df = pd.concat([dataset_df.head(5), dataset_df.tail(5)], ignore_index=True)
    #test_df.to_csv(cfg.REPORT_DIR / "test_data.csv", index=False)
    
    #print(dataset_df.isna().sum())
    dataset_df.to_csv(cfg.REPORT_DIR / "all_features_dataset.csv", index=False)

if __name__ == "__main__":

    test()
