import pandas as pd
from streamlit import columns
import config as cfg
import data_collection as dc
import os


def read_elec_demand_data():
    return dc.read_csv_file(cfg.RAW_DATA_DIR / "electricity_demand_data.csv", cols=cfg.ELEC_DEMAND_FEATURES)   # all columns are needed for electricity demand data
    
def read_weather_data():
    return dc.read_csv_file(cfg.RAW_DATA_DIR / "weather_data.csv", cols=cfg.SEL_WEATHER_FEATURES)   # all columns are needed for weather data

def read_GDP_data():
    return dc.read_csv_file(cfg.RAW_DATA_DIR / "gdp_data.csv")   # all columns are needed for GDP data

def read_population_data():
    return dc.read_csv_file(cfg.RAW_DATA_DIR / "population_data.csv")   # all columns are needed for population data  

def read_holiday_data(start_date, end_date):
    
    return dc.read_holiday_data(start_date, end_date)
# test: ok
def get_date_range_from_elec_demand(elec_demand_df):
    local_times = pd.to_datetime(elec_demand_df["Local time"], format="%m/%d/%Y %H:%M")
    start_date = local_times.min().strftime("%Y-%m-%d")
    end_date = local_times.max().strftime("%Y-%m-%d")
    return start_date, end_date

def add_date_features(df, datetime_col):
    df["date"] = df[datetime_col].dt.normalize()
    df["year"] = df[datetime_col].dt.year
    df["month"] = df[datetime_col].dt.month
    return df

def standardize_date_time_columns(elec_demand_df, weather_df, gdp_df, population_df, holiday_df):
    elec_demand_df["Local time"] = pd.to_datetime(
        elec_demand_df["Local time"],
        format="%m/%d/%Y %H:%M"
    )
    weather_df["time"] = pd.to_datetime(
        weather_df["time"],
        format="%Y-%m-%dT%H:%M"
    )
    gdp_df["observation_date"] = pd.to_datetime(
        gdp_df["observation_date"],
        format="%Y-%m-%d"
    )
    population_df["observation_date"] = pd.to_datetime(
        population_df["observation_date"],
        format="%Y-%m-%d"
    )
    holiday_df["date"] = pd.to_datetime(holiday_df["date"])

    add_date_features(elec_demand_df, "Local time")
    add_date_features(weather_df, "time")
    add_date_features(gdp_df, "observation_date")
    add_date_features(population_df, "observation_date")

    holiday_df["year"] = holiday_df["date"].dt.year
    holiday_df["month"] = holiday_df["date"].dt.month

    return elec_demand_df, weather_df, gdp_df, population_df, holiday_df

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

#process data

def process_data():

    #step 1: read data
    elec_demand_df = read_elec_demand_data()
    weather_df = read_weather_data()
    gdp_df = read_GDP_data()
    population_df = read_population_data()
    holiday_df = read_holiday_data( *get_date_range_from_elec_demand(elec_demand_df))

    #step 2: convert date columns to datetime format and create date features
    elec_demand_df, weather_df, gdp_df, population_df, holiday_df = standardize_date_time_columns(
        elec_demand_df,
        weather_df,
        gdp_df,
        population_df,
        holiday_df
    )
   
    # step 3: Merge datasets 

    weather_merge_df = weather_df.drop(columns=["date", "year", "month"])
    gdp_merge_df = gdp_df.drop(columns=["observation_date", "date", "month"])
    population_merge_df = population_df.drop(columns=["observation_date", "date", "month"])
    holiday_merge_df = holiday_df.drop(columns=["year", "month"])

    merged_df = pd.merge(elec_demand_df, weather_merge_df, left_on='Local time', right_on='time', how='left')
    merged_df = pd.merge(merged_df, gdp_merge_df, left_on='year', right_on='year', how='left')
    merged_df = pd.merge(merged_df, population_merge_df, left_on='year', right_on='year', how='left')
    merged_df = pd.merge(merged_df, holiday_merge_df, left_on='date', right_on='date', how='left')
    
    merged_df["day_of_week"] = merged_df["date"].dt.day_name()
    merged_df = add_previous_day_avg_weather_features(merged_df)
    
    # step 4: zero imputation for missing values in merged_df
    merged_df.fillna(0, inplace=True)
      
   

    return merged_df

def test():
    #dc.read_col_names(cfg.RAW_DATA_DIR / "weather_data.csv")
    # export df to csv file with 5 head and 5 tail rows
    dataset_df=process_data()
    
    #dataset_df.head(5).to_csv(cfg.PROCESSED_DATA_DIR / "head_rows.csv", index=False)
    #dataset_df.tail(5).to_csv(cfg.PROCESSED_DATA_DIR / "tail_rows.csv", index=False)
    #test_df = pd.concat([dataset_df.head(5), dataset_df.tail(5)], ignore_index=True)
    #test_df.to_csv(cfg.REPORT_DIR / "test_data.csv", index=False)
    
    print(dataset_df.isna().sum())
    dataset_df.to_csv(cfg.REPORT_DIR / "all_features_dataset.csv", index=False)

if __name__ == "__main__":

    test()
