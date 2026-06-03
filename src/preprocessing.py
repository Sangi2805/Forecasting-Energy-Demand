import pandas as pd
from streamlit import columns
import config as cfg
import os

def read_col_names_raw(raw_file_name):
    file_path = cfg.RAW_DATA_DIR / raw_file_name

    columns = pd.read_csv(file_path, nrows=0).columns
    print(columns.tolist())

def read_csv_file(file_name, cols=None):
    file_path = cfg.RAW_DATA_DIR / file_name
    if file_path.exists():
        data = pd.read_csv(file_path, usecols=cols)
        return data
    else:
        print(f"File {file_name} does not exist.")
        return None

def read_elec_demand_data():
    return read_csv_file("electricity_demand.csv", cols=cfg.COL_NAMES_ELEC_DEMAND)

def read_weather_data():
    return read_csv_file("weather_data.csv")   # all columns are needed for weather data

def read_GDP_data():
    return read_csv_file("gdp_data.csv")   # all columns are needed for GDP data

def read_population_data():
    return read_csv_file("population_data.csv")   # all columns are needed for population data  

def read_holiday_data():
    
    return read_csv_file("holiday_data.csv")   # all columns are needed for holiday data

#process data
# mapping to day of week
# mapping hoilday if possible
# After selecting 10 important features, 
# => create a new feature that is the temperature of previous day
# => create a new feature that is the temperature of two revious day

def process_data(elec_demand_df, weather_df, gdp_df, population_df, holiday_df):
    # Merge datasets on 'Local time' and 'date' columns
    merged_df = pd.merge(elec_demand_df, weather_df, left_on='Local time', right_on='date', how='left')
    merged_df = pd.merge(merged_df, gdp_df, left_on='Local time', right_on='date', how='left')
    merged_df = pd.merge(merged_df, population_df, left_on='Local time', right_on='date', how='left')
    merged_df = pd.merge(merged_df, holiday_df, left_on='Local time', right_on='date', how='left')

    # Drop redundant columns
    merged_df.drop(columns=['date_x', 'date_y', 'date'], inplace=True)

    # Handle missing values (if any)
    merged_df.fillna(method='ffill', inplace=True)

    return merged_df



#if __name__ == "__main__":
