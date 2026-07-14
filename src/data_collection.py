from datetime import date
from pathlib import Path

import pandas as pd
#from streamlit import columns
import config as cfg
import holidays as hl
import requests
import os

DEFAULT_WEATHER_VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
]

#########################################
# Read column names of  a csv file 
#########################################
def read_col_names(file_path: Path | str)-> None:
    file_path = Path(file_path)
    if not file_path.is_absolute():
        file_path = cfg.BASE_DIR / file_path

    if file_path.exists():
        columns = pd.read_csv(file_path, nrows=0).columns
        print(columns.tolist())
    else:
        print(f"File {file_path} does not exist.")
#########################################
# Read csv file
#########################################
def read_csv_file(file_path: Path | str, cols: list[str] = None) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.is_absolute():
        file_path = cfg.BASE_DIR / file_path

    if file_path.exists():
        data = pd.read_csv(file_path, usecols=cols)
        return data
    else:
        print(f"File {file_path} does not exist.")
        return None
#########################################    
# Read holidays data using the holidays library
# date (str) : yyyy-mm-dd
#########################################
def read_holiday_data(from_date: str, to_date: str) -> pd.DataFrame:
    
    begin_date = pd.Timestamp(from_date).date()
    end_date = pd.Timestamp(to_date).date()

    ny_holidays = hl.country_holidays(
        country="US",
        subdiv="NY",
        years=range(begin_date.year, end_date.year + 1)
    )

    holiday_rows = [
        {"date": date, "holiday": name}
        for date, name in ny_holidays.items()
        if begin_date <= date <= end_date
    ]

    return pd.DataFrame(holiday_rows).sort_values("date").reset_index(drop=True)
#########################################
# Read the weather information from open.meteo API
# date (str) : yyyy-mm-dd
# isforecast (bool) : if True, forecast data, otherwise  historical data
# lat (float) : latitude of the location, default is of New York City
# lon (float) : longitude of the location, default is of New York City
#########################################
def read_weather_data(
        begin_date: str, end_date: str, 
        lat: float=40.7128, lon: float=-74.0060,
        hourly_variables: list[str] | None=None,
        isforecast: bool=False,
        timeout: int=30,
        ) -> pd.DataFrame:
        if hourly_variables is None:
            hourly_variables = DEFAULT_WEATHER_VARIABLES

        if isforecast:      
            url = "https://api.open-meteo.com/v1/forecast"
        else:       
            url = "https://archive-api.open-meteo.com/v1/archive"
            

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": begin_date,
            "end_date": end_date,
            "hourly": ",".join(hourly_variables),
            "timezone": "America/New_York"
        }

        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            hourly_data = data.get("hourly", {})
            return pd.DataFrame(hourly_data)
        except requests.RequestException as exc:
            print(f"Failed to fetch weather data: {exc}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    from_date = "2015-07-01"
    to_date = "2016-07-01"
    #holiday_df = read_holiday_data(from_date, to_date)
    #print(holiday_df.head())

    #weather_df = read_weather_data(from_date, to_date, isforecast=False)
    #print(weather_df.head())

   
