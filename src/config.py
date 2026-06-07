from pathlib import Path
from dotenv import load_dotenv
import os
import mlflow

##########################################################################
# Paths
##########################################################################
BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

DATA_DIR           = BASE_DIR / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR          = BASE_DIR / "models"
REPORT_DIR         = BASE_DIR / "reports"

##########################################################################
# Power.xlsx — raw column selection and rename map
##########################################################################
POWER_COLS = [
    "UTC time", "Local time", "Demand", "Demand forecast",
    "Net generation", "Total interchange",
    "NG: NUC", "NG: WAT", "NG: SUN", "NG: WND", "NG: NG",
]

POWER_RENAME = {
    "UTC time":          "utc_time",
    "Local time":        "local_time",
    "Demand":            "demand",
    "Demand forecast":   "demand_forecast",
    "Net generation":    "net_generation",
    "Total interchange": "total_interchange",
    "NG: NUC":           "ng_nuclear",
    "NG: WAT":           "ng_hydro",
    "NG: SUN":           "ng_solar",
    "NG: WND":           "ng_wind",
    "NG: NG":            "ng_natural_gas",
}

NUMERIC_POWER_COLS = [
    "demand", "demand_forecast", "net_generation", "total_interchange",
    "ng_nuclear", "ng_hydro", "ng_solar", "ng_wind", "ng_natural_gas",
]

##########################################################################
# Weather.csv — rename map (strips unit suffixes from column names)
##########################################################################
WEATHER_RENAME = {
    "temperature_2m (°C)":               "temperature_2m",
    "relative_humidity_2m (%)":          "relative_humidity_2m",
    "dew_point_2m (°C)":                 "dew_point_2m",
    "apparent_temperature (°C)":         "apparent_temperature",
    "precipitation (mm)":                "precipitation",
    "rain (mm)":                         "rain",
    "snowfall (cm)":                     "snowfall",
    "snow_depth (m)":                    "snow_depth",
    "weather_code (wmo code)":           "weather_code",
    "pressure_msl (hPa)":                "pressure_msl",
    "surface_pressure (hPa)":            "surface_pressure",
    "cloud_cover (%)":                   "cloud_cover",
    "cloud_cover_low (%)":               "cloud_cover_low",
    "cloud_cover_mid (%)":               "cloud_cover_mid",
    "cloud_cover_high (%)":              "cloud_cover_high",
    "et0_fao_evapotranspiration (mm)":   "et0_fao_evapotranspiration",
    "vapour_pressure_deficit (kPa)":     "vapour_pressure_deficit",
    "wind_speed_10m (km/h)":             "wind_speed_10m",
    "wind_speed_100m (km/h)":            "wind_speed_100m",
    "wind_direction_10m (°)":            "wind_direction_10m",
    "wind_direction_100m (°)":           "wind_direction_100m",
    "wind_gusts_10m (km/h)":             "wind_gusts_10m",
    "soil_temperature_0_to_7cm (°C)":    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm (°C)":   "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm (°C)": "soil_temperature_28_to_100cm",
    "soil_temperature_100_to_255cm (°C)":"soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm (m³/m³)":    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm (m³/m³)":   "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm (m³/m³)": "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm (m³/m³)":"soil_moisture_100_to_255cm",
}

# Features lagged by 1 and 2 days (previous-day weather signals)
LAG_WEATHER_FEATURES = [
    "temperature_2m", "apparent_temperature",
    "relative_humidity_2m", "dew_point_2m",
]

##########################################################################
# Selected dataset — columns to exclude from the curated version
##########################################################################
COLS_TO_DROP_FOR_SELECTED = [
    "pressure_msl", "surface_pressure",
    "et0_fao_evapotranspiration", "vapour_pressure_deficit",
    "wind_speed_100m", "wind_direction_10m", "wind_direction_100m",
    "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm",
    "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "demand_forecast",
]

##########################################################################
# Output dataset paths
##########################################################################
FULL_TRAIN_PATH     = PROCESSED_DATA_DIR / "features_full_train.parquet"
FULL_TEST_PATH      = PROCESSED_DATA_DIR / "features_full_test.parquet"
SELECTED_TRAIN_PATH = PROCESSED_DATA_DIR / "features_selected_train.parquet"
SELECTED_TEST_PATH  = PROCESSED_DATA_DIR / "features_selected_test.parquet"

##########################################################################
# Model / training config
##########################################################################
TARGET_VARIABLE = "demand"
TARGET_COLS     = ["target_day1", "target_day2", "target_day3"]
TRAIN_CUTOFF    = "2024-01-01"
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
MODEL_NAME      = "xgboost_model.pkl"

##########################################################################
# MLflow / DagsHub
##########################################################################
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
DAGSHUB_USER_NAME   = os.getenv("DAGSHUB_USER_NAME")
DAGSHUB_TOKEN       = os.getenv("DAGSHUB_TOKEN")

if MLFLOW_TRACKING_URI and DAGSHUB_TOKEN:
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER_NAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
