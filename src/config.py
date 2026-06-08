
from pathlib import Path

from dotenv import load_dotenv

##########################################################################
# Project Root Directory
##########################################################################
BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

ELEC_DEMAND_FEATURES = [
    "Local time",
    "Demand"
]
ALL_WEATHER_FEATURES = [
    "time",

    "temperature_2m (°C)",
    "relative_humidity_2m (%)",
    "dew_point_2m (°C)",
    "apparent_temperature (°C)",

    "precipitation (mm)",
    "rain (mm)",
    "snowfall (cm)",
    "snow_depth (m)",

    "weather_code (wmo code)",

    "pressure_msl (hPa)",
    "surface_pressure (hPa)",

    "cloud_cover (%)",
    "cloud_cover_low (%)",
    "cloud_cover_mid (%)",
    "cloud_cover_high (%)",

    "et0_fao_evapotranspiration (mm)",
    "vapour_pressure_deficit (kPa)",

    "wind_speed_10m (km/h)",
    "wind_speed_100m (km/h)",
    "wind_direction_10m (°)",
    "wind_direction_100m (°)",
    "wind_gusts_10m (km/h)",

    "soil_temperature_0_to_7cm (°C)",
    "soil_temperature_7_to_28cm (°C)",
    "soil_temperature_28_to_100cm (°C)",
    "soil_temperature_100_to_255cm (°C)",

    "soil_moisture_0_to_7cm (m³/m³)",
    "soil_moisture_7_to_28cm (m³/m³)",
    "soil_moisture_28_to_100cm (m³/m³)",
    "soil_moisture_100_to_255cm (m³/m³)"
]

SEL_WEATHER_FEATURES = [
    "time",
    "temperature_2m (°C)",
    "apparent_temperature (°C)",
    "relative_humidity_2m (%)",
    "dew_point_2m (°C)",
    "precipitation (mm)",
    "rain (mm)",
    "snowfall (cm)",
    "cloud_cover (%)",
    "wind_speed_10m (km/h)",
    "wind_gusts_10m (km/h)",
    "weather_code (wmo code)"
]

LAG_WEATHER_FEATURES = [
    "temperature_2m (°C)",
    "relative_humidity_2m (%)",
    "dew_point_2m (°C)",
    "apparent_temperature (°C)"
]

##########################################################################
# Project Root Directory
##########################################################################

TARGET_VARIABLE = "demand"
FED_FEATURES = ["temperature_2m (°C)", "relative_humidity_2m (%)", "wind_speed_10m (km/h)", "hour_of_day", "day_of_week"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_NAME = "xgboost_model.pkl"
##########################################################################
# Global Variables
##########################################################################

import os

import dagshub
import mlflow

DAGSHUB_USER_NAME = os.getenv("DAGSHUB_USER_NAME", "Sangi2805")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "Forecasting-Energy-Demand")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"https://dagshub.com/{DAGSHUB_USER_NAME}/{DAGSHUB_REPO_NAME}.mlflow",
)

if DAGSHUB_TOKEN:
    os.environ["DAGSHUB_USER_TOKEN"] = DAGSHUB_TOKEN
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER_NAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

dagshub.init(
    repo_owner=DAGSHUB_USER_NAME,
    repo_name=DAGSHUB_REPO_NAME,
    mlflow=True,
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

