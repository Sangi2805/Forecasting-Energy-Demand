
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
    "Demand",
    "Hour"
]
ALL_WEATHER_FEATURES = [
    "time",

    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",

    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",

    "weather_code",

    "pressure_msl",
    "surface_pressure",

    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",

    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",

    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "wind_gusts_10m",

    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm",
    "soil_temperature_100_to_255cm",

    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm"
]

SEL_WEATHER_FEATURES = [
    "time",
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m"
    ]

CORRELATION_HEATMAP_COLUMNS = [
    "Demand",
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m"
]

LAG_WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature"
]

##########################################################################
# Project Root Directory
##########################################################################

TARGET_VARIABLE = "Demand"
FED_FEATURES = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "hour_of_day", "day_of_week"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_NAME = "xgboost_model.pkl"
##########################################################################
# Global Variables
##########################################################################

import os
import mlflow

def configure_mlflow_tracking():
    dagshub_user = os.getenv("DAGSHUB_USER_NAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    missing_vars = [
        name
        for name, value in {
            "DAGSHUB_USER_NAME": dagshub_user,
            "DAGSHUB_TOKEN": dagshub_token,
            "MLFLOW_TRACKING_URI": tracking_uri
        }.items()
        if not value
    ]

    if missing_vars:
        raise RuntimeError(
            "Missing DagsHub MLflow configuration: "
            f"{', '.join(missing_vars)}. "
            "Local MLflow tracking is disabled."
        )

    os.environ.setdefault("MLFLOW_TRACKING_USERNAME", dagshub_user)
    os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", dagshub_token)
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[OK] MLflow configured for DagsHub at {tracking_uri}")

    return tracking_uri

