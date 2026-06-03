
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
REPORT_DIR = BASE_DIR / "figures"

COL_NAMES_ELEC_DEMAND = [
    "Local time",
    "demand"
]


##########################################################################
# Project Root Directory
##########################################################################

TARGET_VARIABLE = "demand"
FEATURES = ["temperature", "humidity", "wind_speed", "hour_of_day", "day_of_week"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_NAME = "xgboost_model.pkl"
##########################################################################
# Global Variables
##########################################################################

