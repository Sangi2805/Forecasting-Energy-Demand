import os
import mlflow
from dotenv import load_dotenv
from config import configure_mlflow_tracking

load_dotenv()

configure_mlflow_tracking()

with mlflow.start_run(run_name="I_cannot connect to DagsHub_I_need_to_test"):
    mlflow.log_param("test", True)
    mlflow.log_metric("dummy_metric", 1.0)

print("Success")