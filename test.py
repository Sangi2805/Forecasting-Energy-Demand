import mlflow

from src.config import *

with mlflow.start_run(run_name="connection_test"):
    mlflow.log_param("test", True)
    mlflow.log_metric("dummy_metric", 1.0)
    mlflow.log_metric("dummy_metric 2", 2.0)

print("MLflow tracking URI:", mlflow.get_tracking_uri())
