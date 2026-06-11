import pandas as pd
import numpy as np
import statsmodels.api as sm
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USER_NAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")


# ==========================================
# 1. DAGSHUB & MLFLOW SETUP
# ==========================================
# Configured with your specific repository details
mlflow.set_tracking_uri("https://dagshub.com/Sangi2805/Forecasting-Energy-Demand.mlflow")

# ==========================================
# 2. DATA LOADING & PREPARATION
# ==========================================
print("Loading datasets...")
train = pd.read_csv('data/processed/features_selected_train.csv', parse_dates=['date'], index_col='date')
test = pd.read_csv('data/processed/features_selected_test.csv', parse_dates=['date'], index_col='date')

# Select exogenous features (structurally lagged to prevent data leakage over a 3-day horizon)
exog_features = [
    'apparent_temperature',
    'wind_speed_10m',
    'demand_roll_mean_7d',
    'demand_roll_std_7d',
    'demand_lag_3d',
    'demand_lag_14d',
    'is_weekend',
    'is_holiday',
    'day_of_week',
]

# Drop rows with NaNs caused by the rolling windows and lags
train = train.dropna(subset=exog_features + ['demand'])
test = test.dropna(subset=exog_features + ['demand'])

train.index.freq = 'D'
test.index.freq  = 'D'

y_train = train['demand']
X_train = train[exog_features]
y_test = test['demand']
X_test = test[exog_features]

# ==========================================
# 3. MODEL CONFIGURATION
# ==========================================
# SARIMAX Order: (p, d, q) and Seasonal Order: (P, D, Q, s)
# Weekly seasonality for energy demand means s=7
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 7)

# ==========================================
# 4. TRAINING & EXPERIMENT TRACKING
# ==========================================
print("Connecting to MLflow tracking server...")
with mlflow.start_run(run_name="sarimax_baseline"):
    
    # Log your configuration parameters
    mlflow.log_params({
        "model_type": "SARIMAX",
        "order": str(ORDER),
        "seasonal_order": str(SEASONAL_ORDER),
        "features": exog_features,
        "train_size": len(train)
    })
    
    print("Training SARIMAX model...")
    # Initialize and fit the SARIMAX model
    model = sm.tsa.statespace.SARIMAX(
        endog=y_train,
        exog=X_train,
        order=ORDER, 
        seasonal_order=SEASONAL_ORDER, 
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    mlflow.statsmodels.log_model(results, "sarimax_model")

    print("Forecasting next 3 days...")
    # Forecast the next 3 days using the exogenous inputs from the test set
    forecast = results.forecast(steps=3, exog=X_test.iloc[:3])
    
    # Get actuals for those 3 days
    actuals = y_test.iloc[:3]
    
    # Per-day metrics
    for i in range(3):
        day = i + 1
        a = actuals.iloc[i]
        p = forecast.iloc[i]
        mlflow.log_metric(f"day{day}_mae",  abs(a - p))
        mlflow.log_metric(f"day{day}_rmse", np.sqrt((a - p) ** 2))
        mlflow.log_metric(f"day{day}_mape", abs((a - p) / a) * 100)

    # Average metrics
    avg_mae  = mean_absolute_error(actuals, forecast)
    avg_rmse = root_mean_squared_error(actuals, forecast)
    avg_mape = np.mean(np.abs((actuals.values - forecast.values) / actuals.values)) * 100

    mlflow.log_metric("avg_mae",  avg_mae)
    mlflow.log_metric("avg_rmse", avg_rmse)
    mlflow.log_metric("avg_mape", avg_mape)

    print(f"3-Day Forecast RMSE: {avg_rmse:.2f}")
    print(f"3-Day Forecast MAE:  {avg_mae:.2f}")
    print(f"3-Day Forecast MAPE: {avg_mape:.2f}%")
    
    print("Saving artifacts...")
    # Save your forecast actuals vs predictions to a CSV and upload it as an artifact
    forecast_df = pd.DataFrame({'actual': actuals, 'predicted': forecast})
    forecast_df.to_csv("forecast_results.csv")
    mlflow.log_artifact("forecast_results.csv")
    
    print("Run successfully completed and logged to DagsHub!")