# Forecasting Energy Demand

This project builds a Streamlit dashboard for exploring energy demand forecasts and weather context for the NY region.

## Project Structure

```text
app/
  streamlit_app.py      # Streamlit dashboard
src/
  data_collection.py
  preprocessing.py
  train.py
  forecast.py
  evaluate.py
data/
  raw/                  # Raw input data
models/                 # Saved model artifacts
reports/                # Reports and outputs
requirements.txt
params.yaml
```

## Setup

Activate the virtual environment first:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install project dependencies:

```powershell
pip install -r requirements.txt
```

Check important packages:

```powershell
python -m streamlit version
python -c "import plotly.graph_objects as go; print('plotly ok')"
python -c "import mlflow; print(mlflow.__version__)"
```

## Run the Streamlit App

```powershell
streamlit run app\streamlit_app.py
```

The app expects these input files:

```text
data/raw/Region_NY.xlsx
data/raw/weather.csv
```

## MLflow

Start the MLflow UI:

```powershell
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

## Notes

- Use the activated `.venv` before running commands.
- If VS Code shows import errors, select `.venv\Scripts\python.exe` as the Python interpreter.
- `openpyxl` is required because the app reads an Excel file with `pandas.read_excel()`.
