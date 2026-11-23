# Diamond Price Prediction Project

This is a Python project for analysing diamond prices. It uses the Kaggle diamonds
dataset and includes data cleaning, charts, price prediction, anomaly detection,
diamond recommendations and a simple chat interface.

## Features

- Random Forest and XGBoost price models
- Price estimation from diamond attributes
- Fair-price and under/over-price comparison
- Isolation Forest anomaly detection
- KMeans market segmentation
- EDA charts and summary report
- Permutation-based feature importance
- RAG-style diamond advisor
- Streamlit dashboard
- FastAPI endpoints

## How to run

The dashboard can be started from the project folder:

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
```

Open `http://localhost:8501`.

In another terminal, start the API:

```powershell
& ".\.venv\Scripts\python.exe" -m uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Optional LLM setup

The advisor normally answers from `data/diamond_knowledge.json`. An
OpenAI-compatible API can be used to generate answers from these retrieved notes.
Set the following values in the terminal:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
$env:OPENAI_MODEL = "gpt-4o-mini"
```

Do not put an API key in the project files or GitHub.

If the virtual environment is not available:

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Train the models again with:

```powershell
& ".\.venv\Scripts\python.exe" src/train.py
```

Create the EDA report with:

```powershell
& ".\.venv\Scripts\python.exe" src/eda_report.py
```

## Results

The test data was used to compare the models:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | $268.14 | $541.89 | 0.9814 |
| XGBoost | $261.53 | $512.52 | 0.9834 |

The original dataset contains 53,940 rows. After cleaning, 53,774 rows remain.
The price estimates are study outputs and not certified valuations.
