# Diamond Price Prediction

This student project uses diamond details to estimate a price and explore the dataset. It includes a small Streamlit dashboard and a FastAPI service.

## What it does

- Compares Random Forest and XGBoost price models
- Estimates a diamond's price from its carat, cut, color, clarity, and measurements
- Flags unusual diamond records
- Shows charts and budget-based recommendations in the dashboard
- Answers basic diamond questions using a small local search over reference notes

## Run on Windows

The dataset and trained models are included. From the project folder, run:

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
```

Open `http://localhost:8501` for the dashboard.

To run the API in a second terminal:

```powershell
& ".\.venv\Scripts\python.exe" -m uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to try its endpoints.

If you do not have the project environment, create one and install the packages:

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

To train the models again:

```powershell
& ".\.venv\Scripts\python.exe" src/train.py
```

## Model results

The held-out test split selected XGBoost:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | $268.14 | $541.89 | 0.9814 |
| XGBoost | $261.53 | $512.52 | 0.9834 |

The data starts with 53,940 diamonds; cleaning leaves 53,774 records. Price estimates are for learning and comparison, not certified appraisals.
