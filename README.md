# Diamond Price Prediction

This project uses the diamonds dataset to study diamond prices. I used Python
to clean the data, make charts, compare models, and build a small dashboard and API.

## Project work

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

## Running the project

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

## LLM setup

The advisor normally answers from `data/diamond_knowledge.json`. An
OpenAI-compatible API can be used to generate answers from these retrieved notes.
Set the following values in the terminal:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
$env:OPENAI_MODEL = "gpt-4o-mini"
```

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

## Model results

- Random Forest: MAE $268.14, RMSE $541.89, R² 0.9814
- XGBoost: MAE $261.53, RMSE $512.52, R² 0.9834
