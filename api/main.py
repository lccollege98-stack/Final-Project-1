from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.data_pipeline import DiamondFeatureEngineer
from src.explainability import DiamondPriceExplainer
from src.rag import JewelryKnowledgeAssistant, LLMServiceError


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "diamond_model.joblib"
ANOMALY_PATH = ROOT / "models" / "anomaly.joblib"
SHAP_BACKGROUND_PATH = ROOT / "models" / "shap_background.joblib"


class DiamondInput(BaseModel):
    carat: float = Field(gt=0)
    cut: Literal["Fair", "Good", "Very Good", "Premium", "Ideal"]
    color: Literal["D", "E", "F", "G", "H", "I", "J"]
    clarity: Literal["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
    depth: float = Field(gt=0)
    table: float = Field(gt=0)
    x: float = Field(gt=0)
    y: float = Field(gt=0)
    z: float = Field(gt=0)


class ValuationResponse(BaseModel):
    estimated_price: float
    risk_band: str
    anomaly_score: float


app = FastAPI(
    title="Diamond Valuation API",
    description="Price estimation and anomaly detection for diamond listings.",
)
assistant = JewelryKnowledgeAssistant()


@app.on_event("startup")
def load_artifacts():
    if not MODEL_PATH.exists():
        raise RuntimeError("Train the model first with: python src/train.py")
    app.state.model = joblib.load(MODEL_PATH)
    app.state.anomaly = joblib.load(ANOMALY_PATH)
    if not SHAP_BACKGROUND_PATH.exists():
        raise RuntimeError("Run src/train.py to create the SHAP background data.")
    app.state.price_explainer = DiamondPriceExplainer(
        app.state.model,
        joblib.load(SHAP_BACKGROUND_PATH),
    )


def to_frame(diamond: DiamondInput) -> pd.DataFrame:
    return pd.DataFrame([diamond.model_dump()])


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict-price")
def predict_price(diamond: DiamondInput):
    frame = to_frame(diamond)
    estimated = float(app.state.model.predict(frame)[0])
    return {"estimated_price": round(max(estimated, 0), 2)}


@app.post("/explain-price")
def explain_price(diamond: DiamondInput):
    frame = to_frame(diamond)
    estimated = float(app.state.model.predict(frame)[0])
    shap_explanations = app.state.price_explainer.explain(frame)
    contributions = shap_explanations.iloc[0]
    base_value = float(shap_explanations.attrs["base_values"][0])
    explanation = sorted(
        (
            {"feature": feature, "contribution": round(float(value), 2)}
            for feature, value in contributions.items()
        ),
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )
    return {
        "estimated_price": round(max(estimated, 0), 2),
        "base_value": round(base_value, 2),
        "explanation": explanation,
        "note": "SHAP contributions explain this model prediction; they are not causal effects.",
    }


@app.post("/valuation", response_model=ValuationResponse)
def valuation(diamond: DiamondInput, listed_price: float = Query(gt=0)):
    frame = to_frame(diamond)
    estimated = float(app.state.model.predict(frame)[0])
    difference = (listed_price - estimated) / estimated * 100
    if difference > 15:
        risk_band = "Likely overpriced"
    elif difference < -15:
        risk_band = "Likely underpriced"
    else:
        risk_band = "Near fair value"

    engineered = DiamondFeatureEngineer().fit_transform(frame)
    anomaly = app.state.anomaly
    values = engineered[anomaly["columns"]].replace([float("inf"), float("-inf")], 0).fillna(0)
    score = float(anomaly["model"].decision_function(anomaly["scaler"].transform(values))[0])
    return ValuationResponse(
        estimated_price=round(max(estimated, 0), 2),
        risk_band=risk_band,
        anomaly_score=round(score, 4),
    )


@app.post("/anomaly-detection")
def anomaly_detection(diamond: DiamondInput):
    frame = to_frame(diamond)
    engineered = DiamondFeatureEngineer().fit_transform(frame)
    artifact = app.state.anomaly
    values = engineered[artifact["columns"]].replace([float("inf"), float("-inf")], 0).fillna(0)
    prediction = int(artifact["model"].predict(artifact["scaler"].transform(values))[0])
    score = float(artifact["model"].decision_function(artifact["scaler"].transform(values))[0])
    return {"is_anomaly": prediction == -1, "anomaly_score": round(score, 4)}


@app.post("/recommend")
def recommend(diamond: DiamondInput, budget: float = Query(gt=0)):
    frame = to_frame(diamond)
    estimate = float(app.state.model.predict(frame)[0])
    if estimate > budget:
        return {
            "recommended": False,
            "estimated_price": round(max(estimate, 0), 2),
            "message": "The estimated value is above the requested budget.",
        }
    return {
        "recommended": True,
        "estimated_price": round(max(estimate, 0), 2),
        "message": "This diamond falls within the requested budget.",
    }


@app.post("/advisor")
def advisor(question: str):
    if not question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty")
    try:
        return assistant.answer(question)
    except LLMServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
