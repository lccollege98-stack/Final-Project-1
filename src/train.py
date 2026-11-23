from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_pipeline import DiamondFeatureEngineer, clean_diamonds


DATA_PATH = ROOT / "data" / "diamonds.csv"
MODEL_DIR = ROOT / "models"


def build_pipeline(model):
    numerical = [
        "carat", "depth", "table", "x", "y", "z",
        "volume", "surface_area_proxy", "cut_score",
        "color_score", "clarity_score", "premium_score",
        "carat_volume_ratio",
    ]
    categorical = ["cut", "color", "clarity"]
    preprocessing = ColumnTransformer(
        [
            ("num", RobustScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline(
        [
            ("features", DiamondFeatureEngineer()),
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )


def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "name": name,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "model": model,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download diamonds.csv into data/."
        )

    raw = pd.read_csv(DATA_PATH)
    df = clean_diamonds(raw)
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "random_forest": build_pipeline(
            RandomForestRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=42)
        ),
        "xgboost": build_pipeline(
            XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )
        ),
    }

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("diamond-valuation")
    results = []
    with mlflow.start_run(run_name="model_comparison"):
        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                result = evaluate(name, model, X_train, y_train, X_test, y_test)
                mlflow.log_metrics(
                    {"mae": result["mae"], "rmse": result["rmse"], "r2": result["r2"]}
                )
                results.append(result)

    best = min(results, key=lambda item: item["rmse"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], MODEL_DIR / "diamond_model.joblib")

    engineered = DiamondFeatureEngineer().fit_transform(df)
    anomaly_columns = ["carat", "depth", "table", "x", "y", "z", "premium_score"]
    anomaly_scaler_data = engineered[anomaly_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = RobustScaler()
    scaled = scaler.fit_transform(anomaly_scaler_data)
    anomaly_model = IsolationForest(
        n_estimators=300,
        contamination=0.02,
        random_state=42,
        n_jobs=-1,
    )
    anomaly_model.fit(scaled)
    joblib.dump(
        {"model": anomaly_model, "scaler": scaler, "columns": anomaly_columns},
        MODEL_DIR / "anomaly.joblib",
    )

    segmentation_columns = ["carat", "cut_score", "color_score", "clarity_score", "price"]
    segmentation_scaler = RobustScaler()
    segmentation_values = segmentation_scaler.fit_transform(engineered[segmentation_columns])
    segmentation_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    segmentation_model.fit(segmentation_values)
    joblib.dump(
        {
            "model": segmentation_model,
            "scaler": segmentation_scaler,
            "columns": segmentation_columns,
        },
        MODEL_DIR / "segmentation.joblib",
    )

    explainability = permutation_importance(
        best["model"],
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )
    importance = sorted(
        (
            {"feature": str(feature), "importance": float(value)}
            for feature, value in zip(
                X_test.columns,
                explainability.importances_mean,
            )
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )

    metadata = {
        "selected_model": best["name"],
        "metrics": {
            item["name"]: {key: item[key] for key in ("mae", "rmse", "r2")}
            for item in results
        },
        "row_count": len(df),
        "feature_columns": list(X.columns),
        "feature_importance": importance,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
