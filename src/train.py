from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_pipeline import DiamondFeatureEngineer, clean_diamonds
from src.explainability import DiamondPriceExplainer


DATA_PATH = ROOT / "data" / "diamonds.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


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


def cross_validate_model(model, X_train, y_train):
    folds = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=folds,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=1,
        error_score="raise",
    )
    return {
        "mae_mean": float(-scores["test_mae"].mean()),
        "mae_std": float(scores["test_mae"].std()),
        "rmse_mean": float(-scores["test_rmse"].mean()),
        "rmse_std": float(scores["test_rmse"].std()),
        "r2_mean": float(scores["test_r2"].mean()),
        "r2_std": float(scores["test_r2"].std()),
    }


def select_best_model(results):
    return min(
        results,
        key=lambda item: item["cross_validation"]["rmse_mean"],
    )


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
        "linear_regression": build_pipeline(LinearRegression()),
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
                cv_metrics = cross_validate_model(model, X_train, y_train)
                result = evaluate(name, model, X_train, y_train, X_test, y_test)
                result["cross_validation"] = cv_metrics
                mlflow.log_metrics(
                    {
                        "mae": result["mae"],
                        "rmse": result["rmse"],
                        "r2": result["r2"],
                        **{
                            f"cv_{key}": value
                            for key, value in cv_metrics.items()
                            if key.endswith("_mean")
                        },
                    }
                )
                results.append(result)

    best = select_best_model(results)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], MODEL_DIR / "diamond_model.joblib")
    explainability_background = X_train.sample(
        min(100, len(X_train)), random_state=42
    ).reset_index(drop=True)
    joblib.dump(explainability_background, MODEL_DIR / "shap_background.joblib")
    price_explainer = DiamondPriceExplainer(
        best["model"], explainability_background
    )
    shap_sample = X_test.sample(min(200, len(X_test)), random_state=42)
    shap_values = price_explainer.explain(shap_sample)
    shap_importance = (
        shap_values.abs()
        .mean()
        .sort_values(ascending=False)
        .rename("mean_absolute_shap")
        .rename_axis("feature")
        .reset_index()
    )
    shap_importance.to_csv(
        REPORT_DIR / "shap_feature_importance.csv", index=False
    )
    shap_sample_explanations = shap_values.copy()
    shap_sample_explanations.insert(
        0, "actual_price", y_test.loc[shap_sample.index].to_numpy()
    )
    shap_sample_explanations.insert(
        1, "predicted_price", best["model"].predict(shap_sample)
    )
    shap_sample_explanations.to_csv(
        REPORT_DIR / "shap_sample_explanations.csv", index=False
    )
    shap_plot = px.bar(
        shap_importance.sort_values("mean_absolute_shap"),
        x="mean_absolute_shap",
        y="feature",
        orientation="h",
        title="Mean absolute SHAP value on the test sample",
        labels={"mean_absolute_shap": "Mean absolute contribution ($)"},
    )
    test_predictions = pd.DataFrame(
        {
            "actual_price": y_test.to_numpy(),
            "predicted_price": best["model"].predict(X_test),
        }
    )
    test_predictions["residual"] = (
        test_predictions["actual_price"] - test_predictions["predicted_price"]
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    test_predictions.to_csv(REPORT_DIR / "model_predictions.csv", index=False)
    prediction_sample = test_predictions.sample(
        min(5000, len(test_predictions)), random_state=42
    )
    actual_vs_predicted = px.scatter(
        prediction_sample,
        x="actual_price",
        y="predicted_price",
        title=f"{best['name']}: Actual vs Predicted Prices",
        labels={"actual_price": "Actual price ($)", "predicted_price": "Predicted price ($)"},
        opacity=0.55,
    )
    actual_vs_predicted.add_trace(
        go.Scatter(
            x=[0, float(test_predictions["actual_price"].max())],
            y=[0, float(test_predictions["actual_price"].max())],
            mode="lines",
            name="Perfect prediction",
        )
    )
    residual_plot = px.scatter(
        prediction_sample,
        x="predicted_price",
        y="residual",
        title=f"{best['name']}: Residuals vs Predicted Prices",
        labels={"predicted_price": "Predicted price ($)", "residual": "Residual ($)"},
        opacity=0.55,
    )
    residual_plot.add_hline(y=0, line_dash="dash")
    (REPORT_DIR / "model_diagnostics.html").write_text(
        "\n".join(
            [
                "<h1>Model Evaluation</h1>",
                f"<p>Selected model: {best['name']}</p>",
                "<p>These plots use the held-out test split. Residual = actual - predicted.</p>",
                actual_vs_predicted.to_html(full_html=False, include_plotlyjs="cdn"),
                residual_plot.to_html(full_html=False, include_plotlyjs=False),
                "<h2>SHAP feature importance</h2>",
                "<p>Absolute contributions are averaged over a random sample of "
                "200 held-out rows. This describes model behaviour, not causation.</p>",
                shap_plot.to_html(full_html=False, include_plotlyjs=False),
            ]
        ),
        encoding="utf-8",
    )

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
            item["name"]: {
                **{key: item[key] for key in ("mae", "rmse", "r2")},
                "cross_validation": item["cross_validation"],
            }
            for item in results
        },
        "row_count": len(df),
        "evaluation": {
            "test_fraction": 0.20,
            "cross_validation_folds": 3,
            "random_state": 42,
            "selection_metric": "mean cross-validation RMSE",
        },
        "feature_columns": list(X.columns),
        "feature_importance": importance,
        "shap_feature_importance": [
            {
                "feature": row.feature,
                "mean_absolute_shap": float(row.mean_absolute_shap),
            }
            for row in shap_importance.itertuples(index=False)
        ],
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
