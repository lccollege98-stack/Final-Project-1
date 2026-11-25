import json
import joblib
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.data_pipeline import DiamondFeatureEngineer, clean_diamonds
from src.explainability import DiamondPriceExplainer
from src.rag import JewelryKnowledgeAssistant
from src.train import select_best_model


def test_clean_diamonds_removes_invalid_dimensions():
    frame = pd.DataFrame([{
        "carat": 1.0, "cut": "Ideal", "color": "G", "clarity": "VS2",
        "depth": 62.0, "table": 57.0, "x": 0.0, "y": 6.0, "z": 4.0, "price": 5000,
    }])
    cleaned = clean_diamonds(frame)
    assert cleaned.empty


def test_feature_engineering_adds_volume_and_premium_score():
    frame = pd.DataFrame([{
        "carat": 1.0, "cut": "Ideal", "color": "D", "clarity": "IF",
        "depth": 62.0, "table": 57.0, "x": 6.0, "y": 6.0, "z": 4.0,
    }])
    engineered = DiamondFeatureEngineer().fit_transform(frame)
    assert engineered.loc[0, "volume"] == 144.0
    assert engineered.loc[0, "premium_score"] == 20


def test_advisor_uses_retrieved_notes_without_api_key():
    assistant = JewelryKnowledgeAssistant(api_key="")

    result = assistant.answer("What does VS1 clarity mean?")

    assert result["mode"] == "retrieval"
    assert "clarity" in result["answer"].lower()
    assert result["sources"]


def test_rag_retrieves_the_requested_gia_reference():
    assistant = JewelryKnowledgeAssistant(api_key="")

    documents = assistant.retrieve("How is diamond carat weight measured?")

    assert documents
    assert documents[0]["title"] == "Carat weight"
    assert documents[0]["source"].startswith("GIA,")


def test_advisor_sends_retrieved_context_to_compatible_llm():
    assistant = JewelryKnowledgeAssistant(
        api_key="test-key",
        base_url="https://llm.example/v1",
        model="test-model",
    )
    response = MagicMock()
    response.__enter__.return_value.read.return_value = (
        b'{"choices":[{"message":{"content":"VS1 is a clarity grade."}}]}'
    )

    with patch("src.rag.urlopen", return_value=response) as mocked_urlopen:
        result = assistant.answer("What does VS1 clarity mean?")

    request = mocked_urlopen.call_args.args[0]
    request_body = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "https://llm.example/v1/chat/completions"
    assert request_body["model"] == "test-model"
    assert "Diamond clarity" in request_body["messages"][1]["content"]
    assert result["mode"] == "custom_llm"
    assert result["answer"] == "VS1 is a clarity grade."


def test_eda_report_generates_files():
    from src.eda_report import build_report

    build_report()
    root = Path(__file__).resolve().parents[1]

    assert (root / "reports" / "diamond_eda.html").exists()
    assert (root / "reports" / "diamond_eda_summary.md").exists()


def test_shap_attributions_reconstruct_the_prediction():
    root = Path(__file__).resolve().parents[1]
    model = joblib.load(root / "models" / "diamond_model.joblib")
    background = joblib.load(root / "models" / "shap_background.joblib")
    diamond = clean_diamonds(
        pd.read_csv(root / "data" / "diamonds.csv")
    ).drop(columns=["price"]).iloc[[0]]
    explainer = DiamondPriceExplainer(model, background)

    contributions = explainer.explain(diamond)
    reconstructed_price = (
        contributions.sum(axis=1).iloc[0]
        + contributions.attrs["base_values"][0]
    )
    predicted_price = model.predict(diamond)[0]

    assert np.isclose(reconstructed_price, predicted_price, atol=0.1)


def test_shap_feature_groups_keep_engineered_features_separate():
    groups = DiamondPriceExplainer._feature_groups(
        [
            "num__carat",
            "num__carat_volume_ratio",
            "num__clarity_score",
            "cat__clarity_VS1",
        ],
        ["carat", "clarity"],
    )

    assert groups["carat"] == ["num__carat"]
    assert groups["carat_volume_ratio"] == ["num__carat_volume_ratio"]
    assert groups["clarity_score"] == ["num__clarity_score"]
    assert groups["clarity"] == ["cat__clarity_VS1"]


def test_model_selection_uses_cross_validation_not_test_results():
    results = [
        {
            "name": "lower_test_error",
            "rmse": 1,
            "cross_validation": {"rmse_mean": 100},
        },
        {
            "name": "lower_cv_error",
            "rmse": 100,
            "cross_validation": {"rmse_mean": 10},
        },
    ]

    assert select_best_model(results)["name"] == "lower_cv_error"
