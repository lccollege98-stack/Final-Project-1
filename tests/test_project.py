import json
import pandas as pd
from unittest.mock import MagicMock, patch

from src.data_pipeline import DiamondFeatureEngineer, clean_diamonds
from src.rag import JewelryKnowledgeAssistant


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
