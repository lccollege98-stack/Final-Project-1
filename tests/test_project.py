import pandas as pd

from src.data_pipeline import DiamondFeatureEngineer, clean_diamonds


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
