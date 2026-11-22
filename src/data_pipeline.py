from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DiamondFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["volume"] = df["x"] * df["y"] * df["z"]
        df["surface_area_proxy"] = df["x"] * df["y"]

        cut_scores = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
        color_scores = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
        clarity_scores = {
            "I1": 1,
            "SI2": 2,
            "SI1": 3,
            "VS2": 4,
            "VS1": 5,
            "VVS2": 6,
            "VVS1": 7,
            "IF": 8,
        }
        df["cut_score"] = df["cut"].map(cut_scores).fillna(0)
        df["color_score"] = df["color"].map(color_scores).fillna(0)
        df["clarity_score"] = df["clarity"].map(clarity_scores).fillna(0)
        df["premium_score"] = df["cut_score"] + df["color_score"] + df["clarity_score"]
        df["carat_volume_ratio"] = df["carat"] / df["volume"].replace(0, float("nan"))
        df["carat_volume_ratio"] = df["carat_volume_ratio"].fillna(0)
        return df


def clean_diamonds(data: pd.DataFrame) -> pd.DataFrame:
    required = {"carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z", "price"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = data.copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=required)
    dimension_columns = ["x", "y", "z"]
    df = df[(df[dimension_columns] > 0).all(axis=1)]
    df = df[
        df["depth"].between(40, 80)
        & df["table"].between(40, 90)
        & df["carat"].between(0.1, 10)
        & df["price"].between(100, 200_000)
    ]

    df["cut"] = df["cut"].astype(str).str.title()
    df["color"] = df["color"].astype(str).str.upper()
    df["clarity"] = df["clarity"].astype(str).str.upper()
    df["cut"] = df["cut"].replace({"Verygood": "Very Good"})
    return df.reset_index(drop=True)
