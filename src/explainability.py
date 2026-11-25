from __future__ import annotations

import numpy as np
import pandas as pd
import shap


class DiamondPriceExplainer:
    def __init__(self, pipeline, background: pd.DataFrame):
        self.pipeline = pipeline
        self.feature_engineer = pipeline.named_steps["features"]
        self.preprocessing = pipeline.named_steps["preprocessing"]
        self.estimator = pipeline.named_steps["model"]
        background_matrix = self._transform(background)
        encoded_names = self.preprocessing.get_feature_names_out()
        self.feature_groups = self._feature_groups(
            encoded_names, list(background.columns)
        )
        self.explainer = shap.Explainer(
            self.estimator,
            background_matrix,
            feature_names=encoded_names,
        )

    def _transform(self, frame: pd.DataFrame):
        engineered = self.feature_engineer.transform(frame)
        return self.preprocessing.transform(engineered)

    @staticmethod
    def _feature_groups(encoded_names, input_columns):
        groups = {}
        for encoded_name in encoded_names:
            encoded_name = str(encoded_name)
            transformer, _, name = encoded_name.partition("__")
            if not name:
                name = transformer
            source_feature = name
            if transformer == "cat" and name not in input_columns:
                source_feature = next(
                    (
                        column
                        for column in ("cut", "color", "clarity")
                        if column in input_columns
                        and name.startswith(f"{column}_")
                    ),
                    name,
                )
            groups.setdefault(source_feature, []).append(encoded_name)
        return groups

    def explain(self, frame: pd.DataFrame) -> pd.DataFrame:
        shap_explanation = self.explainer(self._transform(frame))
        values = np.asarray(shap_explanation.values)
        if values.ndim == 3:
            values = values[:, :, 0]
        encoded_names = list(self.preprocessing.get_feature_names_out())
        explanations = {}
        for feature, encoded_features in self.feature_groups.items():
            positions = [
                encoded_names.index(encoded_name)
                for encoded_name in encoded_features
            ]
            explanations[feature] = values[:, positions].sum(axis=1)
        result = pd.DataFrame(explanations, index=frame.index)
        base_values = np.asarray(shap_explanation.base_values)
        if base_values.ndim > 1:
            base_values = base_values[:, 0]
        result.attrs["base_values"] = base_values
        return result
