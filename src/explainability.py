"""
Explainable AI (XAI) Module for SentinelGuard AI.
Computes SHAP feature importance and transaction-level explanation payloads.
"""

import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, List


class FraudExplainer:
    """Provides global and local model explanations using TreeSHAP."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(self.model)

    def explain_transaction(self, X_sample: np.ndarray, original_row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate local SHAP feature breakdown for a single transaction.
        """
        shap_values = self.explainer.shap_values(X_sample)
        
        # Handle binary classification array output vs single 1D array
        if isinstance(shap_values, list):
            sv = shap_values[1][0] # class 1 (fraud)
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        contributions = []
        for feat_name, shap_val in zip(self.feature_names, sv):
            orig_val = original_row.get(feat_name, None)
            contributions.append({
                "feature": feat_name,
                "shap_value": round(float(shap_val), 4),
                "feature_value": round(float(orig_val), 4) if orig_val is not None and isinstance(orig_val, (int, float)) else orig_val
            })

        # Sort by absolute SHAP impact
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "base_value": round(float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value), 4),
            "top_drivers": contributions[:7]
        }

    def get_global_feature_importance(self, X_background: np.ndarray) -> List[Dict[str, Any]]:
        """
        Calculate global mean absolute SHAP values across background set.
        """
        shap_values = self.explainer.shap_values(X_background)
        if isinstance(shap_values, list):
            sv = np.abs(shap_values[1]).mean(axis=0)
        elif len(shap_values.shape) == 3:
            sv = np.abs(shap_values[:, :, 1]).mean(axis=0)
        else:
            sv = np.abs(shap_values).mean(axis=0)

        importance_list = [
            {"feature": feat, "importance": round(float(val), 4)}
            for feat, val in zip(self.feature_names, sv)
        ]
        importance_list.sort(key=lambda x: x["importance"], reverse=True)
        return importance_list
