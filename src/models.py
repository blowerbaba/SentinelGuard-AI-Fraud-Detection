"""
Model Zoo Module for SentinelGuard AI.
Implements XGBoost, LightGBM, Random Forest, and Isolation Forest anomaly models.
"""

import joblib
import numpy as np
from typing import Dict, Any, Tuple
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest


class ModelZoo:
    """Manages training, benchmarking, and saving of supervised and unsupervised ML models."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}

    def build_models(self, scale_pos_weight: float = 8.0) -> Dict[str, Any]:
        """Instantiate model zoo with tuned hyperparameters for imbalanced data."""
        self.models = {
            "XGBoost": XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric="logloss",
                n_jobs=-1
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                subsample=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=120,
                max_depth=8,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            ),
            "IsolationForest": IsolationForest(
                n_estimators=100,
                contamination=0.02,
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        return self.models

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train all models in the zoo."""
        if not self.models:
            self.build_models()

        for name, model in self.models.items():
            if name == "IsolationForest":
                # Unsupervised training (only features, fit on all training data)
                model.fit(X_train)
            else:
                model.fit(X_train, y_train)

    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Return fraud probability scores for a specific model."""
        model = self.models[model_name]
        if model_name == "IsolationForest":
            # Convert decision function / anomaly score into pseudo-probability [0, 1]
            scores = -model.decision_function(X) # lower score = more anomalous
            # Normalize to 0..1 range
            min_s, max_s = scores.min(), scores.max()
            norm_scores = (scores - min_s) / (max_s - min_s + 1e-8)
            return norm_scores
        else:
            return model.predict_proba(X)[:, 1]

    def save_champion(self, model_name: str, filepath: str):
        """Save selected champion model to disk."""
        joblib.dump(self.models[model_name], filepath)

    @staticmethod
    def load_model(filepath: str):
        """Load model from disk."""
        return joblib.load(filepath)
