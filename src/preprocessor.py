"""
Preprocessing & Sampling Module for SentinelGuard AI.
Handles scaling, feature alignment, and class imbalance resampling (SMOTE).
"""

import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False



class FraudPreprocessor:
    """Preprocesses features and handles SMOTE oversampling for imbalanced fraud data."""

    def __init__(self, use_smote: bool = True, smote_sampling_strategy: float = 0.15):
        self.scaler = StandardScaler()
        self.use_smote = use_smote
        self.smote_sampling_strategy = smote_sampling_strategy
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit_transform_train(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Fit scaler on train set, scale features, and apply SMOTE to balance train set.
        """
        self.feature_names = list(X_train.columns)
        X_scaled = self.scaler.fit_transform(X_train)
        self.is_fitted = True

        if self.use_smote and HAS_SMOTE:
            smote = SMOTE(sampling_strategy=self.smote_sampling_strategy, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_scaled, y_train)
            return X_resampled, y_resampled, self.feature_names

        
        return X_scaled, y_train.values, self.feature_names

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Scale test/validation/production features without fitting or applying SMOTE.
        """
        if not self.is_fitted:
            raise ValueError("FraudPreprocessor is not fitted yet! Call fit_transform_train first.")
        
        # Ensure column ordering matches training set
        X_aligned = X[self.feature_names]
        return self.scaler.transform(X_aligned)

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "FraudPreprocessor":
        return joblib.load(filepath)
