"""
Unit tests for data synthesis, feature engineering, and preprocessor modules.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_generator import FraudDataGenerator
from src.feature_engineering import FeatureEngineer
from src.preprocessor import FraudPreprocessor
from src.models import ModelZoo
from src.evaluator import Evaluator


def test_data_generator():
    generator = FraudDataGenerator(seed=42)
    df = generator.generate_data(num_samples=500, fraud_ratio=0.02)
    assert len(df) == 500
    assert "is_fraud" in df.columns
    assert df["is_fraud"].sum() > 0
    assert "amount" in df.columns
    assert "V1" in df.columns


def test_feature_engineering():
    generator = FraudDataGenerator(seed=42)
    df = generator.generate_data(num_samples=100)
    fe = FeatureEngineer()
    df_transformed = fe.transform(df)
    
    assert "log_amount" in df_transformed.columns
    assert "is_night_transaction" in df_transformed.columns
    assert "composite_risk_index" in df_transformed.columns
    assert len(df_transformed) == 100


def test_preprocessor_and_smote():
    generator = FraudDataGenerator(seed=42)
    raw_df = generator.generate_data(num_samples=400, fraud_ratio=0.02)
    fe = FeatureEngineer()
    df_eng = fe.transform(raw_df)
    
    X = df_eng.drop(columns=["is_fraud"])
    y = df_eng["is_fraud"]

    preprocessor = FraudPreprocessor(use_smote=True, smote_sampling_strategy=0.2)
    X_res, y_res, feat_names = preprocessor.fit_transform_train(X, y)

    assert len(X_res) > len(X)
    assert y_res.sum() > y.sum()
    assert len(feat_names) == X.shape[1]


def test_model_zoo_training():
    generator = FraudDataGenerator(seed=42)
    raw_df = generator.generate_data(num_samples=300, fraud_ratio=0.05)
    fe = FeatureEngineer()
    df_eng = fe.transform(raw_df)
    
    X = df_eng.drop(columns=["is_fraud"])
    y = df_eng["is_fraud"]

    preprocessor = FraudPreprocessor(use_smote=False)
    X_scaled, y_scaled, _ = preprocessor.fit_transform_train(X, y)

    zoo = ModelZoo(random_state=42)
    zoo.build_models(scale_pos_weight=5.0)
    zoo.train_all(X_scaled, y_scaled)

    xgb_probs = zoo.predict_proba("XGBoost", X_scaled)
    assert len(xgb_probs) == len(X_scaled)
    assert (xgb_probs >= 0.0).all() and (xgb_probs <= 1.0).all()
