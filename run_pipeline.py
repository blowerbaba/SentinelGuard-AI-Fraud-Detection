"""
Master Pipeline Runner for SentinelGuard AI.
Executes data generation, feature engineering, SMOTE resampling, multi-model training, evaluation, and artifact export.
"""

import os
import sys
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure UTF-8 output encoding for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


from src.data_generator import FraudDataGenerator
from src.feature_engineering import FeatureEngineer
from src.preprocessor import FraudPreprocessor
from src.models import ModelZoo
from src.evaluator import Evaluator
from src.explainability import FraudExplainer


def run_full_pipeline():
    print("🚀 [1/6] Initializing SentinelGuard AI Master Pipeline...")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Generate Synthetic Data
    print("📊 [2/6] Generating synthetic transactions with imbalanced fraud (~1.2%)...")
    generator = FraudDataGenerator(seed=42)
    raw_df = generator.generate_data(num_samples=15000, fraud_ratio=0.012)
    raw_df.to_csv(os.path.join(data_dir, "transactions_raw.csv"), index=False)

    # 2. Feature Engineering
    print("⚙️ [3/6] Applying feature engineering (time velocity, spatial deltas, risk indices)...")
    fe = FeatureEngineer()
    engineered_df = fe.transform(raw_df)
    
    # Save feature columns template
    X = engineered_df.drop(columns=["is_fraud"])
    y = engineered_df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 3. Preprocessing & SMOTE Resampling
    print("🔄 [4/6] Preprocessing features & applying SMOTE oversampling to training set...")
    preprocessor = FraudPreprocessor(use_smote=True, smote_sampling_strategy=0.15)
    X_train_res, y_train_res, feature_names = preprocessor.fit_transform_train(X_train, y_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Save preprocessor artifact
    preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
    preprocessor.save(preprocessor_path)

    # Save feature names list
    with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    # 4. Multi-Model Zoo Training & Evaluation
    print("🧠 [5/6] Training Model Zoo (XGBoost, LightGBM, Random Forest, Isolation Forest)...")
    zoo = ModelZoo(random_state=42)
    scale_pos_weight = float((len(y_train) - y_train.sum()) / (y_train.sum() + 1e-8))
    zoo.build_models(scale_pos_weight=scale_pos_weight)
    zoo.train_all(X_train_res, y_train_res)

    evaluator = Evaluator()
    eval_results = {}
    best_model_name = None
    best_pr_auc = -1.0

    for model_name in zoo.models.keys():
        y_prob = zoo.predict_proba(model_name, X_test_scaled)
        metrics = evaluator.evaluate_model(y_test.values, y_prob, model_name=model_name)
        eval_results[model_name] = metrics
        print(f"   - {model_name:15s} | PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1_score']:.4f}")

        if metrics['pr_auc'] > best_pr_auc:
            best_pr_auc = metrics['pr_auc']
            best_model_name = model_name

    print(f"\n🏆 Champion Model Selected: {best_model_name} (PR-AUC: {best_pr_auc:.4f})")
    
    # Save Champion Model
    champion_path = os.path.join(models_dir, "champion_model.pkl")
    zoo.save_champion(best_model_name, champion_path)

    # 5. Explainability (SHAP XAI)
    print("🔍 [6/6] Computing SHAP explainability insights...")
    champion_model = zoo.models[best_model_name]
    global_importance = []
    if hasattr(champion_model, "predict_proba"):
        explainer = FraudExplainer(champion_model, feature_names)
        global_importance = explainer.get_global_feature_importance(X_test_scaled[:300])

    summary_payload = {
        "champion_model": best_model_name,
        "champion_metrics": eval_results[best_model_name],
        "all_models_metrics": eval_results,
        "global_feature_importance": global_importance,
        "dataset_stats": {
            "total_transactions": len(raw_df),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "fraud_cases": int(y.sum()),
            "fraud_rate_pct": round(float(y.mean() * 100), 2)
        }
    }

    metrics_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"✨ SentinelGuard AI Pipeline Completed Successfully!")
    print(f"📁 Saved Model Metrics to: {metrics_path}")
    print(f"📁 Saved Champion Model to: {champion_path}")


if __name__ == "__main__":
    run_full_pipeline()
