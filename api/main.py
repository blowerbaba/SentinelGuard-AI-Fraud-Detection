"""
FastAPI Microservice Engine for SentinelGuard AI Real-Time Fraud Detection.
"""

import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any

from api.schemas import (
    TransactionInput,
    TransactionBatchInput,
    PredictionResponse,
    BatchPredictionResponse,
    DriverImpact
)
from src.feature_engineering import FeatureEngineer
from src.preprocessor import FraudPreprocessor
from src.explainability import FraudExplainer


app = FastAPI(
    title="SentinelGuard AI - Real-Time Fraud Microservice",
    description="Production REST API for high-frequency financial transaction fraud scoring & SHAP explainability",
    version="1.0.0"
)

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state variables
MODEL_STATE: Dict[str, Any] = {
    "champion_model": None,
    "preprocessor": None,
    "feature_engineer": FeatureEngineer(),
    "explainer": None,
    "metrics": {},
    "feature_names": []
}


def load_artifacts():
    """Load pre-trained artifacts from models directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")

    champion_path = os.path.join(models_dir, "champion_model.pkl")
    preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
    metrics_path = os.path.join(models_dir, "model_metrics.json")
    feature_names_path = os.path.join(models_dir, "feature_names.json")

    if not os.path.exists(champion_path) or not os.path.exists(preprocessor_path):
        print("⚠️ Warning: Model artifacts not found! Run pipeline first.")
        return

    MODEL_STATE["champion_model"] = joblib.load(champion_path)
    MODEL_STATE["preprocessor"] = FraudPreprocessor.load(preprocessor_path)

    if os.path.exists(feature_names_path):
        with open(feature_names_path, "r") as f:
            MODEL_STATE["feature_names"] = json.load(f)

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            MODEL_STATE["metrics"] = json.load(f)

    if MODEL_STATE["champion_model"] and MODEL_STATE["feature_names"]:
        try:
            MODEL_STATE["explainer"] = FraudExplainer(
                MODEL_STATE["champion_model"], MODEL_STATE["feature_names"]
            )
        except Exception as e:
            print(f"⚠️ SHAP Explainer warning: {e}")


@app.on_event("startup")
def startup_event():
    load_artifacts()


@app.get("/health", summary="Microservice Health Check")
def health_check():
    is_loaded = MODEL_STATE["champion_model"] is not None
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "champion_model": MODEL_STATE["metrics"].get("champion_model", "N/A")
    }


@app.get("/metrics", summary="Model Performance & XAI Benchmarks")
def get_metrics():
    if not MODEL_STATE["metrics"]:
        load_artifacts()
    if not MODEL_STATE["metrics"]:
        raise HTTPException(status_code=404, detail="Model metrics artifact not found")
    return MODEL_STATE["metrics"]


def _process_single_transaction(txn_dict: Dict[str, Any]) -> PredictionResponse:
    start_time = time.perf_counter()
    
    if MODEL_STATE["champion_model"] is None or MODEL_STATE["preprocessor"] is None:
        load_artifacts()
        if MODEL_STATE["champion_model"] is None:
            raise HTTPException(status_code=500, detail="Model artifacts are missing. Run pipeline first.")

    raw_df = pd.DataFrame([txn_dict])
    fe = MODEL_STATE["feature_engineer"]
    engineered_df = fe.transform(raw_df)
    
    preprocessor = MODEL_STATE["preprocessor"]
    scaled_features = preprocessor.transform(engineered_df)

    model = MODEL_STATE["champion_model"]
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(scaled_features)[0, 1])
    else:
        prob = float(model.predict(scaled_features)[0])

    # Risk Triage Thresholds
    is_fraud_suspected = prob >= 0.45
    if prob >= 0.70:
        risk_level = "HIGH"
        recommendation = "BLOCK_TRANSACTION"
    elif prob >= 0.45:
        risk_level = "MEDIUM"
        recommendation = "FLAG_FOR_REVIEW"
    else:
        risk_level = "LOW"
        recommendation = "APPROVE"

    # SHAP Driver Calculation
    top_drivers = []
    if MODEL_STATE["explainer"] is not None:
        try:
            explanation = MODEL_STATE["explainer"].explain_transaction(scaled_features, engineered_df.iloc[0].to_dict())
            top_drivers = [
                DriverImpact(
                    feature=d["feature"],
                    shap_value=d["shap_value"],
                    feature_value=d.get("feature_value")
                )
                for d in explanation.get("top_drivers", [])
            ]
        except Exception:
            top_drivers = []

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return PredictionResponse(
        fraud_probability=round(prob, 4),
        is_fraud_suspected=is_fraud_suspected,
        risk_level=risk_level,
        recommendation=recommendation,
        latency_ms=round(elapsed_ms, 2),
        top_risk_drivers=top_drivers
    )


@app.post("/predict", response_model=PredictionResponse, summary="Evaluate Single Transaction")
def predict_single(txn: TransactionInput):
    return _process_single_transaction(txn.model_dump())


@app.post("/predict-batch", response_model=BatchPredictionResponse, summary="Evaluate Batch Transactions")
def predict_batch(batch: TransactionBatchInput):
    start_time = time.perf_counter()
    responses = [_process_single_transaction(txn.model_dump()) for txn in batch.transactions]

    
    flagged_count = sum(1 for r in responses if r.is_fraud_suspected)
    total = len(responses)
    pct = (flagged_count / total * 100.0) if total > 0 else 0.0

    return BatchPredictionResponse(
        total_processed=total,
        flagged_fraud_count=flagged_count,
        high_risk_percentage=round(pct, 2),
        predictions=responses
    )


# Serve static web frontend if mounted
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
web_dir = os.path.join(base_dir, "web")
if os.path.exists(web_dir):
    app.mount("/dashboard", StaticFiles(directory=web_dir, html=True), name="web")
