"""
Unit tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert "status" in json_data


def test_predict_single_endpoint():
    payload = {
        "amount": 150.0,
        "distance_from_home": 15.0,
        "time_since_last_txn": 1200.0,
        "velocity_1h": 1,
        "device_risk_score": 0.20,
        "hour_of_day": 14,
        "is_international": 0,
        "is_online": 1,
        "failed_pin_attempts": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "risk_level" in data
    assert "recommendation" in data
    assert "latency_ms" in data


def test_predict_batch_endpoint():
    payload = {
        "transactions": [
            {
                "amount": 45.0,
                "distance_from_home": 2.0,
                "time_since_last_txn": 3000.0,
                "velocity_1h": 1,
                "device_risk_score": 0.05,
                "hour_of_day": 12,
                "is_international": 0,
                "is_online": 1,
                "failed_pin_attempts": 0
            },
            {
                "amount": 1800.0,
                "distance_from_home": 400.0,
                "time_since_last_txn": 10.0,
                "velocity_1h": 9,
                "device_risk_score": 0.95,
                "hour_of_day": 3,
                "is_international": 1,
                "is_online": 1,
                "failed_pin_attempts": 3
            }
        ]
    }
    response = client.post("/predict-batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2
