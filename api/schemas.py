"""
Pydantic Request & Response Schemas for SentinelGuard AI FastAPI Microservice.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class TransactionInput(BaseModel):
    """Schema for a single financial transaction."""
    amount: float = Field(..., json_schema_extra={"example": 120.50}, description="Transaction amount in USD")
    distance_from_home: float = Field(..., json_schema_extra={"example": 12.4}, description="Distance from home location in km")
    time_since_last_txn: float = Field(..., json_schema_extra={"example": 450.0}, description="Time since last transaction in seconds")
    velocity_1h: int = Field(..., json_schema_extra={"example": 2}, description="Count of transactions in past 1 hour")
    device_risk_score: float = Field(..., json_schema_extra={"example": 0.15}, description="Device fingerprint risk score (0.0 to 1.0)")
    hour_of_day: int = Field(..., json_schema_extra={"example": 14}, description="Hour of day (0-23)")
    is_international: int = Field(..., json_schema_extra={"example": 0}, description="1 if international transaction, else 0")
    is_online: int = Field(..., json_schema_extra={"example": 1}, description="1 if e-commerce/online transaction, else 0")
    failed_pin_attempts: int = Field(..., json_schema_extra={"example": 0}, description="Number of failed PIN attempts")

    
    # Anonymized PCA features (Optional, defaulted to 0.0 if omitted)
    V1: float = Field(default=0.0)
    V2: float = Field(default=0.0)
    V3: float = Field(default=0.0)
    V4: float = Field(default=0.0)
    V5: float = Field(default=0.0)
    V6: float = Field(default=0.0)
    V7: float = Field(default=0.0)
    V8: float = Field(default=0.0)
    V9: float = Field(default=0.0)
    V10: float = Field(default=0.0)


class TransactionBatchInput(BaseModel):
    """Schema for batch transaction risk scoring."""
    transactions: List[TransactionInput]


class DriverImpact(BaseModel):
    feature: str
    shap_value: float
    feature_value: Optional[Any] = None


class PredictionResponse(BaseModel):
    """Response payload for a single transaction fraud evaluation."""
    fraud_probability: float
    is_fraud_suspected: bool
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    recommendation: str  # "APPROVE", "FLAG_FOR_REVIEW", "BLOCK_TRANSACTION"
    latency_ms: float
    top_risk_drivers: List[DriverImpact]


class BatchPredictionResponse(BaseModel):
    """Response payload for batch transaction fraud evaluation."""
    total_processed: int
    flagged_fraud_count: int
    high_risk_percentage: float
    predictions: List[PredictionResponse]
