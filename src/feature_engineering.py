"""
Feature Engineering module for SentinelGuard AI.
Transforms raw transaction inputs into rich domain features for fraud detection.
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Computes advanced features for transaction fraud scoring."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations to a DataFrame or single transaction record.
        """
        df = X.copy()

        # 1. Log transform skewed amount & distance
        df["log_amount"] = np.log1p(df["amount"])
        df["log_distance"] = np.log1p(df["distance_from_home"])

        # 2. Time & Hour interaction features
        df["is_night_transaction"] = df["hour_of_day"].isin([0, 1, 2, 3, 4, 5]).astype(int)

        # 3. Velocity & Burst behavior
        df["burst_velocity_flag"] = (
            (df["velocity_1h"] >= 4) & (df["time_since_last_txn"] < 120.0)
        ).astype(int)
        
        df["amount_per_velocity"] = df["amount"] / (df["velocity_1h"] + 1.0)

        # 4. Location & Cross-border risk
        df["high_distance_flag"] = (df["distance_from_home"] > 50.0).astype(int)
        df["cross_border_high_amount"] = (
            (df["is_international"] == 1) & (df["amount"] > 200.0)
        ).astype(int)

        # 5. Composite Security Index
        df["composite_risk_index"] = (
            df["device_risk_score"] * 3.0 +
            df["failed_pin_attempts"] * 1.5 +
            df["is_international"] * 1.0 +
            df["is_night_transaction"] * 1.0
        )

        return df

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


if __name__ == "__main__":
    from data_generator import FraudDataGenerator
    df = FraudDataGenerator().generate_data(num_samples=100)
    fe = FeatureEngineer()
    df_engineered = fe.transform(df)
    print(f"Original cols: {df.shape[1]}, Engineered cols: {df_engineered.shape[1]}")
    print("New columns:", [c for c in df_engineered.columns if c not in df.columns])
