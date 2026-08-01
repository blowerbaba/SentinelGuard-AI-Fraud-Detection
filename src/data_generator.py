"""
Synthetic Data Generator for SentinelGuard AI Fraud Detection System.
Generates realistic credit card transactions with extreme class imbalance (~1% fraud).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class FraudDataGenerator:
    """Generates synthetic high-dimensional financial transaction data with realistic fraud patterns."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(self.seed)

    def generate_data(
        self, num_samples: int = 12000, fraud_ratio: float = 0.012
    ) -> pd.DataFrame:
        """
        Generate synthetic transactions.

        Parameters:
        -----------
        num_samples : int
            Total number of transactions to generate.
        fraud_ratio : float
            Proportion of fraudulent transactions (e.g. 0.012 = 1.2%).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing transaction features and 'is_fraud' target.
        """
        num_fraud = int(num_samples * fraud_ratio)
        num_legit = num_samples - num_fraud

        # --- Legitimate Transactions ---
        legit_amount = np.random.exponential(scale=65.0, size=num_legit) + 2.0
        legit_dist = np.random.gamma(shape=2.0, scale=8.0, size=num_legit)
        legit_time_delta = np.random.exponential(scale=1800.0, size=num_legit) + 10.0 # seconds
        legit_velocity_1h = np.random.poisson(lam=1.5, size=num_legit)
        legit_device_score = np.random.beta(a=1.5, b=8.0, size=num_legit) # low risk
        p_legit = np.array([
            0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.04, 0.06, 0.07, 0.07, 0.07, 0.06,
            0.06, 0.06, 0.06, 0.07, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01
        ])
        p_legit = p_legit / p_legit.sum()
        legit_hour = np.random.choice(range(24), size=num_legit, p=p_legit)

        legit_international = np.random.binomial(n=1, p=0.03, size=num_legit)
        legit_online = np.random.binomial(n=1, p=0.45, size=num_legit)
        legit_failed_pins = np.random.binomial(n=2, p=0.02, size=num_legit)

        # --- Fraudulent Transactions ---
        # Fraud often has higher amounts or micro-test amounts, higher distance, high velocity, high device risk score
        fraud_amount_high = np.random.uniform(350.0, 2500.0, size=int(num_fraud * 0.7))
        fraud_amount_micro = np.random.uniform(0.5, 3.0, size=num_fraud - int(num_fraud * 0.7))
        fraud_amount = np.concatenate([fraud_amount_high, fraud_amount_micro])
        np.random.shuffle(fraud_amount)

        fraud_dist = np.random.gamma(shape=5.0, scale=40.0, size=num_fraud)
        fraud_time_delta = np.random.exponential(scale=60.0, size=num_fraud) + 1.0 # burst activity
        fraud_velocity_1h = np.random.poisson(lam=6.5, size=num_fraud)
        fraud_device_score = np.random.beta(a=6.0, b=2.0, size=num_fraud) # high risk
        p_fraud = np.array([
            0.08, 0.09, 0.10, 0.09, 0.08, 0.06, 0.04, 0.02, 0.02, 0.02, 0.02, 0.02,
            0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.06, 0.07, 0.06
        ])
        p_fraud = p_fraud / p_fraud.sum()
        fraud_hour = np.random.choice(range(24), size=num_fraud, p=p_fraud)

        fraud_international = np.random.binomial(n=1, p=0.35, size=num_fraud)
        fraud_online = np.random.binomial(n=1, p=0.85, size=num_fraud)
        fraud_failed_pins = np.random.binomial(n=3, p=0.30, size=num_fraud)

        # Combine domain features
        amount = np.concatenate([legit_amount, fraud_amount])
        distance_from_home = np.concatenate([legit_dist, fraud_dist])
        time_since_last_txn = np.concatenate([legit_time_delta, fraud_time_delta])
        velocity_1h = np.concatenate([legit_velocity_1h, fraud_velocity_1h])
        device_risk_score = np.concatenate([legit_device_score, fraud_device_score])
        hour_of_day = np.concatenate([legit_hour, fraud_hour])
        is_international = np.concatenate([legit_international, fraud_international])
        is_online = np.concatenate([legit_online, fraud_online])
        failed_pin_attempts = np.concatenate([legit_failed_pins, fraud_failed_pins])

        is_fraud = np.array([0] * num_legit + [1] * num_fraud)

        # Generate synthetic anonymized PCA features V1-V10
        pca_features = {}
        for i in range(1, 11):
            # Fraudulent transactions shift distributions on certain components
            shift = np.random.normal(loc=0, scale=0.5) if i not in [1, 3, 4, 9] else np.random.choice([-1.5, 1.8])
            legit_v = np.random.normal(loc=0.0, scale=1.0, size=num_legit)
            fraud_v = np.random.normal(loc=shift, scale=1.5, size=num_fraud)
            pca_features[f"V{i}"] = np.concatenate([legit_v, fraud_v])

        df_dict = {
            "amount": amount,
            "distance_from_home": distance_from_home,
            "time_since_last_txn": time_since_last_txn,
            "velocity_1h": velocity_1h,
            "device_risk_score": device_risk_score,
            "hour_of_day": hour_of_day,
            "is_international": is_international,
            "is_online": is_online,
            "failed_pin_attempts": failed_pin_attempts,
            "is_fraud": is_fraud
        }

        df_dict.update(pca_features)
        df = pd.DataFrame(df_dict)

        # Shuffle dataset
        df = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        return df


if __name__ == "__main__":
    generator = FraudDataGenerator()
    df = generator.generate_data(num_samples=1000)
    print(f"Generated DataFrame shape: {df.shape}")
    print(f"Fraud count: {df['is_fraud'].sum()} / {len(df)} ({df['is_fraud'].mean():.2%})")
