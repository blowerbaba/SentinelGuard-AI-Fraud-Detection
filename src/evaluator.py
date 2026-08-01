"""
Evaluation & Business Metrics Module for SentinelGuard AI.
Computes PR-AUC, ROC-AUC, F1, Confusion Matrix, and Financial Loss Reduction metrics.
"""

import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)


class Evaluator:
    """Evaluates fraud detection models with statistical and financial loss metrics."""

    def __init__(
        self,
        avg_fraud_cost: float = 250.0,
        false_alert_cost: float = 15.0,
        decision_threshold: float = 0.45
    ):
        self.avg_fraud_cost = avg_fraud_cost
        self.false_alert_cost = false_alert_cost
        self.decision_threshold = decision_threshold

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """Compute full benchmark suite for a model."""
        y_pred = (y_prob >= self.decision_threshold).astype(int)

        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        # Financial Business Loss Analysis
        # Without model: all fraud undetected (total fraud cost = (fn + tp) * avg_fraud_cost)
        total_fraud_count = fn + tp
        unmitigated_loss = total_fraud_count * self.avg_fraud_cost

        # With model: FN incurs fraud loss, FP incurs investigation operational cost
        actual_loss = (fn * self.avg_fraud_cost) + (fp * self.false_alert_cost)
        cost_saved = unmitigated_loss - actual_loss
        savings_percentage = (cost_saved / (unmitigated_loss + 1e-8)) * 100.0

        # Precision-Recall & ROC Curve Points for UI rendering
        precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)

        # Downsample curve points for UI chart efficiency (max 30 points)
        step_pr = max(1, len(precisions) // 30)
        step_roc = max(1, len(fpr) // 30)

        pr_curve = [
            {"recall": round(float(r), 4), "precision": round(float(p), 4)}
            for p, r in zip(precisions[::step_pr], recalls[::step_pr])
        ]
        roc_curve_data = [
            {"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
            for f, t in zip(fpr[::step_roc], tpr[::step_roc])
        ]

        return {
            "model_name": model_name,
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "financial_metrics": {
                "unmitigated_loss_usd": round(unmitigated_loss, 2),
                "actual_loss_usd": round(actual_loss, 2),
                "cost_saved_usd": round(cost_saved, 2),
                "savings_percentage": round(savings_percentage, 2)
            },
            "pr_curve": pr_curve,
            "roc_curve": roc_curve_data
        }
