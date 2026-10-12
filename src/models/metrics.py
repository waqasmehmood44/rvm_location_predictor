"""Evaluation metrics for Coripet regression targets."""

from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RegressionMetrics:
    """Calculates evaluation metrics for operational predictions."""

    @staticmethod
    def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Weighted Absolute Percentage Error: sum(|y - y_hat|) / sum(y)."""
        denom = np.sum(np.abs(y_true))
        if denom == 0:
            return 0.0
        return float(np.sum(np.abs(y_true - y_pred)) / denom)

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
        """Mean Absolute Percentage Error with epsilon stabilization."""
        return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)

    @classmethod
    def calculate_all(
        cls, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
    ) -> Dict[str, float]:
        """Computes full suite of operational regression metrics."""
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        wape_val = cls.wape(y_true, y_pred)
        mape_val = cls.mape(y_true, y_pred)

        return {
            f"{prefix}mae": round(mae, 2),
            f"{prefix}rmse": round(rmse, 2),
            f"{prefix}r2": round(r2, 4),
            f"{prefix}wape": round(wape_val, 4),
            f"{prefix}mape": round(mape_val, 2),
        }
