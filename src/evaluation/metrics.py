from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae_rmse_from_logs(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> tuple[float, float]:
    y_true = np.expm1(y_true_log.squeeze())
    y_pred = np.expm1(y_pred_log.squeeze())
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return float(mae), float(rmse)
