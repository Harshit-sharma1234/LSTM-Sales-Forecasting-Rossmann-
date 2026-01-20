import pandas as pd

def add_lags_and_rolls(frame: pd.DataFrame, target_col: str, lags=(1,7,14,28), roll_windows=(7,28)) -> pd.DataFrame:
    g = frame.copy()
    for L in lags:
        g[f'lag_{L}'] = g.groupby('Store')[target_col].shift(L)
    for w in roll_windows:
        g[f'roll{w}'] = g.groupby('Store')[target_col].shift(1).rolling(w).mean()
    return g

from typing import Tuple
import numpy as np

def make_sequences_by_store(frame: pd.DataFrame, X_scaled: np.ndarray, target: np.ndarray, lookback: int = 28, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    start_idx = 0
    for _, g in frame.groupby('Store'):
        g = g.sort_values('Date')
        n = len(g)
        Xg = X_scaled[start_idx:start_idx+n]
        yg = target[start_idx:start_idx+n]
        for i in range(lookback, n - horizon + 1):
            X_list.append(Xg[i - lookback:i, :])
            y_list.append(yg[i:i + horizon])
        start_idx += n
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)
