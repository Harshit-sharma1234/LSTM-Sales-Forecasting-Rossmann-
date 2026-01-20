from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

from src.features.feature_engineering import add_lags_and_rolls


def predict_next(store_id: int, lookback: int = 28,
                 processed_path: str | Path = 'data/processed/rossmann_processed.csv',
                 model_path: str | Path = 'models/rossmann_lstm_best.keras',
                 scaler_path: str | Path = 'models/feature_scaler.pkl') -> float:
    processed_path = Path(processed_path)
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)

    df = pd.read_csv(processed_path, parse_dates=['Date'])
    df = df[df['Store'] == store_id].sort_values('Date').copy()
    
    if len(df) < lookback + 30:
        raise ValueError(f'Not enough history for store {store_id} (need {lookback + 30}, have {len(df)})')

    # Ensure Sales_log exists
    if 'Sales_log' not in df.columns:
        df['Sales_log'] = np.log1p(df['Sales'].clip(lower=0))

    # Compute lag and roll features directly (no groupby needed for single store)
    target_col = 'Sales_log'
    for L in (1, 7, 14, 28):
        df[f'lag_{L}'] = df[target_col].shift(L)
    for w in (7, 28):
        df[f'roll{w}'] = df[target_col].shift(1).rolling(w).mean()
    
    feature_cols = ['dow','month','year','week','Open','Promo','SchoolHoliday'] + \
                   [f'lag_{L}' for L in (1,7,14,28)] + [f'roll{w}' for w in (7,28)]
    
    # Drop NaN rows only in feature columns (not all columns)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    
    if len(df) < lookback:
        raise ValueError(f'Not enough valid rows after feature engineering for store {store_id} (have {len(df)}, need {lookback})')

    scaler = joblib.load(scaler_path)

    X_store = df[feature_cols].values.astype('float32')
    X_store = scaler.transform(X_store)

    X_window = X_store[-lookback:, :]
    X_window = X_window[None, ...]

    model = keras.models.load_model(model_path)
    pred_log = model.predict(X_window, verbose=0).squeeze()
    pred = float(np.expm1(pred_log))
    return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store_id', type=int, required=True)
    parser.add_argument('--lookback', type=int, default=28)
    parser.add_argument('--processed_path', type=str, default='data/processed/rossmann_processed.csv')
    parser.add_argument('--model_path', type=str, default='models/rossmann_lstm.keras')
    parser.add_argument('--scaler_path', type=str, default='models/feature_scaler.pkl')
    args = parser.parse_args()
    val = predict_next(args.store_id, args.lookback, args.processed_path, args.model_path, args.scaler_path)
    print({'store_id': args.store_id, 'next_day_sales_pred': val})
