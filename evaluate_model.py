import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from tensorflow import keras
from sklearn.metrics import mean_absolute_error

from src.features.feature_engineering import add_lags_and_rolls, make_sequences_by_store

def evaluate_model():
    # Load data
    processed_path = Path('data/processed/rossmann_processed.csv')
    df = pd.read_csv(processed_path, parse_dates=['Date'])
    
    # Feature Engineering (same as training)
    feat_df = add_lags_and_rolls(df, 'Sales_log')
    feat_df = feat_df.dropna().reset_index(drop=True)
    
    feature_cols = ['dow','month','year','week','Open','Promo','SchoolHoliday'] + \
                   [f'lag_{L}' for L in (1,7,14,28)] + [f'roll{w}' for w in (7,28)]
    
    y = feat_df['Sales_log'].values.astype('float32')
    X_num = feat_df[feature_cols].values.astype('float32')
    
    # Scale
    scaler = joblib.load('models/feature_scaler.pkl')
    X_scaled = scaler.transform(X_num)
    
    # Make sequences
    lookback = 28
    horizon = 1
    print("Generating sequences...")
    X_seq, y_seq = make_sequences_by_store(feat_df, X_scaled, y, lookback=lookback, horizon=horizon)
    
    # Split (same as training: last 10% is test)
    n = len(X_seq)
    test_size = int(0.1 * n)
    X_test = X_seq[n - test_size :]
    y_test = y_seq[n - test_size :]
    
    print(f"Evaluating on {len(X_test)} samples...")
    
    # Predict
    model = keras.models.load_model('models/rossmann_lstm_best.keras')
    pred_log = model.predict(X_test, verbose=0)
    
    # Convert back to real currency
    y_true = np.expm1(y_test.squeeze())
    y_pred = np.expm1(pred_log.squeeze())
    
    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print("\n--- Model Accuracy Results ---")
    print(f"Mean Absolute Error (MAE): €{mae:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("------------------------------")

if __name__ == '__main__':
    evaluate_model()
