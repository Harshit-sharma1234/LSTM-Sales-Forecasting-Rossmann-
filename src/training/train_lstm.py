from __future__ import annotations
import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import random

from src.preprocessing.load_data import load_raw, merge_train_store, add_basic_clean, save_processed
from src.features.feature_engineering import add_lags_and_rolls, make_sequences_by_store
from src.models.lstm_model import build_model
from src.evaluation.metrics import mae_rmse_from_logs


def main(config_path: str = 'configs/default.yaml') -> None:
    # Reproducibility
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    paths = cfg['paths']
    model_cfg = cfg['model']
    split_cfg = cfg['split']

    raw_dir = Path(paths['raw_dir'])
    processed_path = Path(paths['processed_path'])
    model_path = Path(paths['model_path'])
    scaler_path = Path(paths['scaler_path'])
    plot_path = Path(paths['plot_path'])
    preds_path = Path(paths['preds_path'])

    # Load and process
    train, test, store = load_raw(raw_dir)
    df = merge_train_store(train, store)
    df = add_basic_clean(df)
    save_processed(df, processed_path)

    # Features
    feat_df = add_lags_and_rolls(df, 'Sales_log')
    feat_df = feat_df.dropna().reset_index(drop=True)

    feature_cols = ['dow','month','year','week','Open','Promo','SchoolHoliday'] + \
                   [f'lag_{L}' for L in (1,7,14,28)] + [f'roll{w}' for w in (7,28)]

    y = feat_df['Sales_log'].values.astype('float32')
    X_num = feat_df[feature_cols].values.astype('float32')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    lookback = int(model_cfg['lookback'])
    horizon = int(model_cfg['horizon'])

    X_seq, y_seq = make_sequences_by_store(feat_df, X_scaled, y, lookback=lookback, horizon=horizon)

    # Splits
    n = len(X_seq)
    test_size = int(split_cfg['test_ratio'] * n)
    val_size = int(split_cfg['val_ratio'] * (n - test_size))

    X_train_full, y_train_full = X_seq[: n - test_size], y_seq[: n - test_size]
    X_test, y_test = X_seq[n - test_size :], y_seq[n - test_size :]

    X_train, y_train = X_train_full[: n - test_size - val_size], y_train_full[: n - test_size - val_size]
    X_val, y_val = X_train_full[n - test_size - val_size :], y_train_full[n - test_size - val_size :]

    # Model
    n_features = X_seq.shape[-1]
    model = build_model(
        units=int(model_cfg['units']), dropout=float(model_cfg['dropout']), lr=float(model_cfg['lr']),
        lookback=lookback, n_features=n_features, horizon=horizon
    )

    callbacks = [
        __import__('tensorflow').keras.callbacks.EarlyStopping(monitor='val_mae', patience=int(model_cfg['early_stopping_patience']), restore_best_weights=True),
        __import__('tensorflow').keras.callbacks.ModelCheckpoint(
            filepath=str(model_path).replace('.keras', '_best.keras'),
            save_best_only=True,
            monitor='val_mae',
            mode='min'
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=int(model_cfg['epochs']),
        batch_size=int(model_cfg['batch_size']),
        verbose=1,
        callbacks=callbacks
    )

    # Eval
    pred_test = model.predict(X_test, verbose=0)
    mae, rmse = mae_rmse_from_logs(y_test, pred_test)
    print({'test_mae': mae, 'test_rmse': rmse})

    # Plot
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.expm1(y_test.squeeze())
    y_pred = np.expm1(pred_test.squeeze())
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    plt.plot(y_true[:500], label='Actual')
    plt.plot(y_pred[:500], label='Predicted')
    plt.legend(); plt.title('Test set (first 500)'); plt.tight_layout()
    plt.savefig(plot_path, dpi=150)

    # Save artifacts
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    import joblib
    joblib.dump(scaler, scaler_path)

    import pandas as pd
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).to_csv(preds_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    main(args.config)
