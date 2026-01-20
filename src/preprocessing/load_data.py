import pandas as pd
import numpy as np
from pathlib import Path

def load_raw(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    train = pd.read_csv(raw_dir / 'train.csv', parse_dates=['Date'])
    test = pd.read_csv(raw_dir / 'test.csv', parse_dates=['Date'])
    store = pd.read_csv(raw_dir / 'store.csv')
    return train, test, store

def merge_train_store(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    df = train.merge(store, on='Store', how='left')
    df = df[(df['Open'] == 1) & (df['Sales'] > 0)].copy()
    return df

def add_basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df['dow'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Sales_log'] = np.log1p(df['Sales'])
    df = df.sort_values(['Store','Date']).reset_index(drop=True)
    return df

def save_processed(df: pd.DataFrame, processed_path: str | Path) -> None:
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    cols = ['Store','Date','Sales','Sales_log','Open','Promo','SchoolHoliday','dow','month','year','week']
    df[cols].to_csv(processed_path, index=False)
