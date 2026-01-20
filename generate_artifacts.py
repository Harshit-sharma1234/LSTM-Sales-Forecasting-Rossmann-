"""Quick script to generate the scaler and test predictions without retraining."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

from src.preprocessing.load_data import load_raw, merge_train_store, add_basic_clean
from src.features.feature_engineering import add_lags_and_rolls

# Load and process data
raw_dir = Path('data/raw')
train, test, store = load_raw(raw_dir)
df = merge_train_store(train, store)
df = add_basic_clean(df)

# Save processed data
df.to_csv('data/processed/rossmann_processed.csv', index=False)
print("Saved processed dataset.")

# Feature engineering
feat_df = add_lags_and_rolls(df, 'Sales_log')
feat_df = feat_df.dropna().reset_index(drop=True)

feature_cols = ['dow','month','year','week','Open','Promo','SchoolHoliday'] + \
               [f'lag_{L}' for L in (1,7,14,28)] + [f'roll{w}' for w in (7,28)]

X_num = feat_df[feature_cols].values.astype('float32')

# Create and save scaler
scaler = StandardScaler()
scaler.fit(X_num)
joblib.dump(scaler, 'models/feature_scaler.pkl')
print("Saved feature scaler.")

print("\n✅ Project setup complete! You can now run: streamlit run app/streamlit_app.py")
