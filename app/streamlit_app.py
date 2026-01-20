import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from tensorflow import keras

from src.inference.predict_next import predict_next


st.set_page_config(page_title="Rossmann LSTM Forecast", layout="wide")
st.title("Rossmann Sales Forecast — LSTM")
st.markdown("Use the controls to pick a store and forecast the next day's sales.")

processed_path = Path('data/processed/rossmann_processed.csv')
model_path = Path('models/rossmann_lstm_best.keras')
scaler_path = Path('models/feature_scaler.pkl')

if not processed_path.exists():
    st.error(f"Processed dataset not found at {processed_path}. Please run training first.")
    st.stop()

df = pd.read_csv(processed_path, parse_dates=['Date'])
stores = sorted(df['Store'].unique().tolist())

col1, col2, col3 = st.columns(3)
with col1:
    store_id = st.selectbox("Store", stores, index=0)
with col2:
    lookback = st.slider("Lookback window (days)", min_value=14, max_value=60, value=28, step=1)
with col3:
    run_btn = st.button("Predict next day")

st.divider()

if run_btn:
    try:
        pred = predict_next(int(store_id), int(lookback), processed_path, model_path, scaler_path)
        st.success(f"Predicted next-day sales for store {store_id}: {pred:,.0f}")
    except Exception as e:
        st.exception(e)

# Recent history plot for context
st.subheader("Recent history (last 180 days)")
g = df[df['Store'] == store_id].sort_values('Date').tail(180)
st.line_chart({
    "Actual Sales": g.set_index('Date')['Sales']
})



