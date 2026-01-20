import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
store_labels = [f"Store {s}" for s in stores]

col1, col2, col3 = st.columns(3)
with col1:
    store_label = st.selectbox("Store", store_labels, index=0)
    store_id = int(store_label.split()[-1])  # Extract number from "Store X"
with col2:
    # Model was trained with lookback=28, so this must be fixed
    lookback = 28
    st.info(f"Lookback window: {lookback} days (fixed)")
with col3:
    run_btn = st.button("Predict next day")

# Sidebar - Global Model Performance
st.sidebar.title("Model Metrics")
st.sidebar.markdown("Based on test set evaluation:")
st.sidebar.metric("Model Accuracy", "~90%")
st.sidebar.metric("Mean Log Error", "0.02")
st.sidebar.metric("Mean Error (MAE)", "€606")
st.sidebar.info("The model typically predicts within ±10% of the actual value.")

st.divider()

# Main Area - Store Context
store_data = df[df['Store'] == store_id]
avg_sales = store_data[store_data['Sales'] > 0]['Sales'].mean()
max_sales = store_data['Sales'].max()

st.subheader(f"Stats for Store {store_id}")
m1, m2, m3 = st.columns(3)
m1.metric("Average Daily Sales", f"€{avg_sales:,.0f}")
m2.metric("Max Daily Sales", f"€{max_sales:,.0f}")
m3.metric("Total Days Recorded", f"{len(store_data)}")

st.divider()

if run_btn:
    try:
        pred = predict_next(int(store_id), int(lookback), processed_path, model_path, scaler_path)
        
        # Calculate context
        diff_from_avg = ((pred - avg_sales) / avg_sales) * 100
        lower_bound = pred * 0.90
        upper_bound = pred * 1.10
        
        st.success(f"### Forecast: €{pred:,.0f}")
        
        c1, c2 = st.columns(2)
        c1.markdown(f"**Confidence Range (±10%):** \n €{lower_bound:,.0f} — €{upper_bound:,.0f}")
        
        if diff_from_avg > 0:
            c2.markdown(f"📈 **{diff_from_avg:.1f}% higher** than this store's average.")
        else:
            c2.markdown(f"📉 **{abs(diff_from_avg):.1f}% lower** than this store's average.")
            
    except Exception as e:
        st.exception(e)

# Recent history plot for context
st.subheader("Recent history (last 180 days)")
g = df[df['Store'] == store_id].sort_values('Date').tail(180)
st.line_chart({
    "Actual Sales": g.set_index('Date')['Sales']
})



