# LSTM Sales Forecasting (Rossmann)

## Setup
1) Install deps:
```bash
pip install -r requirements.txt
```
2) Data: place Rossmann CSVs in `data/raw/` named `train.csv`, `test.csv`, `store.csv`.

## Quick start
- Process and EDA: run `notebooks/01_data_loading_eda.ipynb`
- Train LSTM: run `notebooks/02_lstm_modeling.ipynb` or CLI:
```bash
python -m src.training.train_lstm --config configs/default.yaml
```
- Inference (predict next day for a store):
```bash
python -m src.inference.predict_next --store_id 1 --lookback 28
```

## Project structure
- `data/raw/` raw CSVs
- `data/processed/` processed dataset
- `models/` saved model and scalers
- `reports/figures/` plots
- `src/` code modules

## Dashboard (Streamlit)
After training, launch an interactive dashboard:
```bash
streamlit run app/streamlit_app.py
```
This lets you pick a store, adjust lookback, and forecast next-day sales. Ensure `models/rossmann_lstm_best.keras` and `models/feature_scaler.pkl` exist (produced by training).



