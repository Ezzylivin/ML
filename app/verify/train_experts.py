import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

MODEL_DIR = "data/saved_models/"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_all_experts(symbol="BTC-USD", timeframe="1h"):
    print(f"🚀 STARTING MASS RETRAINING: {symbol}")
    
    # 1. Load and Prepare Data
    df = pd.read_csv(f"data/{symbol}-{timeframe}.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Simple feature engineering for training
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    features = df[['open', 'high', 'low', 'close', 'volume']].tail(2000).fillna(0)
    y = df['target'].tail(2000).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # --- 🟢 TRAIN XGBOOST ---
    print(" - Training XGBoost...")
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
    xgb.fit(X_scaled, y)
    joblib.dump(xgb, f"{MODEL_DIR}{symbol}_xgboost.joblib")

    # --- 🔵 TRAIN RANDOM FOREST ---
    print(" - Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_scaled, y)
    joblib.dump(rf, f"{MODEL_DIR}{symbol}_rf.joblib")

    # --- 🔴 TRAIN LSTM ---
    print(" - Training LSTM (Sequential)...")
    # Reshape for LSTM: (samples, time_steps, features)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    model = Sequential([
        LSTM(50, input_shape=(1, 5)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X_lstm, y, epochs=5, verbose=0)
    model.save(f"{MODEL_DIR}{symbol}_lstm.h5")
    joblib.dump(scaler, f"{MODEL_DIR}{symbol}_lstm_scaler.joblib")

    print(f"✅ ALL EXPERTS TRAINED FOR {symbol}")

if __name__ == "__main__":
    train_all_experts()
