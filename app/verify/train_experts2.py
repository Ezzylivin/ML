import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# Suppress TensorFlow GPU warnings - Force CPU for stability
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
MODEL_DIR = "data/saved_models/"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_all_symbols():
    """Scans data folder for unique symbols."""
    files = [f for f in os.listdir("data/") if f.endswith("-1h.csv")]
    return [f.replace("-1h.csv", "") for f in files]

def train_fleet():
    symbols = get_all_symbols()
    print(f"🚜 FOUND {len(symbols)} ASSETS IN FLEET. STARTING TRAINING...")

    for symbol in symbols:
        try:
            print(f"\n🚀 TRAINING COUNCIL FOR: {symbol}")
            
            # 1. Load Data
            df = pd.read_csv(f"data/{symbol}-1h.csv")
            df.columns = [c.lower().strip() for c in df.columns]
            
            # Target: 1 if next bar close is higher
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            features = df[['open', 'high', 'low', 'close', 'volume']].tail(2000).fillna(0)
            y = df['target'].tail(2000).values
            
            if len(y) < 500:
                print(f"⚠️  {symbol} skipped: Not enough data points.")
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features)
            
            # --- 🟢 XGBOOST ---
            xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
            xgb.fit(X_scaled, y)
            joblib.dump(xgb, f"{MODEL_DIR}{symbol}_xgboost.joblib")

            # --- 🔵 RANDOM FOREST ---
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(X_scaled, y)
            joblib.dump(rf, f"{MODEL_DIR}{symbol}_rf.joblib")

            # --- 🔴 LSTM (Fixed Input Layer) ---
            X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            model = Sequential([
                Input(shape=(1, 5)), # 🟢 Fixed Keras Warning
                LSTM(50),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy')
            model.fit(X_lstm, y, epochs=5, verbose=0)
            # 🟢 Using .keras format as recommended by logs
            model.save(f"{MODEL_DIR}{symbol}_lstm.keras") 
            joblib.dump(scaler, f"{MODEL_DIR}{symbol}_lstm_scaler.joblib")

            print(f"✅ COUNCIL SYNCED FOR {symbol}")

        except Exception as e:
            print(f"❌ FAILED {symbol}: {e}")

if __name__ == "__main__":
    train_fleet()
