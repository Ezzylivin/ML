import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from app.config2 import MODEL_STORAGE_DIR

def build_nn(input_dim):
    """Creates the Sequence Expert (Transformer-lite) with 4-feature input."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def rebuild():
    assets = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "SHIB", "SUI", "PEPE"]
    print(f"🏗️  REBUILDING COUNCIL in {MODEL_STORAGE_DIR}")

    for symbol in assets:
        try:
            csv_path = f"/root/Project/ML/data/{symbol}-USD-1h.csv"
            if not os.path.exists(csv_path): continue
            
            df = pd.read_csv(csv_path)
            # Feature Injection (V3 Standard)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
            df['atr'] = (df['high'] - df['low']).rolling(window=14).mean()
            df['vol_norm'] = df['volume'] / (df['volume'].rolling(window=20).mean() + 1e-9)
            df['target'] = (df['close'].shift(-1) > df['close'] + (df['atr'] * 0.5)).astype(int)
            df = df.fillna(0).tail(3000)

            # THE 4-FEATURE SYNC: [RSI, ATR, VOL, CLOSE]
            X = df[['rsi', 'atr', 'vol_norm', 'close']]
            y = df['target']
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)

            # 1. XGBoost
            xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05)
            xgb.fit(X_s, y)
            joblib.dump(xgb, os.path.join(MODEL_STORAGE_DIR, f"{symbol}-USD_xgboost.joblib"))

            # 2. RandomForest
            rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
            rf.fit(X_s, y)
            joblib.dump(rf, os.path.join(MODEL_STORAGE_DIR, f"{symbol}-USD_rf.joblib"))

            # 3. Transformer (Neural Network)
            nn = build_nn(4)
            nn.fit(X_s, y, epochs=20, batch_size=32, verbose=0)
            nn.save(os.path.join(MODEL_STORAGE_DIR, f"{symbol}-USD_transformer.keras"))
            
            # 4. Scaler
            joblib.dump(scaler, os.path.join(MODEL_STORAGE_DIR, f"{symbol}-USD_scaler.joblib"))
            
            print(f"✅ {symbol:5} | All Experts Synchronized (4 Features)")

        except Exception as e:
            print(f"❌ {symbol:5} | Failed: {e}")

if __name__ == "__main__":
    rebuild()
