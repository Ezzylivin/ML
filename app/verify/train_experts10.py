import os
import glob
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR

# 🟢 STABILITY: Force CPU mode 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 🟢 NEO-V7 STANDARD: Feature Order is Critical
FEATURES_10 = [
    'open', 'high', 'low', 'close', 'rsi', 
    'atr', 'adx', 'adx_logic', 'atr_logic', 'sma_logic'
]

def engineer_10_features(df):
    """Calculates the 10-feature set with absolute dimension safety."""
    df.columns = [c.strip().lower() for c in df.columns]
    
    # 1. Base OHLC
    # 2. Indicators (Explicit Capture to avoid multi-column lookup errors)
    # We use 'iloc[:, 0]' to force a Series if a DataFrame is returned
    df['rsi'] = df.ta.rsi(length=14).iloc[:] 
    df['atr'] = df.ta.atr(length=14).iloc[:]
    
    # ADX returns a DataFrame (ADX, DMI+, DMI-), we only want the first column
    df['adx'] = df.ta.adx(length=14).iloc[:, 0]
    
    # SMA 200
    df['sma_logic'] = df.ta.sma(length=200).iloc[:]
    
    # Logic Columns (Duplicates for NEO-V7 architecture consistency)
    df['adx_logic'] = df['adx']
    df['atr_logic'] = df['atr']
    
    # Fill gaps from indicator warmup
    df = df.fillna(0)
    
    return df[FEATURES_10].copy()

def train_experts_10(symbol):
    print(f"\n🏗️  NEO-V7 TRAINING: {symbol}")
    try:
        path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
        if not os.path.exists(path): return
        
        raw_df = pd.read_csv(path)
        
        # --- 🟢 STEP 1: ENGINEERING ---
        X_full = engineer_10_features(raw_df)
        
        # Ground Truth Labeling
        y = (raw_df['close'].shift(-1) > raw_df['close']).astype(int)
        
        # Syncing indices for a clean dropna()
        X = X_full.values
        y = y.values

        # --- 🟢 STEP 2: XGBOOST ---
        xgb_path = os.path.join(MODEL_STORAGE_DIR, f"XGBoost_{symbol}.joblib")
        joblib.dump(XGBClassifier(n_estimators=100, max_depth=5).fit(X, y), xgb_path)

        # --- 3. RANDOMFOREST ---
        rf_path = os.path.join(MODEL_STORAGE_DIR, f"RandomForest_{symbol}.joblib")
        joblib.dump(RandomForestClassifier(n_estimators=100, max_depth=10).fit(X, y), rf_path)

        # --- 4. TRANSFORMER (Input: 50, 10) ---
        t_path = os.path.join(MODEL_STORAGE_DIR, f"Transformer_{symbol}.keras")
        lookback = 50
        X_seq = np.array([X[i-lookback:i] for i in range(lookback, len(X))])
        y_seq = y[lookback:]
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(lookback, 10)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(X_seq, y_seq, epochs=2, batch_size=64, verbose=0)
        model.save(t_path)
        
        print(f"✅ SUCCESS: {symbol} 10-Feature Stack Ready.")

    except Exception as e:
        print(f"  ❌ FAILED {symbol}: {e}")

if __name__ == "__main__":
    os.makedirs(MODEL_STORAGE_DIR, exist_ok=True)
    files = glob.glob(os.path.join(DATA_DIR, "*-1h.csv"))
    symbols = [os.path.basename(f).replace("-1h.csv", "") for f in files]
    for s in symbols: train_experts_10(s)
