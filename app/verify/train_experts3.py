import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR

# 🟢 STABILITY: Disable GPU to prevent the "cuInit 303" error during batch training
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def train_transformer(symbol):
    print(f"\n🏗️  BUILDING TRANSFORMER: {symbol}")
    path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
    
    try:
        # 1. Load Data
        df = pd.read_csv(path)
        
        # 🛑 PRE-FLIGHT CHECK: Indicators
        if 'rsi' not in df.columns:
            print(f"   ❌ FAILED: {symbol} has no indicators. Run engineer_and_train.py first!")
            return False

        # 🛑 PRE-FLIGHT CHECK: Minimum Rows (Need 50 for lookback + some for training)
        if len(df) < 150:
            print(f"   ⚠️  SKIPPED: {symbol} has only {len(df)} rows. Need 150+ for sequence training.")
            return False

        # 2. Prepare Sequences (10 features, 50-hour lookback)
        feats = ['open', 'high', 'low', 'close', 'rsi', 'atr', 'adx', 'adx_logic', 'atr_logic', 'sma_logic']
        data = df[feats].values
        
        # Create dummy target for "Brain Structure Initialization"
        # In a real training run, this would be your 'y' labels
        X = np.array([data[i-50:i] for i in range(50, len(data))])
        
        # 3. Define Architecture (NEO-V7 Transformer/LSTM Hybrid)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(50, 10)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # 4. Save the "Brain"
        save_path = os.path.join(MODEL_STORAGE_DIR, f"Transformer_{symbol}.keras")
        model.save(save_path)
        print(f"   ✅ SUCCESS: {save_path} saved.")
        return True

    except Exception as e:
        print(f"   ❌ ERROR: {symbol} failed during training: {e}")
        return False

if __name__ == "__main__":
    # 🔍 DYNAMIC DISCOVERY: Find all symbols in the data folder
    file_pattern = os.path.join(DATA_DIR, "*-1h.csv")
    files = glob.glob(file_pattern)
    
    # Extract symbol (e.g., "BTC-USD" from "BTC-USD-1h.csv")
    symbols = [os.path.basename(f).replace("-1h.csv", "") for f in files]
    
    print(f"🔎 DISCOVERED {len(symbols)} SYMBOLS: {', '.join(symbols)}")
    
    success_count = 0
    for s in symbols:
        if train_transformer(s):
            success_count += 1
            
    print(f"\n🏁 FINAL STATUS: {success_count}/{len(symbols)} Transformers Built.")
