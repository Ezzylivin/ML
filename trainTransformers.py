import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR

# 🟢 STABILITY: Disable GPU to prevent CUDA errors on CPU-only servers
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_labels(df, window=5):
    """
    Generates binary labels: 1 if price is higher in 'window' hours, else 0.
    """
    df['target'] = (df['close'].shift(-window) > df['close']).astype(int)
    return df

def train_transformer(symbol):
    print(f"\n🏗️  TRAINING TRANSFORMER-LSTM: {symbol}")
    path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
    
    try:
        # 1. Load Data
        df = pd.read_csv(path)
        
        # 🛑 PRE-FLIGHT CHECKS
        if 'rsi' not in df.columns:
            print(f"   ❌ FAILED: {symbol} has no indicators.")
            return False
        if len(df) < 200:
            print(f"   ⚠️  SKIPPED: {symbol} needs 200+ rows. Has {len(df)}.")
            return False

        # 2. Labeling & Feature Selection
        df = create_labels(df)
        feats = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_50', 'sma_200', 'ema_9', 'ema_21', 'ema_20',
            'rsi', 'atr', 'adx', 'st_trend', 
            'BBL_20_2.0_2.0', 'BBU_20_2.0_2.0', 
            'STOCHk_14_3_3', 'MACD_12_26_9', 'MACDs_12_26_9',
            'pa_high', 'pa_low', 'vol_ma', 
            'adx_logic', 'atr_logic', 'sma_logic'
        ]
        
        # Clean NaNs created by indicators/shifting
        from app.verify.engineer_and_train import apply_mega_features
        df, feats = apply_mega_features(pd.read_csv(path))
        
        lookback = 50
        X_raw = df[feats].values
        y_raw = (df['close'].shift(-5) > df['close']).astype(int).values # 5-hour target
        
        
      
        X, y = [], []
        for i in range(lookback, len(X_raw)-5):
            X.append(X_raw[i-lookback:i])
            y.append(y_raw[i])
            
        X, y = np.array(X), np.array(y)

        # 4. Define Architecture (Sovereign V7 Hybrid)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(lookback, 25),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='swish'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # 5. EXECUTE TRAINING (The missing link)
        print(f"   🧠 Fitting Neural Layers (Epochs: 10)...")
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        
        # 6. Save the Brain
        # Ensure we save with the naming convention the UI expects
        save_name = f"{symbol.split('-')[0].lower()}_1h_transformer_model.keras"
        save_path = os.path.join(MODEL_STORAGE_DIR, save_name)
        
        model.save(save_path)
        print(f"   ✅ SUCCESS: {save_path} certified and saved.")
        return True

    except Exception as e:
        print(f"   ❌ ERROR: {symbol} failed: {e}")
        return False

if __name__ == "__main__":
    # DYNAMIC DISCOVERY
    file_pattern = os.path.join(DATA_DIR, "*-1h.csv")
    files = glob.glob(file_pattern)
    symbols = [os.path.basename(f).replace("-1h.csv", "") for f in files]
    
    print(f"🔎 DISCOVERED {len(symbols)} SYMBOLS: {', '.join(symbols)}")
    
    success_count = 0
    for s in symbols:
        if train_transformer(s):
            success_count += 1
            
    print(f"\n🏁 FINAL STATUS: {success_count}/{len(symbols)} Transformers Fully Trained.")
