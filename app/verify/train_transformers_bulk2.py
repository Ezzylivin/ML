import os
import glob
import pandas as pd
import pandas_ta as ta
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR

# Force CPU to avoid the CUDA 303 error during bulk training
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def build_simple_transformer(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(64, activation='relu')(inputs)
    # Simple temporal processing
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs, outputs)

def train_transformer(symbol):
    print(f"🤖 TRAINING TRANSFORMER: {symbol}")
    try:
        data_path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
        df = pd.read_csv(data_path)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # 🟢 STEP 1: GENERATE INDICATORS
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True)
        
        # Clean names again to handle pandas_ta output
        df.columns = [c.lower() for c in df.columns]
        
        # 🟢 STEP 2: DYNAMIC MAPPING (The Fix for KeyError)
        # We find columns that CONTAIN our keywords but have suffixes
        features = []
        for feat_name in ['rsi', 'atr', 'adx']:
            actual_col = next((c for c in df.columns if feat_name in c and '_' in c), None)
            if actual_col:
                df[feat_name] = df[actual_col] # Map 'rsi_14' -> 'rsi'
                features.append(feat_name)
        
        if len(features) < 3:
            print(f"  ⚠️  Missing indicators for {symbol}. Found: {features}")
            return False

        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df.dropna()
        
        # 🟢 STEP 3: SEQUENCE BUILDING
        lookback = 50
        if len(df) < lookback + 100:
            print(f"  ⚠️  Insufficient data for {symbol}.")
            return False
            
        X, y = [], []
        data_values = df[features].values
        target_values = df['target'].values
        
        for i in range(lookback, len(df)):
            X.append(data_values[i-lookback:i])
            y.append(target_values[i])
            
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # 🟢 STEP 4: TRAINING
        model = build_simple_transformer((lookback, len(features)))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Low epochs for fast bulk expert creation
        model.fit(X, y, epochs=5, batch_size=64, verbose=0)
        
        save_path = os.path.join(MODEL_STORAGE_DIR, f"Transformer_{symbol}.keras")
        model.save(save_path)
        print(f"✅ Saved Transformer: {save_path}")
        return True

    except Exception as e:
        print(f"  ❌ Error training {symbol}: {str(e)}")
        return False

if __name__ == "__main__":
    os.makedirs(MODEL_STORAGE_DIR, exist_ok=True)
    files = glob.glob(os.path.join(DATA_DIR, "*-1h.csv"))
    symbols = [os.path.basename(f).replace("-1h.csv", "") for f in files]
    
    print(f"🔍 Starting Bulk Transformer Training for {len(symbols)} symbols...")
    success_count = 0
    for s in symbols:
        # Check if already exists to save time
        if not os.path.exists(os.path.join(MODEL_STORAGE_DIR, f"Transformer_{s}.keras")):
            if train_transformer(s):
                success_count += 1
        else:
            print(f"⏩ {s} already has a Transformer.")
            
    print(f"\n🏁 BULK TRAINING COMPLETE: {success_count} new Transformers created.")
