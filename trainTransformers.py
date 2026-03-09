import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR
from app.verify.engineer_and_train import apply_mega_features, create_strategic_labels

# 🟢 STABILITY: Disable GPU and noise
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train_transformer(symbol):
    print(f"\n🏗️  TRAINING TRANSFORMER-LSTM: {symbol}")
    path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
    
    try:
        # 1. Load & Engineer Features (Synchronized 25 features)
        raw_df = pd.read_csv(path)
        df, feats = apply_mega_features(raw_df)

        if len(df) < 500:
            print(f"   ⚠️  SKIPPED: {symbol} needs more history for sequences.")
            return False

        # 2. Strategic Labeling (1% TP/SL Engine)
        # We use the same engine as the Experts for Council Consensus
        df['target'] = create_strategic_labels(df, look_forward=24, tp=1.0, sl=1.0)
        
        # 3. Sequence Generation (50-hour lookback)
        lookback = 50
        X_raw = df[feats].values
        y_raw = df['target'].values
        
        X, y = [], []
        # We stop at len - 24 to avoid looking past the labels
        for i in range(lookback, len(X_raw) - 24):
            X.append(X_raw[i-lookback:i])
            y.append(y_raw[i])
            
        X, y = np.array(X), np.array(y)

        # 4. Define Architecture (FIXED SYNTAX)
        model = tf.keras.Sequential([
            # Added missing closing parenthesis here 👇
            tf.keras.layers.Input(shape=(lookback, 25)), 
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='swish'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # 5. EXECUTE TRAINING
        print(f"   🧠 Fitting Neural Layers (Epochs: 10)...")
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        
        # 6. Save the Brain
        save_name = f"{symbol.split('-')[0].lower()}_1h_transformer_model.keras"
        save_path = os.path.join(MODEL_STORAGE_DIR, save_name)
        
        model.save(save_path)
        print(f"   ✅ SUCCESS: {save_name} saved.")
        return True

    except Exception as e:
        print(f"   ❌ ERROR: {symbol} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    file_pattern = os.path.join(DATA_DIR, "*-1h.csv")
    files = glob.glob(file_pattern)
    symbols = [os.path.basename(f).replace("-1h.csv", "") for f in files]
    
    success_count = 0
    for s in symbols:
        if train_transformer(s):
            success_count += 1
            
    print(f"\n🏁 FINAL STATUS: {success_count}/{len(symbols)} Transformers Fully Trained.")
