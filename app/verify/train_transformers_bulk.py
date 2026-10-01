import os
import glob
import pandas as pd
import pandas_ta as ta
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR

def build_simple_transformer(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs, outputs)

def train_transformer(symbol):
    print(f"🤖 TRAINING TRANSFORMER: {symbol}")
    data_path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
    df = pd.read_csv(data_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Feature Engineering
    df.ta.rsi(append=True); df.ta.atr(append=True); df.ta.adx(append=True)
    df.columns = [c.lower() for c in df.columns]
    features = ['rsi', 'atr', 'adx']
    for f in features:
        col = next((c for c in df.columns if f in c and '_' in f), None)
        if col: df[f] = df[col]
    
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    
    # Sequence Building
    lookback = 50
    X, y = [], []
    data = df[features].values
    target = df['target'].values
    
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(target[i])
        
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    model = build_simple_transformer((lookback, len(features)))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X, y, epochs=3, batch_size=32, verbose=0)
    
    save_path = os.path.join(MODEL_STORAGE_DIR, f"Transformer_{symbol}.keras")
    model.save(save_path)
    print(f"✅ Saved Transformer: {save_path}")

if __name__ == "__main__":
    os.makedirs(MODEL_STORAGE_DIR, exist_ok=True)
    symbols = [os.path.basename(f).replace("-1h.csv", "") for f in glob.glob(os.path.join(DATA_DIR, "*-1h.csv"))]
    for s in symbols:
        if not os.path.exists(os.path.join(MODEL_STORAGE_DIR, f"Transformer_{s}.keras")):
            train_transformer(s)
