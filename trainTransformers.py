import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf

# ============================================================
# 🔧 FIX #1: Import from config2 and use existing functions
# ============================================================
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR, FEATURE_COLUMNS
from app.verify.engineer_and_train import apply_mega_features, create_atr_labels

# 🟢 STABILITY: Disable GPU and noise
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_transformer(symbol):
    print(f"\n🏗️  TRAINING LSTM MODEL: {symbol}")
    # ============================================================
    # 🔧 FIX (naming): This is an LSTM, not a Transformer.
    # Transformers use self-attention + positional encoding.
    # This uses stacked LSTM layers. The filename stays "transformer"
    # for backward compatibility with ModelFactory/StackingPredictor,
    # but the print statements are honest now.
    # ============================================================
    
    path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
    if not os.path.exists(path):
        print(f"   ⚠️  SKIPPED: {path} not found")
        return False

    try:
        # 1. Load & Engineer Features
        raw_df = pd.read_csv(path)
        df, feats = apply_mega_features(raw_df)
        
        if len(df) < 500:
            print(f"   ⚠️  SKIPPED: {symbol} needs more history ({len(df)} rows).")
            return False

        # 2. Strategic Labeling (matches expert training)
        df['target'] = create_atr_labels(df, look_forward=24, tp_atr=3.0, sl_atr=1.5)  # FIX #4: align to bot exits
        
        # 3. Sequence Generation
        lookback = 50
        # ============================================================
        # 🔧 FIX #2: Use len(feats) instead of hardcoded 25
        # ============================================================
        n_features = len(feats)
        X_raw = df[feats].values
        y_raw = df['target'].values
        
        X, y = [], []
        # Stop at len - 24 to avoid unreliable labels near the end
        for i in range(lookback, len(X_raw) - 24):
            X.append(X_raw[i - lookback:i])
            y.append(y_raw[i])
            
        X, y = np.array(X), np.array(y)
        
        print(f"   📊 Sequences: {len(X)} | Features: {n_features} | Class balance: {y.mean()*100:.1f}% positive")

        # ============================================================
        # 🔧 FIX #3: Time-based train/test split (80/20)
        # ============================================================
        # OLD: Trained on 100% of sequences. No way to detect overfitting.
        # NEW: First 80% for training, last 20% for evaluation.
        split_idx = int(len(X) * 0.80)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"   📊 Split: {len(X_train)} train / {len(X_test)} test")

        # ============================================================
        # 🔧 FIX #4: Per-feature normalization
        # ============================================================
        # OLD: Raw values fed directly to LSTM. BTC price (~60,000)
        #      sitting next to RSI (0-100) and binary flags (0/1).
        #      LSTMs are very sensitive to input scale — gradients
        #      get dominated by the largest-magnitude features.
        # NEW: Compute mean/std from TRAINING data only (no leakage),
        #      apply to both train and test.
        #      We reshape to 2D for stats, then back to 3D.
        
        # Flatten to (n_samples * lookback, n_features) for stats
        train_flat = X_train.reshape(-1, n_features)
        feat_mean = train_flat.mean(axis=0)
        feat_std = train_flat.std(axis=0)
        
        # Prevent division by zero for constant features
        feat_std[feat_std == 0] = 1.0
        
        # Normalize both sets using TRAINING statistics only
        X_train = (X_train - feat_mean) / feat_std
        X_test = (X_test - feat_mean) / feat_std
        
        # Convert to float32 for TF
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        
        print(f"   🔧 Normalized: mean range [{feat_mean.min():.1f}, {feat_mean.max():.1f}] → [0, 0]")

        # ============================================================
        # 🔧 FIX #5: Output layer produces proper 2-class probabilities
        # ============================================================
        # OLD: Dense(1, activation='sigmoid')
        #      Output shape: (batch, 1)
        #      RawModelAdapter did: preds[0][1] → IndexError
        #      Only worked because of the fallback: preds[0][0]
        # NEW: Dense(2, activation='softmax')
        #      Output shape: (batch, 2) → [prob_down, prob_up]
        #      preds[0][1] now correctly gives the "up" probability.
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(lookback, n_features)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='swish'),
            tf.keras.layers.Dense(2, activation='softmax')  # 🔧 FIX: was Dense(1, sigmoid)
        ])
        
        # ============================================================
        # 🔧 FIX #5b: Loss function matches output layer
        # ============================================================
        # OLD: binary_crossentropy (for single sigmoid output)
        # NEW: sparse_categorical_crossentropy (for 2-class softmax)
        #      "sparse" means y can stay as integers [0, 1] — no one-hot needed.
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 5. EXECUTE TRAINING
        print(f"   🧠 Fitting LSTM Layers (Epochs: 15, Early Stop patience: 3)...")
        
        # ============================================================
        # 🔧 FIX #6: Early stopping to prevent overfitting
        # ============================================================
        # OLD: Fixed 10 epochs, no monitoring.
        # NEW: Up to 15 epochs, stops early if validation loss plateaus.
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=15,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        # 6. Evaluate
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        stopped_epoch = len(history.history['loss'])
        
        print(f"   📈 Stopped at epoch {stopped_epoch}")
        print(f"   📊 Train Acc: {train_acc*100:.1f}% | Test Acc: {test_acc*100:.1f}%", end="")
        
        if train_acc - test_acc > 0.10:
            print(f" ⚠️ OVERFIT GAP: {(train_acc-test_acc)*100:.1f}%")
        else:
            print(f" ✅")
        
        # 7. Save the model
        save_name = f"{symbol.split('-')[0].lower()}_1h_transformer_model.keras"
        save_path = os.path.join(MODEL_STORAGE_DIR, save_name)
        model.save(save_path)
        
        # ============================================================
        # 🔧 FIX #7: Save normalization stats alongside model
        # ============================================================
        # The model expects normalized input. At inference time,
        # we need the same mean/std that were used during training.
        # Save them as a companion .npz file.
        norm_path = os.path.join(MODEL_STORAGE_DIR, f"{symbol.split('-')[0].lower()}_1h_transformer_norm.npz")
        np.savez(norm_path, mean=feat_mean, std=feat_std)
        
        print(f"   ✅ SUCCESS: {save_name} + normalization stats saved.")
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
    
    print(f"🔍 Found {len(symbols)} symbols: {symbols}")
    print(f"📁 Saving to: {MODEL_STORAGE_DIR}")
    print(f"📊 Features: {len(FEATURE_COLUMNS)}")
    
    success_count = 0
    for s in symbols:
        if train_transformer(s):
            success_count += 1
            
    print(f"\n🏁 FINAL STATUS: {success_count}/{len(symbols)} LSTM Models Trained.")
