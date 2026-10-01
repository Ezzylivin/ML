"""
train_judge.py — Stacking Meta-Model (The Judge) Training

============================================================
🔧 CRITICAL FIX: DATA LEAKAGE ELIMINATED
============================================================

OLD (LEAKED):
  1. Experts (XGB, RF, Transformer) trained on 100% of data
  2. Those same experts predict on the SAME data they memorized
  3. Judge trains on those artificially perfect predictions
  4. Result: Judge thinks experts are 80%+ accurate
  5. In production, experts see new data → real accuracy is ~55%
  6. Judge's calibration is completely wrong

NEW (HONEST):
  3-way time-based split:
  
  |------- 60% Expert Train -------|--- 20% Judge Train ---|--- 20% Test ---|
  
  1. Train TEMPORARY experts on first 60% only
  2. Those experts predict on the next 20% (data they NEVER saw)
  3. Judge trains on those HONEST out-of-sample predictions
  4. Evaluate everything on the final 20%
  5. Result: Judge learns how good experts REALLY are
  
  The production experts (from engineer_and_train.py) are trained
  on 80% of data. The judge was calibrated on honest predictions
  from experts trained on less data — this is conservative, which
  is exactly what you want. The judge won't over-trust.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

sys.path.append(os.getcwd())

from app.config2 import DATA_DIR, MODEL_STORAGE_DIR, FEATURE_COLUMNS
from app.verify.engineer_and_train import apply_mega_features, create_strategic_labels
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Disable GPU noise
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_temporary_experts(X_train, y_train, X_predict, n_features, lookback=50):
    """
    Train fresh expert models on X_train, then generate predictions on X_predict.
    These predictions are HONEST because X_predict was never seen during training.
    
    Returns:
        dict of {expert_name: prediction_array} on X_predict
    """
    opinions = {}
    
    # --- XGBoost ---
    print("   🌲 Training temporary XGBoost...")
    xgb = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, base_score=0.5)
    xgb.fit(X_train, y_train)
    opinions['xgboost'] = xgb.predict_proba(X_predict)[:, 1]
    train_acc = (xgb.predict(X_train) == y_train).mean() * 100
    print(f"      Train acc: {train_acc:.1f}%")
    
    # --- RandomForest ---
    print("   🌳 Training temporary RandomForest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10)
    rf.fit(X_train, y_train)
    opinions['randomforest'] = rf.predict_proba(X_predict)[:, 1]
    train_acc = (rf.predict(X_train) == y_train).mean() * 100
    print(f"      Train acc: {train_acc:.1f}%")
    
    # --- LSTM ("Transformer") ---
    print("   🧠 Training temporary LSTM...")
    
    # Build sequences from the TRAINING portion only
    X_seq_train, y_seq_train = [], []
    for i in range(lookback, len(X_train)):
        X_seq_train.append(X_train[i - lookback:i])
        y_seq_train.append(y_train[i])
    X_seq_train = np.array(X_seq_train, dtype='float32')
    y_seq_train = np.array(y_seq_train)
    
    # Normalize using training stats only
    flat = X_seq_train.reshape(-1, n_features)
    feat_mean = flat.mean(axis=0)
    feat_std = flat.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    
    X_seq_train_norm = (X_seq_train - feat_mean) / feat_std
    
    # Build and train
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation='swish'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=3, restore_best_weights=True
    )
    model.fit(X_seq_train_norm, y_seq_train, epochs=10, batch_size=32, verbose=0, callbacks=[early_stop])
    
    # ============================================================
    # Generate predictions on the JUDGE TRAIN portion
    # ============================================================
    # We need the last `lookback` rows of expert_train as context
    # for the first sequences in judge_train
    # Combine the tail of training data with the prediction data for windowing
    combined = np.concatenate([X_train[-(lookback):], X_predict], axis=0)
    
    X_seq_predict = []
    for i in range(lookback, len(combined)):
        X_seq_predict.append(combined[i - lookback:i])
    X_seq_predict = np.array(X_seq_predict, dtype='float32')
    
    # Normalize with same training stats
    X_seq_predict_norm = (X_seq_predict - feat_mean) / feat_std
    
    lstm_preds = model(tf.convert_to_tensor(X_seq_predict_norm), training=False).numpy()
    opinions['transformer'] = np.array([float(p[1]) for p in lstm_preds])
    
    print(f"      LSTM predictions generated: {len(opinions['transformer'])}")
    
    return opinions


def retrain_turbo_judge():
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "PEPE-USD"]
    
    print("\n" + "=" * 60)
    print("⚖️  STACKING JUDGE TRAINING — LEAKAGE-FREE")
    print("=" * 60)
    print(f"📁 Data: {DATA_DIR}")
    print(f"💾 Saving to: {MODEL_STORAGE_DIR}")
    print(f"📊 Features: {len(FEATURE_COLUMNS)}\n")
    
    for symbol in symbols:
        ticker = symbol.split('-')[0].lower()
        csv_path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
        if not os.path.exists(csv_path):
            print(f"⚪ Skipped {symbol}: CSV not found")
            continue
        
        try:
            print(f"\n⚖️  CALIBRATING JUDGE: {symbol}")
            print("-" * 40)
            
            # 1. Load and prepare data
            df_raw = pd.read_csv(csv_path)
            df, feats = apply_mega_features(df_raw)
            n_features = len(feats)
            
            # Strategic labels (same as expert training)
            df['target'] = create_strategic_labels(df, look_forward=24, tp=1.0, sl=1.0)
            
            X_all = df[feats].values
            y_all = df['target'].values
            
            # Drop last 24 rows (labels unreliable near end)
            X_all = X_all[:-24]
            y_all = y_all[:-24]
            
            if len(X_all) < 1000:
                print(f"   ⚠️ SKIPPED: Only {len(X_all)} rows — need at least 1000")
                continue
            
            # ============================================================
            # 2. THREE-WAY TIME SPLIT
            # ============================================================
            # |--- 60% Expert Train ---|--- 20% Judge Train ---|--- 20% Test ---|
            split_expert = int(len(X_all) * 0.60)
            split_judge = int(len(X_all) * 0.80)
            
            X_expert_train = X_all[:split_expert]
            y_expert_train = y_all[:split_expert]
            
            X_judge_train = X_all[split_expert:split_judge]
            y_judge_train = y_all[split_expert:split_judge]
            
            X_test = X_all[split_judge:]
            y_test = y_all[split_judge:]
            
            print(f"   📊 Split: {len(X_expert_train)} expert / {len(X_judge_train)} judge / {len(X_test)} test")
            print(f"   ⚖️ Class balance — Expert: {y_expert_train.mean()*100:.1f}% | Judge: {y_judge_train.mean()*100:.1f}% | Test: {y_test.mean()*100:.1f}%")
            
            # ============================================================
            # 3. Train temporary experts on 60%, predict on 20% (HONEST)
            # ============================================================
            print("\n   🎓 Phase 1: Training temporary experts on first 60%...")
            judge_opinions = train_temporary_experts(
                X_expert_train, y_expert_train,
                X_judge_train, n_features
            )
            
            # ============================================================
            # 🔧 FIX: All opinions aligned to same length
            # ============================================================
            # OLD: Transformer was padded with 0.5 for first 50 rows,
            #      XGB/RF predicted on all rows. Judge trained on misaligned data.
            # NEW: All three experts produce predictions for exactly
            #      X_judge_train rows. No padding, no misalignment.
            min_len = min(len(v) for v in judge_opinions.values())
            
            # Trim all to same length (from the end, to keep time alignment)
            for key in judge_opinions:
                judge_opinions[key] = judge_opinions[key][-min_len:]
            y_judge_aligned = y_judge_train[-min_len:]
            
            print(f"\n   📐 Aligned: {min_len} rows for judge training")
            
            # ============================================================
            # 4. Train the Judge on HONEST predictions
            # ============================================================
            print("\n   ⚖️ Phase 2: Training Judge on honest expert predictions...")
            
            X_meta = np.column_stack([
                judge_opinions['xgboost'],
                judge_opinions['randomforest'],
                judge_opinions['transformer']
            ])
            
            judge = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05)
            judge.fit(X_meta, y_judge_aligned)
            
            judge_train_acc = (judge.predict(X_meta) == y_judge_aligned).mean() * 100
            print(f"   📊 Judge Train Acc: {judge_train_acc:.1f}%")
            
            # ============================================================
            # 5. EVALUATE ON HELD-OUT TEST SET
            # ============================================================
            print("\n   🧪 Phase 3: Evaluating on unseen test data...")
            
            # The temporary experts also need to predict on test data
            # We retrain on expert_train + judge_train combined (first 80%)
            # to give the test evaluation the same conditions as production
            X_full_train = X_all[:split_judge]
            y_full_train = y_all[:split_judge]
            
            test_opinions = train_temporary_experts(
                X_full_train, y_full_train,
                X_test, n_features
            )
            
            # Align test predictions
            test_min_len = min(len(v) for v in test_opinions.values())
            for key in test_opinions:
                test_opinions[key] = test_opinions[key][-test_min_len:]
            y_test_aligned = y_test[-test_min_len:]
            
            X_meta_test = np.column_stack([
                test_opinions['xgboost'],
                test_opinions['randomforest'],
                test_opinions['transformer']
            ])
            
            judge_test_acc = (judge.predict(X_meta_test) == y_test_aligned).mean() * 100
            
            # Also check individual expert accuracy on test set
            print(f"\n   📈 TEST RESULTS (on truly unseen data):")
            for name in ['xgboost', 'randomforest', 'transformer']:
                expert_acc = ((test_opinions[name] > 0.5).astype(int) == y_test_aligned).mean() * 100
                print(f"      {name.ljust(14)}: {expert_acc:.1f}%")
            
            print(f"      {'JUDGE'.ljust(14)}: {judge_test_acc:.1f}%", end="")
            
            if judge_train_acc - judge_test_acc > 10:
                print(f" ⚠️ OVERFIT GAP: {judge_train_acc - judge_test_acc:.1f}%")
            else:
                print(f" ✅")
            
            # Sanity check: is the judge better than the best individual expert?
            best_expert_acc = max(
                ((test_opinions[n] > 0.5).astype(int) == y_test_aligned).mean() * 100
                for n in ['xgboost', 'randomforest', 'transformer']
            )
            if judge_test_acc > best_expert_acc:
                print(f"   🏆 Judge BEATS best expert by {judge_test_acc - best_expert_acc:.1f}%")
            else:
                print(f"   ⚠️ Judge UNDERPERFORMS best expert by {best_expert_acc - judge_test_acc:.1f}%")
                print(f"      Consider using equal-weight averaging instead.")
            
            # ============================================================
            # 6. Save the Judge
            # ============================================================
            expert_names = ['xgboost_score', 'randomforest_score', 'transformer_score']
            save_path = os.path.join(MODEL_STORAGE_DIR, f"{ticker}_1h_stacking_model.joblib")
            
            joblib.dump({
                "model": judge,
                "feature_names": expert_names,
                "is_meta_model": True
            }, save_path)
            
            print(f"\n   💾 JUDGE SAVED: {save_path}")
            print(f"   📊 Summary: Train {judge_train_acc:.1f}% → Test {judge_test_acc:.1f}%")
            
        except Exception as e:
            print(f"   ❌ FAILED {symbol}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    retrain_turbo_judge()
