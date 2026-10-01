import os
import glob
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
from xgboost import XGBClassifier
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR

def train_base_xgboost(symbol):
    print(f"🏗️  TRAINING BASE XGBOOST: {symbol}")
    try:
        data_path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
        df = pd.read_csv(data_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # --- 🟢 STEP 1: GENERATE MISSING INDICATORS ---
        # We must create the features that the KeyError said were missing
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True)
        
        # Clean column names again as pandas_ta adds suffixes like _14
        df.columns = [c.lower() for c in df.columns]
        
        # --- 🟢 STEP 2: DYNAMIC FEATURE MAPPING ---
        # Map generated columns (e.g., 'rsi_14') to base names ('rsi')
        features = []
        for feat in ['rsi', 'atr', 'adx']:
            actual_col = next((c for c in df.columns if feat in c and '_' in c), None)
            if actual_col:
                df[feat] = df[actual_col]
                features.append(feat)
        
        if len(features) < 3:
            print(f"  ⚠️  Could not generate all indicators for {symbol}. Skipping.")
            return False

        # --- 🟢 STEP 3: TARGET LABELING & TRAINING ---
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df.dropna()

        X = df[features].values
        y = df['target'].values

        if len(np.unique(y)) < 2:
            print(f"  ⚠️  One-sided data for {symbol}. Skipping.")
            return False

        model = XGBClassifier(
            n_estimators=100, 
            max_depth=5, 
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=42
        )
        model.fit(X, y)
        
        # Save to MODEL_STORAGE_DIR (usually project/ML/data/saved_models)
        os.makedirs(MODEL_STORAGE_DIR, exist_ok=True)
        save_path = os.path.join(MODEL_STORAGE_DIR, f"XGBoost_{symbol}.joblib")
        joblib.dump(model, save_path)
        print(f"✅ Saved Base Expert: {save_path}")
        return True

    except Exception as e:
        print(f"  ❌ Failed {symbol}: {e}")
        return False

if __name__ == "__main__":
    # Discover and process all 1h CSVs
    files = glob.glob(os.path.join(DATA_DIR, "*-1h.csv"))
    symbols = [os.path.basename(f).replace("-1h.csv", "") for f in files]
    
    print(f"🔍 Found {len(symbols)} symbols to process.")
    
    success_count = 0
    for s in symbols:
        # Only train if model doesn't exist
        model_file = os.path.join(MODEL_STORAGE_DIR, f"XGBoost_{s}.joblib")
        if not os.path.exists(model_file):
            if train_base_xgboost(s):
                success_count += 1
        else:
            print(f"⏩ {s} already has an XGBoost expert.")

    print(f"\n🏁 BULK TRAINING COMPLETE: {success_count} new experts created.")
