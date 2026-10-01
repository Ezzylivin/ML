import os
import glob
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR

def train_missing(symbol):
    print(f"🏗️  PROCESSING BASE EXPERTS: {symbol}")
    data_path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
    df = pd.read_csv(data_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Generate indicators
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.columns = [c.lower() for c in df.columns]
    
    features = []
    for f in ['rsi', 'atr', 'adx']:
        col = next((c for c in df.columns if f in c and '_' in c), None)
        if col: 
            df[f] = df[col]
            features.append(f)

    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    X, y = df[features].values, df['target'].values

    if len(np.unique(y)) < 2: return

    # Train XGBoost if missing
    xgb_path = os.path.join(MODEL_STORAGE_DIR, f"XGBoost_{symbol}.joblib")
    if not os.path.exists(xgb_path):
        m = XGBClassifier(n_estimators=100, max_depth=5).fit(X, y)
        joblib.dump(m, xgb_path)
        print(f"  ✅ Saved XGBoost")

    # Train RandomForest if missing
    rf_path = os.path.join(MODEL_STORAGE_DIR, f"RandomForest_{symbol}.joblib")
    if not os.path.exists(rf_path):
        m = RandomForestClassifier(n_estimators=100, max_depth=10).fit(X, y)
        joblib.dump(m, rf_path)
        print(f"  ✅ Saved RandomForest")

if __name__ == "__main__":
    symbols = [os.path.basename(f).replace("-1h.csv", "") for f in glob.glob(os.path.join(DATA_DIR, "*-1h.csv"))]
    for s in symbols: train_missing(s)
