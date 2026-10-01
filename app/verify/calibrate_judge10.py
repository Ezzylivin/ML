import os
import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
import tensorflow as tf
from xgboost import XGBClassifier
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR
from app.predictors.model_factory import ModelFactory

# Force CPU mode for stability
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

FEATURES_10 = [
    'open', 'high', 'low', 'close', 'rsi', 
    'atr', 'adx', 'adx_logic', 'atr_logic', 'sma_logic'
]

def engineer_10_features(df):
    """Identical feature engineering to ensure 100% alignment with experts."""
    df.columns = [c.strip().lower() for c in df.columns]
    # Indicator Logic
    df['rsi'] = df.ta.rsi(length=14).iloc[:]
    df['atr'] = df.ta.atr(length=14).iloc[:]
    df['adx'] = df.ta.adx(length=14).iloc[:, 0]
    df['sma_logic'] = df.ta.sma(length=200).iloc[:]
    df['adx_logic'] = df['adx']
    df['atr_logic'] = df['atr']
    return df.fillna(0)

def calibrate_10feature_judge(symbol):
    print(f"\n⚖️  CALIBRATING 10-FEATURE JUDGE: {symbol}")
    try:
        path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
        df = pd.read_csv(path)
        df_feats = engineer_10_features(df)
        
        # Expert Quorum Check
        experts = {n: ModelFactory.load_model(n, symbol=symbol) for n in ["XGBoost", "RandomForest", "Transformer"]}
        if not all(experts.values()): 
            print(f"  ❌ Missing Experts for {symbol}")
            return False

        votes_data, targets = [], []
        lookback = 50
        print(f"  📡 Gathering expert consensus from {len(df)} bars...")

        # Step through data to generate meta-features (the 'votes')
        for i in range(lookback, len(df) - 1, 10):
            # Pass a 10-feature slice to the Factory-wrapped experts
            state = df_feats.iloc[:i+1].tail(lookback)[FEATURES_10]
            try:
                # Factory handles the reshapes internally
                row_votes = [experts[name].predict_direction(state) for name in experts]
                actual_up = 1 if df.iloc[i+1]['close'] > df.iloc[i]['close'] else 0
                votes_data.append(row_votes)
                targets.append(actual_up)
            except:
                continue

        if not votes_data: return False

        X_meta, y_meta = np.array(votes_data), np.array(targets)
        unique, counts = np.unique(y_meta, return_counts=True)
        
        # Train Meta-Learner (XGBoost Judge)
        spw = counts[0] / counts[1] if len(unique) > 1 else 1.0
        judge = XGBClassifier(n_estimators=150, max_depth=3, scale_pos_weight=spw).fit(X_meta, y_meta)
        
        save_path = os.path.join(MODEL_STORAGE_DIR, f"Stacking_{symbol}.joblib")
        joblib.dump(judge, save_path)
        print(f"✅ SUCCESS: {symbol} Judge Seated.")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

if __name__ == "__main__":
    import glob
    symbols = [os.path.basename(f).replace("-1h.csv", "") for f in glob.glob(os.path.join(DATA_DIR, "*-1h.csv"))]
    for s in symbols: calibrate_10feature_judge(s)
