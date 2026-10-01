import os
import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
import traceback
from xgboost import XGBClassifier
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR
from app.predictors.model_factory import ModelFactory

# Force CPU to bypass CUDA 303 locks
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def calibrate_judge_surgical(symbol):
    print(f"\n🔬 SURGICAL CALIBRATION: {symbol}")
    try:
        data_path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
        df = pd.read_csv(data_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # 1. Feature Prep
        df.ta.rsi(length=14, append=True); df.ta.atr(length=14, append=True); df.ta.adx(length=14, append=True)
        df.columns = [c.lower() for c in df.columns]
        features = []
        for feat in ['rsi', 'atr', 'adx']:
            actual_col = next((c for c in df.columns if feat in c and '_' in c), None)
            if actual_col:
                df[feat] = df[actual_col]
                features.append(feat)

        # 2. Expert Quorum
        expert_names = ["XGBoost", "RandomForest", "Transformer"]
        experts = {n: ModelFactory.load_model(n, symbol=symbol) for n in expert_names}
        if not all(experts.values()): return False

        # 3. History Scan with Error Exposure
        lookback = 50
        votes_data, targets = [], []
        
        # Step 1: Force a single prediction to catch the error
        sample_idx = lookback + 10
        state = np.ascontiguousarray(df.iloc[:sample_idx].tail(lookback)[features].values, dtype=np.float32)
        
        print(f"  🧪 Testing Expert Quorum on shape: {state.shape}")
        for name in expert_names:
            try:
                # Some models need (50, 3), some need (1, 50, 3). Let's see what happens.
                val = experts[name].predict_direction(state)
                print(f"    ✅ {name} Vote: {val}")
            except Exception as e:
                print(f"    ❌ {name} FAILED: {str(e)}")
                # CRITICAL: If the Transformer fails here, it's likely a shape issue
                # Try reshaping to (1, 50, 3) as a fallback
                try:
                    val = experts[name].predict_direction(state.reshape(1, lookback, 3))
                    print(f"    🔄 {name} Fixed with Reshape (1, 50, 3). Vote: {val}")
                except:
                    print(f"    💀 {name} Hard Failure.")

        # 4. Main Scan (Only if experts are responding)
        for i in range(lookback, len(df) - 1, 10):
            # Applying the reshape that Transformers usually require
            raw_state = df.iloc[:i+1].tail(lookback)[features].values.astype(np.float32)
            
            try:
                # We try both shapes to be safe across different expert types
                row_votes = []
                for name in expert_names:
                    try:
                        v = experts[name].predict_direction(raw_state)
                    except:
                        v = experts[name].predict_direction(raw_state.reshape(1, lookback, 3))
                    row_votes.append(v)
                
                actual_up = 1 if df.iloc[i+1]['close'] > df.iloc[i]['close'] else 0
                votes_data.append(row_votes)
                targets.append(actual_up)
            except:
                continue

        if not votes_data:
            print("  ⚠️  Zero votes collected after full scan.")
            return False

        # 5. Training
        X_meta, y_meta = np.array(votes_data), np.array(targets)
        unique, counts = np.unique(y_meta, return_counts=True)
        if len(unique) < 2: return False

        judge = XGBClassifier(n_estimators=100, scale_pos_weight=counts[0]/counts[1]).fit(X_meta, y_meta)
        joblib.dump(judge, os.path.join(MODEL_STORAGE_DIR, f"Stacking_{symbol}.joblib"))
        print(f"✅ SUCCESS: {symbol} Judge Online.")
        return True

    except Exception as e:
        print(f"  💥 Fatal Symbol Error: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    import glob
    symbols = [os.path.basename(f).replace("-1h.csv", "") for f in glob.glob(os.path.join(DATA_DIR, "*-1h.csv"))]
    for s in symbols: calibrate_judge_surgical(s)
