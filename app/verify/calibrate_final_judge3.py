import os
import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
from xgboost import XGBClassifier
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR
from app.predictors.model_factory import ModelFactory

# Force CPU to stabilize environment
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def calibrate_judge_labeled(symbol):
    print(f"\n🏷️  LABEL-AWARE CALIBRATION: {symbol}")
    try:
        data_path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
        df = pd.read_csv(data_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # 1. Indicator Injection (Must match the training features)
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
        if not all(experts.values()): 
            print(f"  ❌ Missing experts: {[k for k,v in experts.items() if v is None]}")
            return False

        # 3. History Scan
        lookback = 50
        votes_data, targets = [], []
        print(f"  📡 Scanning {len(df)} bars with Labeled Dataframes...")

        for i in range(lookback, len(df) - 1, 10):
            # ✅ THE FIX: Provide a labeled DataFrame slice instead of raw NumPy
            # This satisfies any internal expert code looking for 'rsi', 'atr', or 'adx'
            state_df = df.iloc[:i+1].tail(lookback)[features].copy()
            
            try:
                row_votes = []
                for name in expert_names:
                    # Some experts might still want NumPy, some want DF. 
                    # We try the DF first as the error suggests it's looking for labels.
                    try:
                        v = experts[name].predict_direction(state_df)
                    except:
                        v = experts[name].predict_direction(state_df.values)
                    row_votes.append(v)
                
                actual_up = 1 if df.iloc[i+1]['close'] > df.iloc[i]['close'] else 0
                votes_data.append(row_votes)
                targets.append(actual_up)
            except Exception:
                continue

        if not votes_data:
            print("  ⚠️  Zero votes collected. The experts are still rejecting the input format.")
            return False

        # 4. Meta-Model Training
        X_meta, y_meta = np.array(votes_data), np.array(targets)
        unique, counts = np.unique(y_meta, return_counts=True)
        print(f"  📊 Distribution: {dict(zip(unique, counts))}")

        if len(unique) < 2: return False

        spw = counts[0] / counts[1] if counts[1] > 0 else 1.0
        judge = XGBClassifier(n_estimators=100, scale_pos_weight=spw, learning_rate=0.05)
        judge.fit(X_meta, y_meta)

        save_path = os.path.join(MODEL_STORAGE_DIR, f"Stacking_{symbol}.joblib")
        joblib.dump(judge, save_path)
        print(f"✅ SUCCESS: {symbol} Judge Online.")
        return True

    except Exception as e:
        print(f"  ❌ Fatal Error: {str(e)}")
        return False

if __name__ == "__main__":
    import glob
    os.makedirs(MODEL_STORAGE_DIR, exist_ok=True)
    symbols = [os.path.basename(f).replace("-1h.csv", "") for f in glob.glob(os.path.join(DATA_DIR, "*-1h.csv"))]
    for s in symbols:
        calibrate_judge_labeled(s)
