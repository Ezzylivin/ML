import os
import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
from xgboost import XGBClassifier
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR
from app.predictors.model_factory import ModelFactory

def calibrate_judge_nuclear(symbol):
    print(f"\n☢️  NUCLEAR CALIBRATION: {symbol}")
    try:
        data_path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
        df = pd.read_csv(data_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # 1. Feature Generation (Ensuring experts have data)
        df.ta.rsi(append=True); df.ta.atr(append=True); df.ta.adx(append=True)
        df.columns = [c.lower() for c in df.columns]
        for f in ['rsi', 'atr', 'adx']:
            col = next((c for c in df.columns if f in c and '_' in c), None)
            if col: df[f] = df[col]

        # 2. Load Experts (Quorum Check)
        expert_names = ["XGBoost", "RandomForest", "Transformer"]
        experts = {n: ModelFactory.load_model(n, symbol=symbol) for n in expert_names}
        
        if not all(experts.values()): 
            missing = [n for n, m in experts.items() if m is None]
            print(f"  ❌ Missing Experts for {symbol}: {missing}")
            return False

        # 3. Full-History Scanning
        # We scan the entire file to find varied market conditions
        lookback, features = 50, ['rsi', 'atr', 'adx']
        votes_data, targets = [], []
        print(f"  📡 Scanning total history ({len(df)} bars) for expert variety...")
        
        for i in range(lookback, len(df) - 1, 10): # Step by 10 for global coverage speed
            state = np.ascontiguousarray(df.iloc[:i+1].tail(lookback)[features].values, dtype=np.float32)
            try:
                row_votes = [experts[n].predict_direction(state) for n in expert_names]
                actual_up = 1 if df.iloc[i+1]['close'] > df.iloc[i]['close'] else 0
                votes_data.append(row_votes)
                targets.append(actual_up)
            except: continue

        X_meta, y_meta = np.array(votes_data), np.array(targets)
        unique, counts = np.unique(y_meta, return_counts=True)

        if len(unique) < 2:
            print(f"  ⚠️  Critical Failure: Even full history is one-sided ({dict(zip(unique, counts))})")
            return False

        # 4. Train Meta-Model with Class Balancing
        print(f"  🚀 Training Stacking Judge with {len(y_meta)} samples...")
        spw = counts[0] / counts[1] if counts[1] > 0 else 1.0
        judge = XGBClassifier(
            n_estimators=150, 
            max_depth=3, 
            scale_pos_weight=spw, 
            learning_rate=0.05,
            objective='binary:logistic'
        )
        judge.fit(X_meta, y_meta)

        save_path = os.path.join(MODEL_STORAGE_DIR, f"Stacking_{symbol}.joblib")
        joblib.dump(judge, save_path)
        print(f"✅ SUCCESS: {symbol} Judge is now ONLINE.")
        return True

    except Exception as e:
        print(f"  ❌ Calibration Crash: {e}")
        return False

if __name__ == "__main__":
    import glob
    files = glob.glob(os.path.join(DATA_DIR, "*-1h.csv"))
    symbols = [os.path.basename(f).replace("-1h.csv", "") for f in files]
    
    success = 0
    for s in symbols:
        if calibrate_judge_nuclear(s): success += 1
    print(f"\n🏁 FINAL REPORT: {success}/{len(symbols)} Judges Calibrated and Saved.")
