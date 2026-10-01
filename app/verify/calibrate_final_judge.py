import os
import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
from xgboost import XGBClassifier
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR
from app.predictors.model_factory import ModelFactory

# Force CPU to avoid CUDA 303 locks during calibration
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def calibrate_judge_final(symbol):
    print(f"\n⚖️  FINAL CALIBRATION: {symbol}")
    try:
        data_path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
        df = pd.read_csv(data_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # 1. Indicator Injection
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.columns = [c.lower() for c in df.columns]

        features = []
        for feat in ['rsi', 'atr', 'adx']:
            actual_col = next((c for c in df.columns if feat in c and '_' in c), None)
            if actual_col:
                df[feat] = df[actual_col]
                features.append(feat)

        # 2. Expert Quorum Check
        expert_names = ["XGBoost", "RandomForest", "Transformer"]
        experts = {n: ModelFactory.load_model(n, symbol=symbol) for n in expert_names}
        
        if not all(experts.values()):
            missing = [n for n, m in experts.items() if m is None]
            print(f"  ❌ Missing Experts: {missing}")
            return False

        # 3. Nuclear History Scan
        lookback = 50
        votes_data, targets = [], []
        print(f"  📡 Collecting votes from {len(df)} bars...")

        # Step through history to find variety
        for i in range(lookback, len(df) - 1, 5): 
            # Deep Cast to prevent indexing errors
            state = np.ascontiguousarray(df.iloc[:i+1].tail(lookback)[features].values, dtype=np.float32)
            
            try:
                row_votes = [experts[name].predict_direction(state) for name in expert_names]
                actual_up = 1 if df.iloc[i+1]['close'] > df.iloc[i]['close'] else 0
                votes_data.append(row_votes)
                targets.append(actual_up)
            except:
                continue

        if not votes_data:
            print("  ⚠️  Failed to collect any valid votes.")
            return False

        X_meta, y_meta = np.array(votes_data), np.array(targets)
        unique, counts = np.unique(y_meta, return_counts=True)
        print(f"  📊 Distribution: {dict(zip(unique, counts))}")

        if len(unique) < 2:
            print(f"  ⚠️  Market still too one-sided for {symbol}.")
            return False

        # 4. Train the Judge (Meta-Model)
        # scale_pos_weight balances the Judge even if the market trended mostly one way
        spw = counts[0] / counts[1]
        judge = XGBClassifier(
            n_estimators=150,
            max_depth=3,
            scale_pos_weight=spw,
            learning_rate=0.05,
            objective='binary:logistic',
            random_state=42
        )
        judge.fit(X_meta, y_meta)

        # 5. Save the Stacking Expert
        save_path = os.path.join(MODEL_STORAGE_DIR, f"Stacking_{symbol}.joblib")
        joblib.dump(judge, save_path)
        print(f"✅ SUCCESS: {symbol} Judge is now ONLINE.")
        return True

    except Exception as e:
        print(f"  ❌ Calibration Error: {str(e)}")
        return False

if __name__ == "__main__":
    import glob
    os.makedirs(MODEL_STORAGE_DIR, exist_ok=True)
    files = glob.glob(os.path.join(DATA_DIR, "*-1h.csv"))
    symbols = [os.path.basename(f).replace("-1h.csv", "") for f in files]
    
    success_count = sum(1 for s in symbols if calibrate_judge_final(s))
    print(f"\n🏁 FINAL REPORT: {success_count}/{len(symbols)} Judges seated.")
