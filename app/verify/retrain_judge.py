import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR
from app.predictors.model_factory import ModelFactory

def create_stacking_expert(symbol):
    """
    The 'Judge' calibration: Trains a Meta-Model to learn how to combine 
    XGBoost, RandomForest, and Transformer predictions into one signal.
    """
    print(f"\n⚖️  CALIBRATING JUDGE FOR: {symbol}")
    
    # 1. Load the Engineered Data
    data_path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
    if not os.path.exists(data_path):
        print(f"  ❌ Data file missing: {data_path}")
        return False
    
    df = pd.read_csv(data_path)
    
    # ✅ Clean headers to match engine expectations
    df.columns = [c.strip().lower() for c in df.columns]

    # 2. Collect Experts
    expert_names = ["XGBoost", "RandomForest", "Transformer"]
    experts = {}
    for name in expert_names:
        model = ModelFactory.load_model(name, symbol=symbol)
        if model:
            experts[name] = model
        else:
            print(f"  ❌ Expert '{name}' missing. Run training scripts first.")
            return False

    # 3. Generate Meta-Features (Historical 'Votes')
    print(f"  📡 Collecting expert votes over 200 bars...")
    votes_data = []
    targets = []
    
    # Features exactly as required by the experts
    features = ['rsi', 'atr', 'adx']
    lookback = 50 

    for i in range(len(df) - 201, len(df) - 1):
        # ✅ THE FIX: Deep-Cast the state to a pure NumPy array
        # Instead of 'state_df = df.iloc[:i+1]', we slice only the features and cast to float32 values.
        state_raw = df.iloc[:i+1].tail(lookback)[features].values
        state_numpy = np.ascontiguousarray(state_raw, dtype=np.float32)
        
        # Experts now receive a clean numeric grid
        try:
            row_votes = [experts[name].predict_direction(state_numpy) for name in expert_names]
            votes_data.append(row_votes)
            
            # Ground Truth: Did price actually go up in the next hour?
            actual_up = 1 if df.iloc[i+1]['close'] > df.iloc[i]['close'] else 0
            targets.append(actual_up)
        except Exception as e:
            # Silently skip bars that have insufficient data for lookback
            continue

    # 4. Train the Stacking Meta-Model
    X_meta = np.array(votes_data)
    y_meta = np.array(targets)
    
    if len(np.unique(y_meta)) < 2:
        print(f"  ⚠️  Skipping {symbol}: Market was too one-sided for calibration.")
        return False

    print(f"  🚀 Training Judge (XGBoost Meta-Learner)...")
    judge = XGBClassifier(
        n_estimators=50, 
        max_depth=3, 
        learning_rate=0.05, 
        base_score=0.5,
        objective='binary:logistic'
    )
    judge.fit(X_meta, y_meta)

    # 5. Save the Stacking Expert
    save_path = os.path.join(MODEL_STORAGE_DIR, f"Stacking_{symbol}.joblib")
    joblib.dump(judge, save_path)
    print(f"✅ SUCCESS: Stacking Expert saved at {save_path}")
    return True

if __name__ == "__main__":
    os.makedirs(MODEL_STORAGE_DIR, exist_ok=True)
    import glob
    files = glob.glob(os.path.join(DATA_DIR, "*-1h.csv"))
    symbols = [os.path.basename(f).replace("-1h.csv", "") for f in files]
    
    success_count = 0
    for s in symbols:
        if create_stacking_expert(s):
            success_count += 1
            
    print(f"\n🏁 JUDGE CALIBRATION COMPLETE: {success_count}/{len(symbols)} symbols ready.")
