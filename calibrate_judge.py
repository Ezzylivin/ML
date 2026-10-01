import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR
from app.predictors.model_factory import ModelFactory

def create_stacking_expert(symbol):
    print(f"\n⚖️  CALIBRATING JUDGE FOR: {symbol}")
    
    data_path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
    if not os.path.exists(data_path):
        print(f"  ❌ Data file missing: {data_path}")
        return False
    
    df = pd.read_csv(data_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # 1. Load Experts
    expert_names = ["XGBoost", "RandomForest", "Transformer"]
    experts = {}
    for name in expert_names:
        model = ModelFactory.load_model(name, symbol=symbol)
        if model:
            experts[name] = model
        else:
            # Silently skip missing experts to avoid console clutter
            return False

    # 2. Deep Memory Collection
    # We increase the training window from 200 to 1500 bars to ensure class balance
    lookback = 50
    training_window = min(1500, len(df) - lookback - 1) 
    print(f"  📡 Collecting expert votes over {training_window} bars...")
    
    votes_data = []
    targets = []
    features = ['rsi', 'atr', 'adx']

    # Start from the end of the file and work backwards
    start_idx = len(df) - training_window - 1
    for i in range(start_idx, len(df) - 1):
        # Deep-Cast for model compatibility
        state_raw = df.iloc[:i+1].tail(lookback)[features].values
        state_numpy = np.ascontiguousarray(state_raw, dtype=np.float32)
        
        try:
            row_votes = [experts[name].predict_direction(state_numpy) for name in expert_names]
            votes_data.append(row_votes)
            
            # Ground Truth: 1 if price went up, 0 if down
            actual_up = 1 if df.iloc[i+1]['close'] > df.iloc[i]['close'] else 0
            targets.append(actual_up)
        except:
            continue

    # 3. Validation & Training
    X_meta = np.array(votes_data)
    y_meta = np.array(targets)
    
    # Check for both Buy and Sell examples
    if len(np.unique(y_meta)) < 2:
        print(f"  ⚠️  Still too one-sided after {training_window} bars. Skipping.")
        return False

    print(f"  🚀 Training Judge with {len(y_meta)} samples...")
    judge = XGBClassifier(
        n_estimators=100, # Increased for better ensemble learning
        max_depth=4, 
        learning_rate=0.03, 
        objective='binary:logistic',
        random_state=42
    )
    judge.fit(X_meta, y_meta)

    # 4. Save
    save_path = os.path.join(MODEL_STORAGE_DIR, f"Stacking_{symbol}.joblib")
    joblib.dump(judge, save_path)
    print(f"✅ SUCCESS: {symbol} Judge Calibrated.")
    return True

if __name__ == "__main__":
    os.makedirs(MODEL_STORAGE_DIR, exist_ok=True)
    import glob
    files = glob.glob(os.path.join(DATA_DIR, "*-1h.csv"))
    symbols = [os.path.basename(f).replace("-1h.csv", "") for f in files]
    
    success_count = sum(1 for s in symbols if create_stacking_expert(s))
    print(f"\n🏁 CALIBRATION COMPLETE: {success_count}/{len(symbols)} symbols ready.")
