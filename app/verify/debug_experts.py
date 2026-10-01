import os
import pandas as pd
import pandas_ta as ta
import numpy as np
import traceback
from app.config2 import DATA_DIR
from app.predictors.model_factory import ModelFactory

# Force CPU to stabilize
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def debug_single_symbol(symbol="BTC-USD"):
    print(f"🕵️‍♂️ DIAGNOSING EXPERT INPUTS FOR: {symbol}")
    
    # 1. Load and Prepare Data
    data_path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
    df = pd.read_csv(data_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Feature Engineering
    df.ta.rsi(append=True); df.ta.atr(append=True); df.ta.adx(append=True)
    df.columns = [c.lower() for c in df.columns]
    features = ['rsi', 'atr', 'adx']
    
    # Map technical names
    for f in features:
        col = next((c for c in df.columns if f in c and '_' in c), None)
        if col: df[f] = df[col]

    # 2. Test Expert Loading
    expert_name = "XGBoost" 
    print(f"🔄 Testing Expert: {expert_name}")
    expert = ModelFactory.load_model(expert_name, symbol=symbol)
    
    if not expert:
        print(f"❌ {expert_name} model file not found!")
        return

    # 3. Create Test States
    lookback = 50
    state_df = df.iloc[:lookback+1].tail(lookback)[features]
    state_np = state_df.values.astype(np.float32)

    # 🧪 TEST 1: NUMPY
    print(f"\n--- TEST 1: RAW NUMPY (Shape: {state_np.shape}) ---")
    try:
        res = expert.predict_direction(state_np)
        print(f"✅ Success with NumPy! Prob: {res}")
    except Exception:
        print("💥 NumPy Failed!")
        traceback.print_exc()

    # 🧪 TEST 2: DATAFRAME
    print(f"\n--- TEST 2: PANDAS DATAFRAME (Shape: {state_df.shape}) ---")
    try:
        res = expert.predict_direction(state_df)
        print(f"✅ Success with DataFrame! Prob: {res}")
    except Exception:
        print("💥 DataFrame Failed!")
        traceback.print_exc()

if __name__ == "__main__":
    debug_single_symbol()
