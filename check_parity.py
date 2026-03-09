import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
from app.config2 import MODEL_STORAGE_DIR, DATA_DIR
from app.predictors.model_factory import ModelFactory


def apply_v7_features(df):
    """Parity Feature Engineering: Matches backtest and training exactly."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    # Core 10 Features
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_logic'] = (df['atr'] / df['close']) * 1000
    
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx_df['ADX_14']
    df['adx_logic'] = np.where(df['adx'] > 25, 1, 0)
    
    df['sma_200'] = ta.sma(df['close'], length=200)
    df['sma_logic'] = np.where(df['close'] > df['sma_200'], 1, -1)
    
    # List of 10 features in strict order
    features = ['open', 'high', 'low', 'close', 'rsi', 'atr', 'adx', 'adx_logic', 'atr_logic', 'sma_logic']
    return df.dropna(subset=features), features

def run_parity_audit(symbol="BTC-USD"):
    print(f"\n⚖️⚖️⚖️ LOGIC PARITY AUDIT: {symbol} ⚖️⚖️⚖️")
    
    # 1. Load and Prepare Data
    path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
    df_raw = pd.read_csv(path)
    df, features = apply_v7_features(df_raw)
    
    # Take a 60-row slice to test both 50-lookback and 1-row logic
    test_slice = df.tail(60) 
    print(f"📊 Testing with {len(test_slice)} rows and {len(features)} numeric features.")

    # 2. Load the Council via Factory
    experts = ["XGBoost", "RandomForest", "Transformer"]
    results = {}

    for name in experts:
        print(f"Attempting {name}...")
        model = ModelFactory.load_model(name, symbol=symbol)
        if model:
            try:
                # This calls the 'bridge' we built in ModelFactory
                prob = model.predict_direction(test_slice)
                results[name] = prob
                print(f"   ✅ {name} Prediction: {prob:.4f}")
            except Exception as e:
                print(f"   ❌ {name} LOGIC FAILURE: {e}")
        else:
            print(f"   ❌ {name} FILE MISSING.")

    # 3. Check Stacking Judge (The MetaModel)
    print("\n⚖️ Testing Stacking Judge...")
    stacker = ModelFactory.load_model("stacking", symbol=symbol)
    if stacker:
        try:
            final_prob = stacker.predict_direction(test_slice)
            print(f"   🎯 FINAL STACKED PROBABILITY: {final_prob:.4f}")
            
            if final_prob == 0.5:
                print("   ⚠️ WARNING: Result is exactly 0.5. Check if MetaModel is failing.")
            else:
                print("   ✨ PARITY PASSED: Council is fully synchronized.")
        except Exception as e:
            print(f"   ❌ STACKING CRASH: {e}")

if __name__ == "__main__":
    run_parity_audit("BTC-USD")
(venv) root@intelligent-mendel:~/Project/ML# 
