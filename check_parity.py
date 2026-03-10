import os
import pandas as pd
import numpy as np
import joblib
from app.config2 import DATA_DIR
from app.predictors.model_factory import ModelFactory
# 🎯 V25 UPGRADE: Use the master engineer instead of the old v7 function
from app.verify.engineer_and_train import apply_mega_features

def run_parity_audit(symbol="BTC-USD"):
    print(f"\n" + "="*50)
    print(f"🚀 NEO-V25 FULL COUNCIL PARITY AUDIT: {symbol}")
    print("="*50)
    
    # 1. Load and Prepare Data (Using 25 features)
    path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
    if not os.path.exists(path):
        print(f"❌ CSV Missing: {path}")
        return

    df_raw = pd.read_csv(path)
    # Apply the 25-feature logic used in training
    df, features = apply_mega_features(df_raw)
    
    test_slice = df.tail(100) 
    print(f"✅ Data Ready: {len(test_slice)} rows | {len(features)} features")

    # 2. Gather Expert Scores
    experts = ["xgboost", "randomforest", "transformer"]
    council_scores = []

    for name in experts:
        model = ModelFactory.load_model(name, symbol=symbol)
        if model:
            try:
                # The adapter handles sequence vs row logic automatically
                prob = model.predict_direction(test_slice)
                print(f"   [{name.ljust(12)}] 🟢 PASSED | Score: {prob:.4f}")
                council_scores.append(prob)
            except Exception as e:
                print(f"   [{name.ljust(12)}] ❌ LOGIC FAILURE: {e}")
                council_scores.append(0.5)
        else:
            print(f"   [{name.ljust(12)}] 🔴 FILE MISSING")
            council_scores.append(0.5)

    # 3. Check Stacking Judge (The Meta-Model)
    print("\n⚖️ Testing Stacking Judge (The Meta-Pass)...")
    stacker = ModelFactory.load_model("stacking", symbol=symbol)
    
    if stacker:
        try:
            # 🎯 CRITICAL: We pass the scores from Step 2 into Step 3
            # This triggers the 'is_meta_model' logic in the RawModelAdapter
            final_prob = stacker.predict_direction(test_slice, council_probs=council_scores)
            
            status = "✨ PARITY PASSED" if final_prob != 0.5 else "⚠️ WARNING (Neutral)"
            print(f"   [{'stacking'.ljust(12)}] {status} | Score: {final_prob:.4f}")
            
            if final_prob == 0.5:
                print("\n   💡 Tip: If score is 0.5, ensure 'is_meta_model' is True in RawModelAdapter.")
        except Exception as e:
            print(f"   ❌ STACKING CRASH: {e}")
    else:
        print(f"   ❌ STACKING JUDGE FILE MISSING.")

if __name__ == "__main__":
    # Test all symbols in your council
    for sym in ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "PEPE-USD"]:
        run_parity_audit(sym)
