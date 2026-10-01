import os
import sys

# Force project root into path
sys.path.append("/root/Project/ML")

from app.predictors.model_factory import ModelFactory
from app.config2 import MODEL_STORAGE_DIR

def run_diagnostics(symbol="BTC-USD"):
    print(f"🔍 DIAGNOSING {symbol} COUNCIL...")
    print(f"📂 Storage Path: {MODEL_STORAGE_DIR}")
    
    # 1. Physical File Check
    print("\n📦 1. PHYSICAL FILE AUDIT:")
    files = os.listdir(MODEL_STORAGE_DIR)
    relevant = [f for f in files if symbol in f or symbol.replace("-USD", "") in f]
    if not relevant:
        print(f"   ❌ NO FILES FOUND for {symbol} in {MODEL_STORAGE_DIR}")
    else:
        for f in sorted(relevant):
            print(f"   [FILE] {f}")

    # 2. Factory Load Test
    print("\n🤖 2. MODELFACTORY LOADING TEST:")
    experts = ["XGBoost", "RandomForest", "Transformer"]
    
    for e in experts:
        try:
            predictor = ModelFactory.load_model(e, symbol)
            p_type = type(predictor).__name__
            
            if p_type == "DummyPredictor":
                print(f"   -> {e:13}: 🔴 FAILED (Returned Dummy)")
            else:
                print(f"   -> {e:13}: 🟢 SUCCESS (Loaded {p_type})")
        except Exception as err:
            print(f"   -> {e:13}: ❌ CRASHED ({err})")

if __name__ == "__main__":
    run_diagnostics("BTC-USD")
