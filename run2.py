import joblib
import os
import numpy as np
import sys

# Force project root into path
sys.path.append("/root/Project/ML")

from app.predictors.stacking_predictor import StackingPredictor
from app.config2 import MODEL_STORAGE_DIR

def prove_mismatch(symbol="BTC-USD"):
    print(f"🕵️  INVESTIGATING {symbol} STACKING FAILURE...")
    
    # 1. Inspect the Predictor's Roster
    predictor = StackingPredictor(symbol=symbol)
    print(f"\n📋 BOT ROSTER (expert_types): {predictor.expert_types}")
    print(f"📏 Bot is trying to collect {len(predictor.expert_types)} votes.")

    # 2. Inspect the Judge's Brain
    meta_path = os.path.join(MODEL_STORAGE_DIR, f"MetaModel_{symbol}.joblib")
    judge = joblib.load(meta_path)
    expected = judge.n_features_in_
    print(f"\n⚖️  JUDGE BRAIN (Expected Inputs): {expected}")

    # 3. The "Smoking Gun" Comparison
    if len(predictor.expert_types) != expected:
        print("\n❌ MISMATCH DETECTED!")
        print(f"   The Bot is building a list of {len(predictor.expert_types)} probabilities.")
        print(f"   But the Judge can ONLY accept {expected} probabilities.")
        print("\n⚠️  Because of this, the Stacking Layer crashes silently and returns 0.5.")
    else:
        print("\n✅ No mismatch found here. (Check LSTM vs Transformer probabilities)")

if __name__ == "__main__":
    prove_mismatch("BTC-USD")
