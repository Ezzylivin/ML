import os
import sys
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Ensure project root is in path
sys.path.append("/root/Project/ML")

from app.predictors.model_factory import ModelFactory
from app.predictors.stacking_predictor import StackingPredictor
from app.config2 import MODEL_STORAGE_DIR

def run_sovereign_audit():
    symbol = "BTC-USD"
    print("\n" + "="*60)
    print("🛡️  NEO-V7 SOVEREIGN SYSTEM HEALTH REPORT (FIXED)")
    print("="*60)

    # 1. Council Roster (Strictly 3 Experts)
    experts = ['XGBoost', 'RandomForest', 'Transformer']
    results = {}
    
    print(f"\n⚖️  2. COUNCIL OF EXPERTS AUDIT ({symbol}):")
    
    # Load a small sample of data for the audit
    try:
        df = pd.read_csv(f"/root/Project/ML/data/{symbol}-1h.csv").tail(50)
    except:
        print("❌ Data file missing. Cannot audit.")
        return

    for e_name in experts:
        try:
            model = ModelFactory.load_model(e_name, symbol)
            prob = model.predict_direction(df)
            results[e_name] = prob
            print(f" - {e_name:15}: 🟢 ONLINE  | Prob: {prob:.4f}")
        except Exception as e:
            print(f" - {e_name:15}: 🔴 FAILED  | Error: {e}")
            results[e_name] = 0.5

    # 2. Stacking Judge Audit
    print(f"\n⚖️  3. JUDGE ALIGNMENT CHECK:")
    try:
        predictor = StackingPredictor(symbol=symbol)
        stack_prob = predictor.predict_direction(df)
        
        # Check if the Judge file exists and matches the roster
        if predictor.judge is not None:
            expected = predictor.judge.n_features_in_
            if expected == len(experts):
                print(f" - Stacking Judge: 🟢 ONLINE  | Expects {expected} experts.")
                print(f" - Stacking Verdict: 🟢 ACTIVE  | Combined Prob: {stack_prob:.4f}")
            else:
                print(f" - Stacking Judge: 🟡 MISMATCH | Expects {expected}, but Roster has {len(experts)}.")
                print(f" - Stacking Verdict: 🔴 OFFLINE (Safe 0.5)")
        else:
            print(" - Stacking Judge: 🔴 NOT FOUND")
    except Exception as e:
        print(f" - Stacking Audit Failed: {e}")

    print("\n" + "="*60)
    print("✅ AUDIT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_sovereign_audit()
