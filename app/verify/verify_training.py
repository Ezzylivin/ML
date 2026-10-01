import os
import sys
import pandas as pd
import numpy as np
import logging

# Path Injector
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.predictors.model_factory import ModelFactory

def run_variance_check(symbol="BTC-USD"):
    print("\n" + "="*60)
    print(f"🕵️  AI COUNCIL VARIANCE TEST: {symbol}")
    print("="*60)

    # 1. Load REAL data to test against (last 100 bars)
    data_path = os.path.join(project_root, f"data/{symbol}-1h.csv")
    df = pd.read_csv(data_path)
    df.columns = [c.lower().strip() for c in df.columns]
    
    experts = ['XGBoost', 'LSTM', 'RandomForest', 'Transformer', 'Stacking']
    
    for e_type in experts:
        print(f"\n--- Testing {e_type} ---")
        try:
            predictor = ModelFactory.load_model(e_type, symbol=symbol)
            
            # We will test 5 different slices of history
            results = []
            for i in range(10, 15):
                state = df.iloc[i-10:i] # Feed small slices
                prob = predictor.predict_direction(state)
                results.append(round(prob, 4))
            
            # ANALYSIS
            unique_results = set(results)
            if len(unique_results) == 1 and list(unique_results)[0] == 0.5:
                print(f"❌ STATUS: ZOMBIE MODEL. Model returns static 0.5000.")
                print(f"   Reason: Model is likely untrained or file is missing.")
            elif len(unique_results) == 1:
                print(f"⚠️  STATUS: OVERFIT/STATIC. Returns constant {list(unique_results)[0]}.")
            else:
                print(f"✅ STATUS: LIVE. Unique predictions: {results}")
                print(f"   Variance: {max(results) - min(results):.4f}")

        except Exception as e:
            print(f"❌ CRITICAL: {e}")

    print("\n" + "="*60)
    print("✅ VARIANCE CHECK COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_variance_check()
