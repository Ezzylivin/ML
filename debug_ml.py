import os
import sys
import pandas as pd
import numpy as np

# Set project root in path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.predictors.model_factory import ModelFactory
    from app.predictors.stacking_predictor import StackingPredictor
    from app.config2 import MODEL_STORAGE_DIR
    print("✅ Core ML modules imported successfully.")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def run_deep_audit():
    print(f"\n--- 📂 DIRECTORY AUDIT ---")
    print(f"Storage Path: {MODEL_STORAGE_DIR}")
    if not os.path.exists(MODEL_STORAGE_DIR):
        print(f"❌ ERROR: Storage directory not found at {MODEL_STORAGE_DIR}")
        return
    
    files = os.listdir(MODEL_STORAGE_DIR)
    print(f"Files found: {len(files)}")
    for f in files:
        if any(x in f.lower() for x in ['xgboost', 'rf', 'transformer', 'meta']):
            print(f"  - {f}")

    print(f"\n--- 🧪 COMPONENT LOADING TEST ---")
    symbol = "BTC-USD"
    # Testing individual experts
    experts = ["XGBoost", "RandomForest", "Transformer"]
    loaded_count = 0
    
    for e in experts:
        print(f"Attempting to load expert: {e}...")
        try:
            model = ModelFactory.load_model(e, symbol=symbol)
            if model is not None:
                print(f"  ✅ {e} LOADED SUCCESSFULLY.")
                loaded_count += 1
            else:
                print(f"  ❌ {e} RETURNED NONE.")
        except Exception as ex:
            print(f"  ❌ {e} CRASHED DURING LOAD: {ex}")

    print(f"\n--- ⚖️ STACKING JUDGE TEST ---")
    print("Attempting to initialize StackingPredictor...")
    try:
        predictor = StackingPredictor(symbol=symbol)
        if predictor is not None:
            print("  ✅ StackingPredictor instance created.")
            
            # Create dummy state data (50 candles, standard features)
            dummy_state = pd.DataFrame(np.random.randn(50, 10))
            print("Running test prediction...")
            prob = predictor.predict_direction(dummy_state)
            print(f"  🎯 PREDICTION RESULT: {prob:.4f}")
            
            if prob == 0.5:
                print("  ⚠️ WARNING: Prediction is exactly 0.5. Experts may be failing silently.")
        else:
            print("  ❌ StackingPredictor initialization returned None.")
    except Exception as ex:
        print(f"  ❌ StackingPredictor CRASHED: {ex}")

if __name__ == "__main__":
    run_deep_audit()
