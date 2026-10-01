import joblib
import glob
import os

# 🎯 Targets your specific naming convention: btc_1h_stacking_model.joblib
model_files = glob.glob("/root/Project/ML/app/models/btc_1h_*_model.joblib")

if not model_files:
    print("❌ No model files found. Check the path or naming pattern.")
else:
    for model_path in model_files:
        print(f"\n🔍 Checking: {os.path.basename(model_path)}")
        try:
            payload = joblib.load(model_path)
            # Handle cases where the model is wrapped in a dict
            model = payload['model'] if isinstance(payload, dict) else payload
            
            # 1. Check for Probability Support
            has_proba = hasattr(model, 'predict_proba')
            print(f"✅ Supports predict_proba: {has_proba}")

            # 2. Check Stacking Logic (if applicable)
            if "stacking" in model_path:
                # Check the final estimator (the 'brain' that makes the final call)
                final_est = getattr(model, 'final_estimator_', 'Not trained yet')
                print(f"🧩 Meta-Model: {type(final_est).__name__}")
                if hasattr(final_est, 'predict_proba'):
                    print(f"🗳️ Meta-Model Supports Probs: True")
                else:
                    print(f"⚠️ Meta-Model Supports Probs: False (This causes the 1.0/0.0 issue)")

        except Exception as e:
            print(f"❌ Load Error: {e}")
