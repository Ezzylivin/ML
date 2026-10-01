import joblib
import os
import sys

MODEL_PATH = "/root/Project/ML/app/models/btc_4h_mlpclassifier_model.joblib"

print(f"🔍 Inspecting: {MODEL_PATH}")

try:
    # Load the file
    data = joblib.load(MODEL_PATH)
    
    if isinstance(data, dict):
        print("\n📦 It IS a Dictionary containing these keys:")
        print("-" * 40)
        for key in data.keys():
            print(f" 🔑 {key}")
        print("-" * 40)
        
        # Check if one of them looks like a model
        for k in ['model', 'classifier', 'pipeline', 'estimator', 'clf']:
            if k in data:
                print(f"\n✅ FOUND IT! The model is inside data['{k}']")
                model = data[k]
                if hasattr(model, "classes_"):
                    print(f"   Classes: {model.classes_}")
    else:
        print(f"🤔 Weird. It loaded as type: {type(data)}")

except Exception as e:
    print(f"❌ Error: {e}")
