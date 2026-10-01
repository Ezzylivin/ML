import joblib
import os
import sys

# Standard paths
MODEL_PATH = "/root/Project/ML/data/saved_models/BTC-USD_rf.joblib"

print(f"🕵️ ATTEMPTING RAW LOAD: {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    print("✅ SUCCESS: Model loaded into memory.")
    print(f"📊 Model Type: {type(model)}")
except Exception as e:
    print("❌ CRASHED DURING LOAD!")
    print(f"📝 ERROR MESSAGE: {e}")
    
    # If it's a version error, let's check your current environment
    import sklearn
    print(f"🛠️ Current sklearn version: {sklearn.__version__}")
