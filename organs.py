import os
from app.config2 import MODEL_STORAGE_DIR

def verify_environment():
    print(f"--- 📂 SYSTEM PATH AUDIT ---")
    print(f"Expected Model Directory: {os.path.abspath(MODEL_STORAGE_DIR)}")
    
    if not os.path.exists(MODEL_STORAGE_DIR):
        print("❌ ERROR: Model directory does not exist.")
        return

    files = os.listdir(MODEL_STORAGE_DIR)
    print(f"✅ Found {len(files)} files in storage.")
    
    required_files = ["MetaModel_BTC-USD.joblib", "XGBoost_BTC-USD.joblib", "RandomForest_BTC-USD.joblib"]
    for f in required_files:
        status = "✅ PRESENT" if f in files else "❌ MISSING"
        print(f"   {f}: {status}")

if __name__ == "__main__":
    verify_environment()
