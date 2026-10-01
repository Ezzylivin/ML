import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from app.config2 import DATA_DIR, MODEL_STORAGE_DIR

def debug_system():
    print("🔍 --- DEEP SYSTEM DIAGNOSTICS ---")
    
    # 1. Path Verification
    print(f"📂 Config DATA_DIR: {DATA_DIR}")
    print(f"📂 Config MODEL_STORAGE_DIR: {MODEL_STORAGE_DIR}")
    
    # 2. Permission Check
    if os.access(MODEL_STORAGE_DIR, os.W_OK):
        print("✅ Permission: WRITE access granted to model folder.")
    else:
        print("❌ Permission: NO WRITE access to model folder!")

    # 3. Directory Content Audit
    print(f"\n📄 Files currently in {MODEL_STORAGE_DIR}:")
    files = os.listdir(MODEL_STORAGE_DIR)
    for f in files:
        if "Stacking" in f:
            print(f"   found -> {f}")
    if not any("Stacking" in f for f in files):
        print("   (No Stacking files found)")

    # 4. Attempt a Test Save
    print("\n💾 Attempting Test Save (debug_test.joblib)...")
    try:
        test_model = {"status": "operational"}
        test_path = os.path.join(MODEL_STORAGE_DIR, "debug_test.joblib")
        joblib.dump(test_model, test_path)
        if os.path.exists(test_path):
            print(f"✅ SUCCESS: Test file saved to {test_path}")
            os.remove(test_path) # Clean up
        else:
            print("❌ FAILURE: joblib.dump finished but file does not exist!")
    except Exception as e:
        print(f"❌ ERROR during save: {e}")

if __name__ == "__main__":
    debug_system()
