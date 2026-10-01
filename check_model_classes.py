import joblib
import os
import sys
import numpy as np

# --- CONFIG ---
# We will check the specific model that was selected in your debug run
MODEL_NAME = "btc_4h_mlpclassifier_model.joblib"
MODEL_DIR = "/root/Project/ML/app/models"
PATH = os.path.join(MODEL_DIR, MODEL_NAME)

print(f"🔍 Inspecting: {PATH}")

if not os.path.exists(PATH):
    print(f"❌ Error: File not found at {PATH}")
    # Try finding any model to check
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.joblib')]
    if files:
        PATH = os.path.join(MODEL_DIR, files[0])
        print(f"⚠️ Falling back to: {files[0]}")
    else:
        sys.exit(1)

try:
    model = joblib.load(PATH)
    
    print(f"\n✅ Model Loaded Successfully!")
    print(f"   Type: {type(model).__name__}")
    
    if hasattr(model, "classes_"):
        classes = model.classes_
        count = len(classes)
        print(f"   🏷️  Classes Found: {classes}")
        print(f"   🔢 Class Count:   {count}")
        
        print("-" * 30)
        if count == 2:
            print("👉 VERDICT: 2-CLASS (BINARY) MODEL")
            print("   It predicts [Down, Up].")
            print("   Code accessing column [2] will CRASH.")
        elif count == 3:
            print("👉 VERDICT: 3-CLASS (MULTICLASS) MODEL")
            print("   It predicts [Sell, Hold, Buy].")
            print("   This matches standard logic.")
        else:
            print(f"👉 VERDICT: {count}-CLASS MODEL (Unusual)")
    else:
        print("⚠️ Model does not have standard 'classes_' attribute.")

    # Check expected features while we are here
    if hasattr(model, "n_features_in_"):
        print(f"   🧠 Expects:       {model.n_features_in_} columns")
    if hasattr(model, "feature_names_in_"):
        print(f"   📝 Column Names:  {list(model.feature_names_in_)}")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
