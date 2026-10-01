import os
import sys

# Add the project root to sys.path so it can find app.config2
sys.path.append(os.getcwd())

try:
    from app.config2 import MODEL_STORAGE_DIR
except ImportError:
    print("❌ Could not import app.config2. Ensure you are in /root/Project/ML")
    sys.exit(1)

symbol = "BTC-USD"
timeframe = "1h"
experts = ["xgboost", "randomforest", "lightgbm", "mlpclassifier", "transformer"]

print(f"\n📂 Project Root: {os.getcwd()}")
print(f"📂 Configured Path: {MODEL_STORAGE_DIR}")
print("-" * 60)

clean_symbol = symbol.split('-')[0].lower()

for ex in experts:
    ext = "keras" if ex == "transformer" else "joblib"
    filename = f"{clean_symbol}_{timeframe}_{ex}_model.{ext}"
    full_path = os.path.join(MODEL_STORAGE_DIR, filename)
    exists = os.path.exists(full_path)
    status = "✅ FOUND  " if exists else "❌ MISSING"
    print(f"{status} | {filename}")
    if not exists:
        print(f"   Searching at: {full_path}")
print("-" * 60)
