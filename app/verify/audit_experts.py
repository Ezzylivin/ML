import os
import joblib
import pandas as pd
from app.config2 import MODEL_STORAGE_DIR, DATA_DIR

def audit_expert_presence(symbol="BTC-USD"):
    print(f"\n🕵️ AUDITING EXPERT PULSE: {symbol}")
    
    # 🟢 EXACT filenames used by the Factory
    expected_files = {
        "XGBoost": f"XGBoost_{symbol}.joblib",
        "RandomForest": f"RandomForest_{symbol}.joblib",
        "Transformer": f"Transformer_{symbol}.keras",
        "Stacking": f"Stacking_{symbol}.joblib"
    }
    
    all_present = True
    for name, filename in expected_files.items():
        path = os.path.join(MODEL_STORAGE_DIR, filename)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # KB
            print(f"  ✅ {name:12} | Found: {filename} ({size:.1f} KB)")
        else:
            print(f"  ❌ {name:12} | MISSING at {path}")
            all_present = False
            
    if all_present:
        print(f"✨ ALL EXPERTS ONLINE for {symbol}")
    else:
        print(f"⚠️  COUNCIL INCOMPLETE for {symbol}")

if __name__ == "__main__":
    # Test our active coins
    for s in ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]:
        audit_expert_presence(s)
