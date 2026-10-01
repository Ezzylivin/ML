import os
import shutil
from app.config2 import MODEL_STORAGE_DIR

def normalize_filenames():
    print(f"--- 🔄 ALIGNING MODEL FILENAMES ---")
    files = os.listdir(MODEL_STORAGE_DIR)
    
    # Mapping of your current disk names -> what the system expects
    # Pattern: {Symbol}_{type}.joblib -> {Type}_{Symbol}.joblib
    mapping = {
        "xgboost.joblib": "XGBoost",
        "rf.joblib": "RandomForest",
        "transformer.keras": "Transformer"
    }
    
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD", "DOGE-USD", "SUI-USD", "PEPE-USD", "SHIB-USD"]
    
    count = 0
    for f in files:
        for sym in symbols:
            for suffix, model_type in mapping.items():
                old_name = f"{sym}_{suffix}"
                new_name = f"{model_type}_{sym}.joblib" if not suffix.endswith('.keras') else f"{model_type}_{sym}.keras"
                
                if f == old_name:
                    old_path = os.path.join(MODEL_STORAGE_DIR, f)
                    new_path = os.path.join(MODEL_STORAGE_DIR, new_name)
                    
                    if not os.path.exists(new_path):
                        shutil.copy2(old_path, new_path)
                        print(f"✅ Aligned: {old_name} ➔ {new_name}")
                        count += 1

    print(f"\n✨ Alignment Complete. {count} files normalized.")

if __name__ == "__main__":
    normalize_filenames()
