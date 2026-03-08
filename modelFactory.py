import joblib
import os
import logging

logger = logging.getLogger("ModelFactory")

class ModelFactory:
    @staticmethod
    def load_model(model_name, symbol, timeframe="1h"): # 🎯 Added default
        base_dir = "app/models" 
        
        # 🎯 FIX: Your symbols are "BTC-USD", not "BTC/USD"
        ticker = symbol.split('-')[0].lower() if '-' in symbol else symbol.split('/')[0].lower()
        
        candidates = [
            f"{base_dir}/{ticker}_{timeframe}_{model_name.lower()}_model.joblib", 
            f"{base_dir}/{model_name}.joblib",                                     
            f"{base_dir}/{model_name}"                                             
        ]
        
        path = None
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
        
        if not path:
            return None

        try:
            # Load the file
            obj = joblib.load(path)
            
            # Check if it's a dictionary (Metadata format) or raw model (Legacy)
            if isinstance(obj, dict) and 'model' in obj:
                logger.info(f"✅ Loaded Metadata Model: {path}")
                return obj['model']
            else:
                logger.info(f"✅ Loaded Legacy Model: {path}")
                return obj 
                
        except Exception as e:
            logger.error(f"❌ Corrupt Model File {path}: {e}")
            return None
(venv) root@intelligent-mendel:~/Project/ML# 
