import joblib
import os
import logging

logger = logging.getLogger("ModelFactory")

class ModelFactory:
    @staticmethod
    def load_model(model_name, symbol, timeframe="1h"):
        # 1. Setup paths
        base_dir = "models" # Based on your training script output folder
        ticker = symbol.split('-')[0].upper() if '-' in symbol else symbol.split('/')[0].upper()
        
        # 🎯 MATCHING YOUR TRAINING FILENAMES: XGBoost_BTC-USD.joblib
        candidates = [
            f"{base_dir}/{model_name}_{symbol}.joblib",
            f"{base_dir}/{model_name}_{ticker}-USD.joblib",
            f"{base_dir}/{ticker}_{timeframe}_{model_name.lower()}_model.joblib"
        ]
        
        path = None
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
        
        if not path:
            logger.warning(f"⚠️ Model file not found for {model_name} on {symbol}")
            return None

        # 2. Bridge the model with the Adapter
        try:
            # Load the raw payload (The dict containing 'model' and 'feature_names')
            payload = joblib.load(path)
            
            # Import the bridge class
            from app.backtest2 import RawModelAdapter
            
            # 🎯 WRAP: This gives the model the .predict_direction() method
            # and the 25-feature list automatically.
            return RawModelAdapter(payload)
            
        except Exception as e:
            logger.error(f"❌ ModelFactory Crash: Could not load {path}. Error: {e}")
            return None
