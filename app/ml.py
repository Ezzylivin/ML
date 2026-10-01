import os
import logging
import traceback
import joblib  # Make sure to install: pip install joblib scikit-learn

# Try to import config from the same directory (app/)
try:
    from .config import MODEL_DIR
except ImportError:
    from config import MODEL_DIR

# Set up logging for the service
logger = logging.getLogger("MLService")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- GLOBAL CACHE (The Secret to fixing OOM) ---
# We store loaded models here. They are only added when requested.
_model_cache = {}

def get_available_models_logic() -> list:
    """
    Scans the MODEL_DIR for .joblib files and returns them as a list.
    """
    try:
        # Check if directory exists
        if not os.path.exists(MODEL_DIR):
            logger.warning(f"Model directory not found: {MODEL_DIR}")
            return []

        models = []
        for filename in os.listdir(MODEL_DIR):
            if filename.endswith('.joblib'):
                # Strip extension for the ID/Name
                model_name = filename.rsplit('.', 1)[0]
                models.append({
                    "id": model_name,
                    "name": model_name
                })

        # Sort models alphabetically
        models.sort(key=lambda x: x['id'])
        return models

    except Exception as e:
        logger.error(f"Error fetching ML models: {traceback.format_exc()}")
        return []

def load_model_logic(model_id: str):
    """
    Safely loads a model from disk ONLY if it's not already in memory.
    This prevents the server from crashing on startup.
    """
    global _model_cache
    
    # 1. Check if already loaded (Fast return)
    if model_id in _model_cache:
        return _model_cache[model_id]
    
    # 2. If not loaded, load it now
    try:
        path = os.path.join(MODEL_DIR, f"{model_id}.joblib")
        
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return None
            
        logger.info(f"⏳ Loading heavy model '{model_id}' into RAM...")
        
        # The heavy operation
        loaded_model = joblib.load(path)
        
        # Save to cache
        _model_cache[model_id] = loaded_model
        
        logger.info(f"✅ Model '{model_id}' loaded successfully.")
        return loaded_model

    except Exception as e:
        logger.error(f"CRASH loading model {model_id}: {e}")
        return None
