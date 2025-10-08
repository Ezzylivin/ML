import numpy as np
import joblib
import os
from typing import List, Dict

# --- Service-level Model Cache ---
# This dictionary will store models in memory after they are loaded once.
# This avoids slow disk access on every prediction call.
_model_cache: Dict[str, any] = {}

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# The original GitHub links show the models in the `models/` directory. 
# If you have a `saved/` subdirectory, change the line above to:
# MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")

def load_model(symbol: str):
    """
    Load a trained model for a symbol, using a cache for efficiency.

    First, it checks if the model is already in the memory cache. 
    If not, it loads the model from disk and adds it to the cache.
    """
    # 1. Check the cache first for immediate retrieval
    if symbol in _model_cache:
        return _model_cache[symbol]

    # 2. If not in cache, load from disk
    # CORRECTED: Changed file extension from .pkl to .joblib to match your files
    model_path = os.path.join(MODELS_DIR, f"{symbol}_xgboost_model.joblib")
    
    # CORRECTED: Fixed the typo from 'models_path' to 'model_path'
    if not os.path.exists(model_path):
        # Raising a more specific error is helpful for debugging
        raise FileNotFoundError(f"Model file not found for symbol '{symbol}' at path: {model_path}")
    
    print(f"Loading model for '{symbol}' from disk...") # Informative log
    try:
        model = joblib.load(model_path)
        # 3. Store the newly loaded model in the cache
        _model_cache[symbol] = model
        return model
    except Exception as e:
        raise IOError(f"Failed to load model for '{symbol}'. File may be corrupt. Error: {e}")

def predict(symbol: str, features: List[float]) -> dict:
    """
    Run inference using a trained model. It will be fast after the first run
    for a given symbol due to the model cache.
    """
    # The load_model function now efficiently gets the model from cache or disk
    model = load_model(symbol)

    # Convert features into numpy array with correct shape
    X = np.array(features).reshape(1, -1)
    
    prediction = model.predict(X)[0]

    # Get confidence score if the model supports it
    confidence = 1.0 # Default confidence
    if hasattr(model, "predict_proba"):
        # The prediction is the class label, which is the index for predict_proba
        # We find the probability of the predicted class.
        probabilities = model.predict_proba(X)[0]
        confidence = float(probabilities[int(prediction)]) if int(prediction) in model.classes_ else float(max(probabilities))

    return {
        "prediction": int(prediction),
        "confidence": confidence
    }

def clear_model_cache():
    """Helper function to clear the cache if models need to be reloaded."""
    global _model_cache
    _model_cache = {}
    print("Model cache cleared.")
