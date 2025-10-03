import numpy as np
import joblib
import os
from typing import List

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")

# Ensure directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

def load_model(symbol: str):
    """
    Load a trained model for a symbol.
    Example: BTC-USD -> BTC-USD_model.pkl
    """
    model_path = os.path.join(MODELS_DIR, f"{symbol}_xgboost_model.pkl")
    if not os.path.exists(models_path):
        raise FileNotFoundError(f"No trained model found for {symbol}. Train first!")
    return joblib.load(model_path)

def predict(symbol: str, features: List[float]) -> dict:
    """
    Run inference using a trained model.
    """
    model = load_model(symbol)

    # Convert features into numpy array with correct shape
    X = np.array(features).reshape(1, -1)
    pred = model.predict(X)[0]

    # If model has predict_proba
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba(X)[0]))

    return {
        "prediction": int(pred),
        "confidence": confidence if confidence else 1.0
    }
