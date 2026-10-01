import os
import logging
import pandas as pd
import numpy as np
import joblib

# Use the same BasePredictor structure from your Factory
try:
    from .model_factory import BasePredictor
    from app.config2 import MODEL_STORAGE_DIR
except ImportError:
    # Local fallback for standalone testing
    MODEL_STORAGE_DIR = "data/saved_models"
    class BasePredictor:
        def __init__(self, m_id, b_id): pass

try:
    from tabpfn import TabPFNClassifier
except ImportError:
    TabPFNClassifier = None

logger = logging.getLogger("TabPFNPredictor")

class TabPFNPredictor(BasePredictor):
    """
    NEO-V7 TabPFN Driver: The 'Foundation Model' Brain.
    Uses In-Context Learning to estimate probabilities without classic training.
    """
    def __init__(self, model_id, bot_id):
        self.model_id = model_id
        self.bot_id = bot_id
        # TabPFN is usually initialized live, but we check for 'context' snapshots
        self.model = self._load_from_disk() 
        if self.model is None and TabPFNClassifier:
            self.model = TabPFNClassifier(device='cpu') # 'cuda' for GPU

    def _load_from_disk(self):
        # We look for a calibrated snapshot if it exists
        path = os.path.join(MODEL_STORAGE_DIR, f"{self.model_id}.joblib")
        return joblib.load(path) if os.path.exists(path) else None

    def predict_direction(self, state_df: pd.DataFrame) -> float:
        """
        Calculates Bullish probability by feeding the context window 
        into the foundation transformer.
        """
        if self.model is None or state_df.empty:
            return 0.5
            
        try:
            # Clean numeric features
            features = state_df.select_dtypes(include=[np.number])
            
            # TabPFN handles small datasets (N < 1000) best.
            # It uses the tail of the data as the 'context' to predict the last row.
            probs = self.model.predict_proba(features.tail(1))
            
            # Return probability of class 1 (Price Increase)
            return float(probs[0][1])
        except Exception:
            return 0.5
