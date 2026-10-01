import os
import torch
import pandas as pd
import numpy as np
import logging

try:
    from .model_factory import BasePredictor
    from app.config2 import MODEL_STORAGE_DIR, ML_CONFIG
except ImportError:
    MODEL_STORAGE_DIR = "data/saved_models"
    ML_CONFIG = {"DEFAULT_LOOKBACK_WINDOW": 50}
    class BasePredictor:
        def __init__(self, m_id, b_id): pass

try:
    from pytorch_forecasting import TemporalFusionTransformer
except ImportError:
    TemporalFusionTransformer = None

logger = logging.getLogger("TransformerPredictor")

class TransformerPredictor(BasePredictor):
    """
    NEO-V7 Transformer Driver: The 'Attention' Brain.
    Optimized for multi-horizon regime sensing and non-linear temporal patterns.
    """
    def __init__(self, model_id, bot_id):
        self.model_id = model_id
        self.bot_id = bot_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_from_disk()

    def _load_from_disk(self):
        # Transformers use .ckpt (Checkpoints) instead of .joblib
        path = os.path.join(MODEL_STORAGE_DIR, f"{self.model_id}.ckpt")
        if os.path.exists(path) and TemporalFusionTransformer:
            try:
                # Load the PyTorch Lightning Checkpoint
                return TemporalFusionTransformer.load_from_checkpoint(path).to(self.device)
            except Exception as e:
                logger.error(f"Failed to load Transformer: {e}")
        return None

    def predict_direction(self, state_df: pd.DataFrame) -> float:
        """
        Runs a forward pass through the attention layers.
        """
        if self.model is None or state_df.empty:
            return 0.5

        try:
            # Extract the lookback window
            lookback = ML_CONFIG.get('DEFAULT_LOOKBACK_WINDOW', 50)
            data = state_df.tail(lookback).select_dtypes(include=[np.number]).values
            
            # Prepare 3D Tensor for PyTorch: (Batch, TimeSteps, Features)
            tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Inference Mode
            self.model.eval()
            with torch.no_grad():
                # TFT outputs quantiles; we take the median (0.5 quantile)
                prediction = self.model(tensor)
                # Normalize output to a 0.0-1.0 probability scale
                prob = torch.sigmoid(prediction.prediction[0][0][0]).item()
                
            return float(prob)
        except Exception:
            return 0.5
