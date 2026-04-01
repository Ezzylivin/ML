"""
RawModelAdapter — Unified prediction interface for all model types.

Extracted from backtest2.py to break the circular import:
    ModelFactory → backtest2.RawModelAdapter → ModelFactory

Now both ModelFactory and backtest2 can import from here safely.

Location: app/predictors/model_adapter.py
"""
import numpy as np
import logging
import tensorflow as tf

from app.config2 import FEATURE_COLUMNS

logger = logging.getLogger("ModelAdapter")


class RawModelAdapter:
    """
    Bridge between raw model files (.joblib/.keras) and the prediction interface.
    Gives every model a unified .predict_direction() method.
    
    Handles three model types:
    - Keras/TF models (Transformer/LSTM): Takes 50-bar sequences, shape (1, 50, N_features)
    - Sklearn tree models (XGBoost/RF): Takes single row, shape (1, N_features)
    - Meta models (Stacking Judge): Takes expert probability vector, shape (1, 3)
    """
    
    def __init__(self, model_payload):
        self.default_features = FEATURE_COLUMNS
        
        if isinstance(model_payload, dict):
            self.model = model_payload.get('model')
            self.feature_names = model_payload.get('feature_names') or self.default_features
            self.is_meta_model = model_payload.get('is_meta_model', False)
        else:
            self.model = model_payload
            self.feature_names = self.default_features
            self.is_meta_model = False

        # Pre-warm Keras models to avoid first-call latency
        if hasattr(self.model, "input_shape") and not self.is_meta_model:
            try:
                dummy_input = tf.zeros((1, 50, len(self.feature_names)))
                self.model(dummy_input, training=False)
                logger.info("✅ Keras model graph compiled & locked.")
            except Exception as e:
                logger.warning(f"⚠️ Could not pre-warm model: {e}")

    def predict_direction(self, input_data, council_probs=None):
        """
        Returns a float between 0.0 and 1.0 representing bullish confidence.
        
        Args:
            input_data: DataFrame or numpy array of OHLCV + features
            council_probs: List of [xgb_score, rf_score, tf_score] for meta-model
        
        Returns:
            float: Probability of upward movement (0.0 = strong sell, 1.0 = strong buy)
        """
        try:
            # META-MODEL PATH: Judge uses expert scores, ignores raw data
            if self.is_meta_model and council_probs is not None:
                X = np.array([council_probs]) 
                return float(self.model.predict_proba(X)[0][1])

            is_numpy = isinstance(input_data, np.ndarray)

            # KERAS PATH: Needs 3D tensor (batch, timesteps, features)
            if hasattr(self.model, "input_shape"):
                if is_numpy:
                    X_raw = input_data.astype('float32')
                else:
                    if len(input_data) < 50:
                        return 0.5
                    X_raw = input_data[self.feature_names].tail(50).values.astype('float32')
                
                X_tensor = tf.convert_to_tensor(X_raw)
                X_tensor = tf.expand_dims(X_tensor, 0)  # Add batch dim
                preds = self.model(X_tensor, training=False)
                
                # Handle both Dense(1, sigmoid) and Dense(2, softmax) outputs
                return float(preds[0][1] if preds.shape[1] > 1 else preds[0][0])

            # TREE MODEL PATH: Needs 2D array (batch, features)
            if is_numpy:
                last_row = input_data[-1:].astype('float32')
            else:
                last_row = input_data[self.feature_names].iloc[[-1]]

            if hasattr(self.model, "predict_proba"):
                conf = float(self.model.predict_proba(last_row)[0][1])
                return min(0.99, max(0.01, conf))
            
            return float(self.model.predict(last_row)[0])

        except Exception as e:
            logger.error(f"❌ Adapter Prediction Error: {e}")
            return 0.5

    def predict(self, df_history, council_probs=None):
        """Alias for predict_direction — kept for backward compatibility."""
        return self.predict_direction(df_history, council_probs=council_probs)
