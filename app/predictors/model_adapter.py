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
        
        # ============================================================
        # 🔧 FIX: Support normalization stats for Keras models
        # ============================================================
        # train_transformer.py now saves a companion .npz file with
        # the mean/std used during training. If present in the payload,
        # we apply the same normalization at inference time.
        self.norm_mean = None
        self.norm_std = None
        
        if isinstance(model_payload, dict):
            self.model = model_payload.get('model')
            self.feature_names = model_payload.get('feature_names') or self.default_features
            self.is_meta_model = model_payload.get('is_meta_model', False)
            self.norm_mean = model_payload.get('norm_mean')
            self.norm_std = model_payload.get('norm_std')
        else:
            self.model = model_payload
            self.feature_names = self.default_features
            self.is_meta_model = False

        # FIX #15: sibling judge trainers (create_judge.py, create_judges.py,
        # repairJudge.py) save the stacking judge WITHOUT the is_meta_model flag.
        # That made the judge fall through to the 25-feature tree path with
        # input_data=None -> None[features] TypeError -> silent 0.5, while
        # "Full Council assembled" still logged. Detect a judge by its input
        # width: the stacking judge is the only model trained on exactly 3
        # features (the expert probability vector).
        if not self.is_meta_model and self.model is not None:
            try:
                if int(getattr(self.model, 'n_features_in_', 0)) == 3:
                    self.is_meta_model = True
                    logger.info("⚖️ Judge auto-detected as meta-model (3 inputs); flag was missing.")
            except Exception:
                pass

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
                # FIX #16: read the same predict_proba column the backtester does
                # (col 2 for a 3-class model, else col 1) so live and backtest
                # probabilities agree.
                proba = self.model.predict_proba(X)
                col   = 2 if proba.shape[1] == 3 else 1
                return float(proba[0][col])

            is_numpy = isinstance(input_data, np.ndarray)

            # KERAS PATH: Needs 3D tensor (batch, timesteps, features)
            if hasattr(self.model, "input_shape"):
                if is_numpy:
                    X_raw = input_data.astype('float32')
                else:
                    if len(input_data) < 50:
                        return 0.5
                    X_raw = input_data[self.feature_names].tail(50).values.astype('float32')
                
                # 🔧 FIX: Apply normalization if stats are available
                # Without this, the model receives raw values (BTC ~60000)
                # but was trained on normalized values (mean ~0, std ~1).
                if self.norm_mean is not None and self.norm_std is not None:
                    X_raw = (X_raw - self.norm_mean) / self.norm_std
                
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
                # FIX #16: match the backtester's column selection (col 2 for
                # 3-class models, else col 1) to keep live/backtest consistent.
                proba = self.model.predict_proba(last_row)
                col   = 2 if proba.shape[1] == 3 else 1
                conf  = float(proba[0][col])
                return min(0.99, max(0.01, conf))
            
            return float(self.model.predict(last_row)[0])

        except Exception as e:
            logger.error(f"❌ Adapter Prediction Error: {e}")
            return 0.5

    def predict(self, df_history, council_probs=None):
        """Alias for predict_direction — kept for backward compatibility."""
        return self.predict_direction(df_history, council_probs=council_probs)
