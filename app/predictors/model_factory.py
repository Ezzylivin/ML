"""
ModelFactory — central loader + in-memory cache for all trained models.

Loads the raw model files produced by the training scripts and wraps each in a
RawModelAdapter so every model exposes a uniform ``.predict_direction()``.

File naming convention (see app/verify/engineer_and_train.py, trainTransformers.py,
train_judge.py)::

    {ticker}_{timeframe}_{model_type}_model.joblib   # xgboost, randomforest, stacking
    {ticker}_{timeframe}_transformer_model.keras     # + {ticker}_{timeframe}_transformer_norm.npz

where ``ticker = symbol.split('-')[0].lower()`` (e.g. "BTC-USD" -> "btc").

Location: app/predictors/model_factory.py
"""
import os
import logging

import joblib
import numpy as np

from app.config2 import MODEL_DIR, FEATURE_COLUMNS
from app.predictors.model_adapter import RawModelAdapter

logger = logging.getLogger("ModelFactory")

# In-memory cache of successfully loaded adapters.
# Key: (model_type, ticker, timeframe) -> RawModelAdapter
_MODEL_CACHE = {}

_JOBLIB_TYPES = {"xgboost", "randomforest", "stacking"}
_KERAS_TYPES = {"transformer"}


def _ticker(symbol: str) -> str:
    return symbol.split("-")[0].lower()


class ModelFactory:
    """Loads and caches trained models, wrapped in RawModelAdapter."""

    @staticmethod
    def load_model(model_type, symbol, timeframe="1h"):
        """Return a RawModelAdapter for the requested model, or None if the
        model file does not exist. Successful loads are cached in memory so
        repeat calls (e.g. per-symbol StackingPredictor inits) are cheap.
        """
        ticker = _ticker(symbol)
        cache_key = (model_type, ticker, timeframe)

        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            if model_type in _JOBLIB_TYPES:
                path = os.path.join(
                    MODEL_DIR, f"{ticker}_{timeframe}_{model_type}_model.joblib"
                )
                if not os.path.exists(path):
                    logger.warning(f"⚠️ Model file not found: {path}")
                    return None
                payload = joblib.load(path)  # dict: {model, feature_names, [is_meta_model]}
                adapter = RawModelAdapter(payload)

            elif model_type in _KERAS_TYPES:
                path = os.path.join(
                    MODEL_DIR, f"{ticker}_{timeframe}_transformer_model.keras"
                )
                if not os.path.exists(path):
                    logger.warning(f"⚠️ Model file not found: {path}")
                    return None
                import tensorflow as tf  # lazy: only the transformer path needs TF here
                keras_model = tf.keras.models.load_model(path)
                payload = {"model": keras_model, "feature_names": FEATURE_COLUMNS}

                # Companion normalization stats saved by trainTransformers.py
                norm_path = os.path.join(
                    MODEL_DIR, f"{ticker}_{timeframe}_transformer_norm.npz"
                )
                if os.path.exists(norm_path):
                    stats = np.load(norm_path)
                    payload["norm_mean"] = stats["mean"]
                    payload["norm_std"] = stats["std"]
                else:
                    logger.warning(f"⚠️ Normalization stats missing: {norm_path}")

                adapter = RawModelAdapter(payload)

            else:
                logger.error(f"❌ Unknown model_type '{model_type}'")
                return None

        except Exception as e:
            logger.error(
                f"❌ Failed to load {model_type} for {symbol} ({timeframe}): {e}"
            )
            return None

        _MODEL_CACHE[cache_key] = adapter
        logger.info(f"📦 Loaded {model_type} model for {ticker} ({timeframe}).")
        return adapter


def clear_model_cache():
    """Drop all cached model adapters (e.g. after retraining)."""
    _MODEL_CACHE.clear()
    logger.info("🧹 Model cache cleared.")
