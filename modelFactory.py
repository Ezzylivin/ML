import joblib
import os
import logging

# ============================================================
# 🔧 FIX #1: Import MODEL_DIR from config2
# ============================================================
# OLD: base_dir = "models" (relative, never matched training output)
# Training saved to: app/models/btc_1h_xgboost_model.joblib
# Factory searched:  models/BTC_1h_xgboost_model.joblib
# Result: Models NEVER loaded. Predictor always returned 0.5 (neutral).
# NEW: Uses the canonical MODEL_DIR from config2
from app.config2 import MODEL_DIR

logger = logging.getLogger("ModelFactory")

# ============================================================
# 🔧 FIX #2: In-memory model cache
# ============================================================
# OLD: Every call to load_model() hit disk with joblib.load().
#      In live mode (0.5s heartbeat), StackingPredictor.__init__
#      calls load_model 4 times = 8 disk reads per second.
# NEW: Cache by file path. Loaded once, reused forever.
#      Cache is invalidated only on server restart or manual clear.
_MODEL_CACHE = {}


def clear_model_cache():
    """Call this after retraining to force reload from disk."""
    _MODEL_CACHE.clear()
    logger.info("🗑️ Model cache cleared — next load will read from disk.")


class ModelFactory:
    @staticmethod
    def load_model(model_name, symbol, timeframe="1h"):
        # 1. Setup paths using config2's canonical directory
        ticker = symbol.split('-')[0].upper() if '-' in symbol else symbol.split('/')[0].upper()
        ticker_lower = ticker.lower()
        
        # ============================================================
        # 🔧 FIX #1b: Search paths that MATCH training output
        # ============================================================
        # Training saves: {MODEL_DIR}/{ticker_lower}_1h_{model_name}_model.joblib
        # Example:        app/models/btc_1h_xgboost_model.joblib
        #
        # We search multiple patterns to handle both old and new naming:
        candidates = [
            # Primary: matches engineer_and_train.py output exactly
            os.path.join(MODEL_DIR, f"{ticker_lower}_{timeframe}_{model_name.lower()}_model.joblib"),
            # Keras models from train_transformer.py
            os.path.join(MODEL_DIR, f"{ticker_lower}_{timeframe}_{model_name.lower()}_model.keras"),
            # Legacy patterns (in case old files exist)
            os.path.join(MODEL_DIR, f"{model_name}_{symbol}.joblib"),
            os.path.join(MODEL_DIR, f"{model_name}_{ticker}-USD.joblib"),
        ]
        
        path = None
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
        
        if not path:
            # 🔧 FIX #1c: Log which paths were searched (makes debugging trivial)
            logger.warning(f"⚠️ Model not found for {model_name}/{symbol}. Searched:")
            for c in candidates:
                logger.warning(f"   ❌ {c}")
            return None

        # ============================================================
        # 🔧 FIX #2b: Check cache before loading from disk
        # ============================================================
        if path in _MODEL_CACHE:
            logger.debug(f"⚡ Cache hit: {os.path.basename(path)}")
            return _MODEL_CACHE[path]

        # 3. Load and wrap the model
        try:
            logger.info(f"📂 Loading from disk: {path}")
            
            # ============================================================
            # 🔧 FIX #3: Handle both .joblib and .keras files
            # ============================================================
            if path.endswith('.keras'):
                import tensorflow as tf
                raw_model = tf.keras.models.load_model(path)
                payload = {
                    'model': raw_model,
                    'feature_names': None,  # Will use defaults from adapter
                    'is_meta_model': False
                }
            else:
                payload = joblib.load(path)
            
            # ============================================================
            # 🔧 FIX #4: Break circular import
            # ============================================================
            # OLD: from app.backtest2 import RawModelAdapter
            #      backtest2 imports ModelFactory → ModelFactory imports backtest2
            #      Only worked because import was inside this method (lazy).
            #      Any refactor that moves it to top-level = crash.
            #
            # NEW: Import from dedicated adapter module.
            #      If you haven't moved RawModelAdapter yet, the old import
            #      still works as fallback. But move it when you can.
            try:
                from app.predictors.model_adapter import RawModelAdapter
            except ImportError:
                # Fallback: old location (remove this once you move the class)
                from app.backtest2 import RawModelAdapter
            
            adapter = RawModelAdapter(payload)
            
            # 🔧 FIX #2c: Store in cache
            _MODEL_CACHE[path] = adapter
            logger.info(f"✅ Loaded & cached: {os.path.basename(path)}")
            
            return adapter
            
        except Exception as e:
            logger.error(f"❌ ModelFactory Crash: Could not load {path}. Error: {e}")
            return None
