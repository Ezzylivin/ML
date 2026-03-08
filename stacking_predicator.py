import os
import joblib
import numpy as np
import pandas as pd
from .model_factory import ModelFactory
from app.config2 import MODEL_STORAGE_DIR

class StackingPredictor:
    def __init__(self, bot_id="backtest_generic", symbol=None):
        """
        Initializes the Stacking Council.
        Synchronized for 10-feature input parity.
        """
        self.bot_id = bot_id
        self.symbol = symbol or "BTC-USD"
        
        # 🟢 Experts must match ModelFactory mapping keys exactly
        self.expert_types = ['XGBoost', 'RandomForest', 'Transformer']
        
        # Load the Meta-Model (The Judge) if it exists
        meta_path = os.path.join(MODEL_STORAGE_DIR, f"MetaModel_{self.symbol}.joblib")
        self.judge = None
        
        if os.path.exists(meta_path):
            try:
                self.judge = joblib.load(meta_path)
                print(f"⚖️ JUDGE ONLINE for {self.symbol}. Features expected: {self.judge.n_features_in_}")
            except Exception as e:
                print(f"❌ MetaModel Load Failed: {e}")

    def predict_direction(self, state_df: pd.DataFrame) -> float:
        results = {}
        for e_type in self.expert_types:
            expert = ModelFactory.load_model(e_type, symbol=self.symbol)
            if expert is None:
                results[e_type] = 0.5
                continue
                
            try:
                # 🎯 Ensure individual experts return probabilities
                results[e_type] = expert.predict_direction(state_df)
            except Exception as e:
                results[e_type] = 0.5

        # --- ⚖️ REFINED CONFLUENCE LOGIC ---
        if self.judge and len(results) == 3:
            try:
                X_meta = np.array([
                    results['XGBoost'], 
                    results['RandomForest'], 
                    results['Transformer']
                ]).reshape(1, -1)
                
                # Check if judge actually supports probabilities
                if hasattr(self.judge, "predict_proba"):
                    return float(self.judge.predict_proba(X_meta)[0][1])
                else:
                    # 🎯 FIXED: If judge is 'hard', use a weighted average instead of rounding
                    return np.average([results['XGBoost'], results['RandomForest'], results['Transformer']], weights=[0.4, 0.4, 0.2])
            except Exception as e:
                return np.mean(list(results.values()))
        
        return np.mean(list(results.values()))
