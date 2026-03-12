import numpy as np
import pandas as pd
import logging
import os
from .model_factory import ModelFactory
import time

logger = logging.getLogger("StackingPredictor")

class StackingPredictor:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.expert_types = ['xgboost', 'randomforest', 'transformer']
        
        # 🎯 HIRE ONCE: Load all models into memory right now
        self.experts = {}
        for e_type in self.expert_types:
            model = ModelFactory.load_model(e_type, self.symbol, self.timeframe)
            if model:
                self.experts[e_type] = model
        
        self.judge = ModelFactory.load_model("stacking", self.symbol, self.timeframe)
        logger.info(f"⚖️ Council of Experts assembled for {symbol}")

    def predict_direction(self, df_history, precalc_transformer=None) -> float:
        """
        Calculates the final ensemble score.
        Turbo Mode: Uses precalc_transformer if provided (Backtesting).
        Live Mode: Polls all experts individually (Real-time).
        """
        opinions = {}
        
        # 1. Gather Individual Expert Testimony
        for e_type in self.expert_types:
            # 🏎️ TURBO SHORTCUT: If we already have the Transformer score from the batch
            if e_type == 'transformer' and precalc_transformer is not None:
                opinions['transformer'] = precalc_transformer
                continue

            expert = self.experts.get(e_type)
            if expert:
                # 🕒 Audit Timing (Live Mode only)
                start = time.time()
                opinions[e_type] = expert.predict_direction(df_history)
                duration = (time.time() - start) * 1000

                # Log performance for the very first real-time call
                if not hasattr(self, '_audited'):
                    logger.info(f"⏱️ AUDIT: {e_type} took {duration:.2f}ms")
            else:
                opinions[e_type] = 0.5

        # Mark audit as done to prevent log spam
        if not hasattr(self, '_audited'):
            self._audited = True

        # 2. Consult the Stacking Judge
        # We ensure all three keys exist to avoid KeyError in the expert_vector
        xg = opinions.get('xgboost', 0.5)
        rf = opinions.get('randomforest', 0.5)
        tf_score = opinions.get('transformer', 0.5)
        
        expert_vector = [xg, rf, tf_score]

        if self.judge:
            # The Judge uses the Expert Vector to make the final call
            return self.judge.predict_direction(df_history, council_probs=expert_vector)
        else:
            # Fallback: Weighted average if the Stacking Judge .joblib is missing
            return np.average(expert_vector, weights=[0.4, 0.4, 0.2])

    def _get_sentiment_label(self, score):
        if score > 0.85: return "STRONG BUY"
        if score > 0.70: return "BUY"
        if score < 0.20: return "STRONG SELL"
        if score < 0.35: return "SELL"
        return "NEUTRAL"
