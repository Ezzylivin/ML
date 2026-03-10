import numpy as np
import pandas as pd
import logging
import os
from .model_factory import ModelFactory

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

    def predict_direction(self, df_history: pd.DataFrame) -> float:
        opinions = {}
        
        # 1. Gather Individual Expert Testimony (From RAM, not Disk!)
        for e_type in self.expert_types:
            expert = self.experts.get(e_type) # 🚀 Ultra-fast lookup
            if expert:
                opinions[e_type] = expert.predict_direction(df_history)
            else:
                opinions[e_type] = 0.5 

        # 2. Consult the Stacking Judge
        if self.judge:
            expert_vector = [opinions['xgboost'], opinions['randomforest'], opinions['transformer']]
            final_score = self.judge.predict_direction(df_history, council_probs=expert_vector)
        else:
            final_score = np.average(
                [opinions['xgboost'], opinions['randomforest'], opinions['transformer']], 
                weights=[0.4, 0.4, 0.2]
            )

        return final_score

    def _get_sentiment_label(self, score):
        if score > 0.85: return "STRONG BUY"
        if score > 0.70: return "BUY"
        if score < 0.20: return "STRONG SELL"
        if score < 0.35: return "SELL"
        return "NEUTRAL"
