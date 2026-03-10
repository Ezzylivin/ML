import numpy as np
import pandas as pd
import logging
import os
from .model_factory import ModelFactory

logger = logging.getLogger("StackingPredictor")

class StackingPredictor:
    def __init__(self, symbol="BTC-USD", timeframe="1h"):
        """
        Initializes the Council of Experts. 
        Standardized for the 25-feature Mega-Models.
        """
        self.symbol = symbol
        self.timeframe = timeframe
        # The core experts the Judge listens to
        self.expert_types = ['xgboost', 'randomforest', 'transformer']

    def predict_direction(self, df_history: pd.DataFrame) -> float:
        """
        Polls all experts and returns the Stacking Judge's final decision.
        """
        opinions = {}
        
        # 1. Gather Individual Expert Testimony
        for e_type in self.expert_types:
            expert = ModelFactory.load_model(e_type, self.symbol, self.timeframe)
            if expert:
                # RawModelAdapter handles the 25-feature logic & sequence slicing
                opinions[e_type] = expert.predict_direction(df_history)
            else:
                # Fallback if a model is missing
                opinions[e_type] = 0.5 

        # 2. Consult the Stacking Judge (The Meta-Model)
        judge = ModelFactory.load_model("stacking", self.symbol, self.timeframe)
        
        if judge:
            # Prepare the 'Expert Vector' for the Judge
            expert_vector = [opinions['xgboost'], opinions['randomforest'], opinions['transformer']]
            
            # predict_direction handles 'is_meta_model' logic automatically
            final_score = judge.predict_direction(df_history, council_probs=expert_vector)
        else:
            # Emergency Fallback: Weighted Average if Judge is offline
            # Weights: XGB (40%), RF (40%), Transformer (20%)
            final_score = np.average(
                [opinions['xgboost'], opinions['randomforest'], opinions['transformer']], 
                weights=[0.4, 0.4, 0.2]
            )

        # 3. 🎙️ Live "Consensus" Monitoring Output
        debate_log = " | ".join([f"{k.upper()}: {v:.2f}" for k, v in opinions.items()])
        sentiment = self._get_sentiment_label(final_score)
        
        #print(f"[{self.symbol}] {debate_log} ⚖️ JUDGE FINAL: {final_score:.2f} ({sentiment})")

        return final_score

    def _get_sentiment_label(self, score):
        if score > 0.85: return "STRONG BUY"
        if score > 0.70: return "BUY"
        if score < 0.20: return "STRONG SELL"
        if score < 0.35: return "SELL"
        return "NEUTRAL"
