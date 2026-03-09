import numpy as np
import pandas as pd
import logging
from .model_factory import ModelFactory

logger = logging.getLogger("Council")

class CouncilPredictor:
    def __init__(self, symbol="BTC-USD", timeframe="1h"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.expert_types = ['xgboost', 'randomforest', 'transformer']

    def predict_direction(self, df_history: pd.DataFrame) -> dict:
        """
        Gathers the 'Internal Debate' and returns the Judge's decision.
        """
        opinions = {}
        
        # 1. Gather Expert Opinions
        for e_type in self.expert_types:
            expert = ModelFactory.load_model(e_type, self.symbol, self.timeframe)
            if expert:
                # RawModelAdapter handles the 25-feature slicing and .keras vs .joblib
                opinions[e_type] = expert.predict_direction(df_history)
            else:
                opinions[e_type] = 0.5 # Neutral fallback

        # 2. Consult the Stacking Judge
        judge = ModelFactory.load_model("stacking", self.symbol, self.timeframe)
        
        if judge:
            expert_vector = [opinions['xgboost'], opinions['randomforest'], opinions['transformer']]
            # RawModelAdapter handles the Meta-Logic automatically
            final_score = judge.predict_direction(df_history, council_probs=expert_vector)
        else:
            # Simple fallback if the Judge isn't trained yet
            final_score = np.mean(list(opinions.values()))

        # 3. Log the Consensus (for your Live Monitor)
        debate = " | ".join([f"{k.upper()}: {v:.2f}" for k, v in opinions.items()])
        print(f"[{self.symbol}] {debate} ⚖️ JUDGE: {final_score:.2f}")

        return {
            "score": final_score,
            "opinions": opinions
        }
