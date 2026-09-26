import numpy as np
import logging
import time
from .model_factory import ModelFactory

logger = logging.getLogger("StackingPredictor")


class StackingPredictor:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.expert_types = ['xgboost', 'randomforest', 'transformer']
        
        # 🎯 HIRE ONCE: Load all models into memory on init
        # (ModelFactory now caches internally, so repeat inits are cheap too)
        self.experts = {}
        for e_type in self.expert_types:
            model = ModelFactory.load_model(e_type, self.symbol, self.timeframe)
            if model:
                self.experts[e_type] = model
            else:
                logger.warning(f"⚠️ Expert '{e_type}' not available for {symbol}")
        
        self.judge = ModelFactory.load_model("stacking", self.symbol, self.timeframe)
        
        if self.judge:
            logger.info(f"⚖️ Full Council assembled for {symbol}: {list(self.experts.keys())} + Judge")
        else:
            logger.info(f"⚖️ Council assembled for {symbol}: {list(self.experts.keys())} (no Judge — using weighted avg)")
        
        # Audit flag: log timing on first prediction only
        self._audited = False

    def predict_direction(self, df_history, precalc_transformer=None) -> float:
        """
        Calculates the final ensemble score.
        
        Turbo Mode: Uses precalc_transformer if provided (Backtesting).
        Live Mode: Polls all experts individually (Real-time).
        
        Returns:
            float: 0.0 (strong sell) to 1.0 (strong buy)
        """
        opinions = {}
        
        # 1. Gather Individual Expert Testimony
        for e_type in self.expert_types:
            # 🏎️ TURBO SHORTCUT: If we already have the Transformer score from batch
            if e_type == 'transformer' and precalc_transformer is not None:
                opinions['transformer'] = precalc_transformer
                continue
            
            expert = self.experts.get(e_type)
            if expert:
                start = time.time()
                opinions[e_type] = expert.predict_direction(df_history)
                duration = (time.time() - start) * 1000
                
                # Log timing on first call only
                if not self._audited:
                    logger.info(f"⏱️ AUDIT: {e_type} took {duration:.2f}ms")
            else:
                opinions[e_type] = 0.5
        
        if not self._audited:
            self._audited = True

        # 2. Build the expert probability vector
        xg = opinions.get('xgboost', 0.5)
        rf = opinions.get('randomforest', 0.5)
        tf_score = opinions.get('transformer', 0.5)
        expert_vector = [xg, rf, tf_score]

        # 3. Consult the Stacking Judge (or fallback)
        if self.judge:
            # ============================================================
            # 🔧 FIX #1: Don't pass df_history to the judge
            # ============================================================
            # OLD: self.judge.predict_direction(df_history, council_probs=expert_vector)
            #      The judge is a meta-model — when is_meta_model=True, 
            #      RawModelAdapter ignores df_history entirely and only uses
            #      council_probs. Passing the full DataFrame was wasteful and
            #      confusing to anyone reading the code.
            # NEW: Pass None for input_data. The meta-model path doesn't use it.
            return self.judge.predict_direction(None, council_probs=expert_vector)
        else:
            # ============================================================
            # 🔧 FIX #2: Rebalanced fallback weights
            # ============================================================
            # OLD: weights=[0.4, 0.4, 0.2]
            #      XGB and RF are trained on identical features with identical
            #      labels. They produce highly correlated predictions.
            #      Giving them 80% combined weight = double-counting one opinion
            #      while barely listening to the Transformer (your most complex model).
            #
            # NEW: Equal weights. Until you have evidence that one model
            #      consistently outperforms, equal weighting is the most
            #      robust default. If you want to tune this, use your
            #      backtest holdout set to measure individual model accuracy,
            #      then weight proportionally.
            return float(np.average(expert_vector, weights=[1/3, 1/3, 1/3]))

    # ============================================================
    # 🔧 FIX #3: Removed _get_sentiment_label (dead code)
    # ============================================================
    # Was never called anywhere. The same logic exists inline in
    # main4.py's heartbeat loop. No reason to duplicate it here.
