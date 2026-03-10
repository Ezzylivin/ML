import pandas as pd
import pandas_ta as ta
import os
import logging
import numpy as np
from fastapi import HTTPException
from app.predictors.model_factory import ModelFactory

logger = logging.getLogger("BacktestEngine")

class RawModelAdapter:
    def __init__(self, model_payload):
        self.default_features = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_50', 'sma_200', 'ema_9', 'ema_21', 'ema_20',
            'rsi', 'atr', 'adx', 'st_trend', 
            'BBL_20_2.0_2.0', 'BBU_20_2.0_2.0', 
            'STOCHk_14_3_3', 'MACD_12_26_9', 'MACDs_12_26_9',
            'pa_high', 'pa_low', 'vol_ma', 
            'adx_logic', 'atr_logic', 'sma_logic'
        ]
        
        if isinstance(model_payload, dict):
            self.model = model_payload.get('model')
            self.feature_names = model_payload.get('feature_names', self.default_features)
            # 🎯 SYNCED NAME: using 'is_meta_model' to match the predict function
            self.is_meta_model = model_payload.get('is_meta_model', False) if isinstance(model_payload, dict) else False
        else:
            self.model = model_payload
            self.feature_names = self.default_features
            self.is_meta_model = False

    def predict_direction(self, df_history, council_probs=None):
        try:
            # 1. JUDGE LOGIC (Stacking/Meta-Model)
            # Now self.is_meta_model will correctly evaluate to True
            if self.is_meta_model and council_probs is not None:
                X = np.array([council_probs]) 
                return float(self.model.predict_proba(X)[0][1])

            # --- 🧠 TRANSFORMER LOGIC (.keras) ---
            if hasattr(self.model, "input_shape"):
                if len(df_history) < 50:
                    return 0.5
                
                # 🎯 CRITICAL FIX: Extract EXACTLY 50 rows as a raw numpy array
                # This ensures the input shape is ALWAYS (1, 50, 25)
                X_seq = df_history[self.feature_names].tail(50).values
                
                # Validate for NaNs (Transformer will fail on NaNs)
                if np.isnan(X_seq).any():
                    return 0.5
                    
                # Reshape to 3D tensor
                X_tensor = X_seq.reshape(1, 50, len(self.feature_names))
                
                # verbose=0 is mandatory for speed in batch mode
                pred = self.model.predict(X_tensor, verbose=0)[0]
                return float(pred[1] if len(pred) > 1 else pred[0])

            # 3. EXPERT LOGIC (XGB/RF/LightGBM)
            last_row = df_history[self.feature_names].iloc[[-1]]
            if hasattr(self.model, "predict_proba"):
                conf = float(self.model.predict_proba(last_row)[0][1])
                return min(0.99, max(0.01, conf))
            
            return float(self.model.predict(last_row)[0])

        except Exception as e:
            logger.error(f"❌ Adapter Prediction Error: {e}")
            return 0.5


def predict(self, df_history, council_probs=None):
        """
        Redirects legacy '.predict()' calls to our new '.predict_direction()' 
        to prevent BacktestEngine crashes.
        """
        return self.predict_direction(df_history, council_probs=council_probs)


import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from fastapi import HTTPException
from app.predictors.model_factory import ModelFactory

logger = logging.getLogger("BacktestEngine")

class Backtester:
    def __init__(self, config: dict):
        self.config = config 
        self.symbol = config.get('symbol')
        self.timeframe = config.get('timeframe')
        self.start_str = config.get('startDate')
        self.end_str = config.get('endDate')
        self.params = config.get('params', {})
        self.initial_balance = float(config.get('initialBalance', 1000))
        self.combination_rule = config.get('combinationRule', 'OR').upper()

        p = self.params
        self.tp_pct = float(p.get('take_profit', 0.13))
        self.sl_pct = float(p.get('stop_loss', 0.086))
        self.ts_pct = float(p.get('trailing_stop', 0.086))
        
        self.strategies = config.get('strategies', [])
        raw_risk = config.get('risk_percentage', 1.0)
        self.risk_mult = float(raw_risk) / 100.0
        
        self.ml_limit_long = float(config.get('mlThresholdLong', 0.80))
        self.ml_limit_short = float(config.get('mlThresholdShort', 0.80))
        self.model_name = config.get('mlModel', 'stacking')

    def calculate_indicators(self, df):
        """
        V25 MEGA-ENGINEER: Calculates all 25 features required for 
        XGBoost, RandomForest, and Transformer parity.
        """
        if len(df) < 200: return df 
        
        # 1. Foundation
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['ema_9'] = ta.ema(df['close'], length=9)
        df['ema_21'] = ta.ema(df['close'], length=21)
        df['ema_20'] = ta.ema(df['close'], length=20)
        
        # 2. ADX (Trend Strength)
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is not None:
            df['adx'] = adx_df.iloc[:, 0]
            
        # 3. Bollinger Bands (Volatility Walls)
        bb = ta.bbands(df['close'], length=20, std=2.0)
        if bb is not None:
            df['BBL_20_2.0_2.0'] = bb.iloc[:, 0]
            df['BBU_20_2.0_2.0'] = bb.iloc[:, 2]

        # 4. Supertrend (Directional Bias)
        st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
        df['st_trend'] = st.iloc[:, 1] if st is not None else 0

        # 5. Stochastic & MACD (Momentum)
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None: df['STOCHk_14_3_3'] = stoch.iloc[:, 0]
        
        macd = ta.macd(df['close'])
        if macd is not None:
            df['MACD_12_26_9'] = macd.iloc[:, 0]
            df['MACDs_12_26_9'] = macd.iloc[:, 2]

        # 6. Price Action & Volume (Structure)
        df['pa_high'] = df['high'].rolling(window=20).max()
        df['pa_low'] = df['low'].rolling(window=20).min()
        df['vol_ma'] = ta.sma(df['volume'], length=20)

        # 7. LOGIC FEATURES (Essential for the Stacking Judge)
        df['adx_logic'] = np.where(df['adx'] > 25, 1, 0)
        df['atr_logic'] = (df['atr'] / df['close']) * 1000
        df['sma_logic'] = np.where(df['close'] > df['sma_200'], 1, -1)

        return df.dropna()

    def get_signal(self, row, strategies):
        """ Calculates the combined 'Vote' of all active strategies """
        votes = 0
        for strat in strategies:
            code = strat.get('code')
            p = strat.get('params', {})
            
            if code == "stoch":
                k_val = row.get('STOCHk_14_3_3', 50)
                if k_val < 20: votes += 1
                elif k_val > 80: votes -= 1
            elif code in ["bb_fade", "bollinger_bands"]:
                lower = row.get('BBL_20_2.0_2.0', 0)
                upper = row.get('BBU_20_2.0_2.0', 999999)
                if row['close'] < lower: votes += 1
                elif row['close'] > upper: votes -= 1
            elif code == "rsi_threshold":
                rsi = row.get('rsi', 50)
                if rsi < p.get('oversold', 30): votes += 1
                elif rsi > p.get('overbought', 70): votes -= 1
            elif code == "sma_crossover":
                if row.get('sma_50', 0) > row.get('sma_200', 0): votes += 1
                else: votes -= 1
            elif code == "supertrend":
                if row.get('st_trend', 0) == 1: votes += 1
                elif row.get('st_trend', 0) == -1: votes -= 1
            elif code == "macd_crossover":
                if row.get('MACD_12_26_9', 0) > row.get('MACDs_12_26_9', 0): votes += 1
                else: votes -= 1
            elif code == "atr_breakout":
                upper = row.get('ema_20', 0) + (row.get('atr', 0) * p.get('multiplier', 1.5))
                lower = row.get('ema_20', 0) - (row.get('atr', 0) * p.get('multiplier', 1.5))
                if row['close'] > upper: votes += 1
                elif row['close'] < lower: votes -= 1
            elif code == "ema_cloud":
                if row.get('ema_9', 0) > row.get('ema_21', 0): votes += 1
                else: votes -= 1
            elif code == "pa_breakout":
                if row['close'] >= row.get('pa_high', 999999): votes += 1
                elif row['close'] <= row.get('pa_low', 0): votes -= 1
            elif code == "vol_profile":
                if row.get('volume', 0) > (row.get('vol_ma', 0) * p.get('threshold', 1.5)):
                    votes += (1 if row['close'] > row.get('sma_50', 0) else -1)

        if self.combination_rule == "AND":
            if votes >= len(strategies): return 1
            if votes <= -len(strategies): return -1
            return 0
        else: 
            if votes > 0: return 1
            if votes < 0: return -1
            return 0

    async def load_data(self):
        from datetime import timedelta
        fetch_start = (pd.to_datetime(self.start_str) - timedelta(days=15)).strftime('%Y-%m-%d')
        try:
            from main4 import ensure_full_data 
            df = await ensure_full_data(self.symbol, self.timeframe, fetch_start, self.end_str)
            if df is None or df.empty:
                raise HTTPException(status_code=400, detail="Data gap could not be filled.")
            df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            logger.error(f"Data Sync Error: {e}")
            raise HTTPException(status_code=500, detail=f"Data Sync Failed: {str(e)}")

    import pandas as pd
import pandas_ta as ta
import os
import logging
import numpy as np
from fastapi import HTTPException
from app.predictors.model_factory import ModelFactory

logger = logging.getLogger("BacktestEngine")

class RawModelAdapter:
    def __init__(self, model_payload):
        self.default_features = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_50', 'sma_200', 'ema_9', 'ema_21', 'ema_20',
            'rsi', 'atr', 'adx', 'st_trend', 
            'BBL_20_2.0_2.0', 'BBU_20_2.0_2.0', 
            'STOCHk_14_3_3', 'MACD_12_26_9', 'MACDs_12_26_9',
            'pa_high', 'pa_low', 'vol_ma', 
            'adx_logic', 'atr_logic', 'sma_logic'
        ]
        
        if isinstance(model_payload, dict):
            self.model = model_payload.get('model')
            self.feature_names = model_payload.get('feature_names', self.default_features)
            # 🎯 SYNCED NAME: using 'is_meta_model' to match the predict function
            self.is_meta_model = model_payload.get('is_meta_model', False) if isinstance(model_payload, dict) else False
        else:
            self.model = model_payload
            self.feature_names = self.default_features
            self.is_meta_model = False

    def predict_direction(self, df_history, council_probs=None):
        try:
            # 1. JUDGE LOGIC (Stacking/Meta-Model)
            # Now self.is_meta_model will correctly evaluate to True
            if self.is_meta_model and council_probs is not None:
                X = np.array([council_probs]) 
                return float(self.model.predict_proba(X)[0][1])

            # 2. TRANSFORMER LOGIC (.keras)
            if hasattr(self.model, "input_shape"):
                if len(df_history) < 50: 
                    return 0.5
                X_seq = df_history[self.feature_names].tail(50).values
                X_seq = np.expand_dims(X_seq, axis=0)
                pred = self.model.predict(X_seq, verbose=0)[0]
                return float(pred[1] if len(pred) > 1 else pred[0])

            # 3. EXPERT LOGIC (XGB/RF/LightGBM)
            last_row = df_history[self.feature_names].iloc[[-1]]
            if hasattr(self.model, "predict_proba"):
                conf = float(self.model.predict_proba(last_row)[0][1])
                return min(0.99, max(0.01, conf))
            
            return float(self.model.predict(last_row)[0])

        except Exception as e:
            logger.error(f"❌ Adapter Prediction Error: {e}")
            return 0.5


def predict(self, df_history, council_probs=None):
        """
        Redirects legacy '.predict()' calls to our new '.predict_direction()' 
        to prevent BacktestEngine crashes.
        """
        return self.predict_direction(df_history, council_probs=council_probs)


import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from fastapi import HTTPException
from app.predictors.model_factory import ModelFactory

logger = logging.getLogger("BacktestEngine")

class Backtester:
    def __init__(self, config: dict):
        self.config = config 
        self.symbol = config.get('symbol')
        self.timeframe = config.get('timeframe')
        self.start_str = config.get('startDate')
        self.end_str = config.get('endDate')
        self.params = config.get('params', {})
        self.initial_balance = float(config.get('initialBalance', 1000))
        self.combination_rule = config.get('combinationRule', 'OR').upper()

        p = self.params
        self.tp_pct = float(p.get('take_profit', 0.13))
        self.sl_pct = float(p.get('stop_loss', 0.086))
        self.ts_pct = float(p.get('trailing_stop', 0.086))
        
        self.strategies = config.get('strategies', [])
        raw_risk = config.get('risk_percentage', 1.0)
        self.risk_mult = float(raw_risk) / 100.0
        
        self.ml_limit_long = float(config.get('mlThresholdLong', 0.80))
        self.ml_limit_short = float(config.get('mlThresholdShort', 0.80))
        self.model_name = config.get('mlModel', 'stacking')

    def calculate_indicators(self, df):
        """
        V25 MEGA-ENGINEER: Calculates all 25 features required for 
        XGBoost, RandomForest, and Transformer parity.
        """
        if len(df) < 200: return df 
        
        # 1. Foundation
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['ema_9'] = ta.ema(df['close'], length=9)
        df['ema_21'] = ta.ema(df['close'], length=21)
        df['ema_20'] = ta.ema(df['close'], length=20)
        
        # 2. ADX (Trend Strength)
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is not None:
            df['adx'] = adx_df.iloc[:, 0]
            
        # 3. Bollinger Bands (Volatility Walls)
        bb = ta.bbands(df['close'], length=20, std=2.0)
        if bb is not None:
            df['BBL_20_2.0_2.0'] = bb.iloc[:, 0]
            df['BBU_20_2.0_2.0'] = bb.iloc[:, 2]

        # 4. Supertrend (Directional Bias)
        st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
        df['st_trend'] = st.iloc[:, 1] if st is not None else 0

        # 5. Stochastic & MACD (Momentum)
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None: df['STOCHk_14_3_3'] = stoch.iloc[:, 0]
        
        macd = ta.macd(df['close'])
        if macd is not None:
            df['MACD_12_26_9'] = macd.iloc[:, 0]
            df['MACDs_12_26_9'] = macd.iloc[:, 2]

        # 6. Price Action & Volume (Structure)
        df['pa_high'] = df['high'].rolling(window=20).max()
        df['pa_low'] = df['low'].rolling(window=20).min()
        df['vol_ma'] = ta.sma(df['volume'], length=20)

        # 7. LOGIC FEATURES (Essential for the Stacking Judge)
        df['adx_logic'] = np.where(df['adx'] > 25, 1, 0)
        df['atr_logic'] = (df['atr'] / df['close']) * 1000
        df['sma_logic'] = np.where(df['close'] > df['sma_200'], 1, -1)

        return df.dropna()

    def get_signal(self, row, strategies):
        """ Calculates the combined 'Vote' of all active strategies """
        votes = 0
        for strat in strategies:
            code = strat.get('code')
            p = strat.get('params', {})
            
            if code == "stoch":
                k_val = row.get('STOCHk_14_3_3', 50)
                if k_val < 20: votes += 1
                elif k_val > 80: votes -= 1
            elif code in ["bb_fade", "bollinger_bands"]:
                lower = row.get('BBL_20_2.0_2.0', 0)
                upper = row.get('BBU_20_2.0_2.0', 999999)
                if row['close'] < lower: votes += 1
                elif row['close'] > upper: votes -= 1
            elif code == "rsi_threshold":
                rsi = row.get('rsi', 50)
                if rsi < p.get('oversold', 30): votes += 1
                elif rsi > p.get('overbought', 70): votes -= 1
            elif code == "sma_crossover":
                if row.get('sma_50', 0) > row.get('sma_200', 0): votes += 1
                else: votes -= 1
            elif code == "supertrend":
                if row.get('st_trend', 0) == 1: votes += 1
                elif row.get('st_trend', 0) == -1: votes -= 1
            elif code == "macd_crossover":
                if row.get('MACD_12_26_9', 0) > row.get('MACDs_12_26_9', 0): votes += 1
                else: votes -= 1
            elif code == "atr_breakout":
                upper = row.get('ema_20', 0) + (row.get('atr', 0) * p.get('multiplier', 1.5))
                lower = row.get('ema_20', 0) - (row.get('atr', 0) * p.get('multiplier', 1.5))
                if row['close'] > upper: votes += 1
                elif row['close'] < lower: votes -= 1
            elif code == "ema_cloud":
                if row.get('ema_9', 0) > row.get('ema_21', 0): votes += 1
                else: votes -= 1
            elif code == "pa_breakout":
                if row['close'] >= row.get('pa_high', 999999): votes += 1
                elif row['close'] <= row.get('pa_low', 0): votes -= 1
            elif code == "vol_profile":
                if row.get('volume', 0) > (row.get('vol_ma', 0) * p.get('threshold', 1.5)):
                    votes += (1 if row['close'] > row.get('sma_50', 0) else -1)

        if self.combination_rule == "AND":
            if votes >= len(strategies): return 1
            if votes <= -len(strategies): return -1
            return 0
        else: 
            if votes > 0: return 1
            if votes < 0: return -1
            return 0

    async def load_data(self):
        from datetime import timedelta
        fetch_start = (pd.to_datetime(self.start_str) - timedelta(days=15)).strftime('%Y-%m-%d')
        try:
            from main4 import ensure_full_data 
            df = await ensure_full_data(self.symbol, self.timeframe, fetch_start, self.end_str)
            if df is None or df.empty:
                raise HTTPException(status_code=400, detail="Data gap could not be filled.")
            df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            logger.error(f"Data Sync Error: {e}")
            raise HTTPException(status_code=500, detail=f"Data Sync Failed: {str(e)}")

    async def run(self):
         try:
             # 1. SETUP: Load data with a 10-day lookback padding for indicators
             df_raw = await self.load_data()
             df_full = self.calculate_indicators(df_raw)
            
             # Identify the user's exact start line
             user_start = pd.to_datetime(self.start_str).replace(tzinfo=None)
             
             # Create a "Neural Slice" that includes enough history for the Transformer (50 rows)
             # but is filtered for the user's requested range
             df_slice = df_full[df_full.index >= (user_start - pd.Timedelta(hours=60))].copy()
            
             if df_slice.empty:
                 return {"status": "failed", "error": "No data found for range."}
            
             # 2. 🚀 BATCH PRE-CALCULATION (The Speed Boost)
             # Instead of asking the AI in a loop, we do it all once here.
             from app.predictors.stacking_predictor import StackingPredictor
             ml_engine = None
             if self.model_name and self.model_name != "off":
                 ml_engine = StackingPredictor(symbol=self.symbol, timeframe=self.timeframe)
                
                 logger.info(f"🏎️ TURBO: Pre-calculating AI scores for {len(df_slice)} candles...")
                 ai_scores = []
                
                # We start 50 candles in to satisfy the Transformer sequence requirement
                 for i in range(50, len(df_slice)):
                    # Silent predict (Internal debate logs should be commented out in stacking_predictor.py)
                     score = ml_engine.predict_direction(df_slice.iloc[:i+1])
                     ai_scores.append(score)
                
                # Align the dataframe to the user's start (removing the 50-row padding)
                 df_final = df_slice.iloc[50:].copy()
                 df_final['ai_conf'] = ai_scores
             else:
                 df_final = df_slice[df_slice.index >= user_start].copy()
                 df_final['ai_conf'] = 1.0 # Default if AI is disabled

            # 3. INSTANT TRADING LOOP
            # This logic now runs at CPU speed because AI math is already done.
             balance, position, entry_price = self.initial_balance, None, 0
             tp_price, tsl_price = 0, 0
             equity_curve, trades, vetoed_logs = [], [], []

             for i in range(len(df_final)):
                 row = df_final.iloc[i]
                 current_time = str(row.name)
                 signal = self.get_signal(row, self.strategies)

                # Neural Gate (Instant Lookup)
                 conf_score = row['ai_conf']
                 is_short_trend = row['close'] < row.get('sma_200', 0)
                 limit = self.ml_limit_short if is_short_trend else self.ml_limit_long
                 gate_passed = conf_score >= limit

                # Sample Vetoes (Only every 5th to keep response payload light)
                 if signal != 0 and not gate_passed and position is None and i % 5 == 0:
                     vetoed_logs.append({
                         "time": current_time, "signal": "Long" if signal == 1 else "Short",
                         "conf_score": round(conf_score, 4), "limit": round(limit, 4), "price": row['close']
                     })

                # Exit Monitoring
                 if position:
                     exit_p, exit_r = None, None
                     if position == 'long':
                         if row['high'] >= tp_price: exit_p, exit_r = tp_price, "Take Profit"
                         elif row['low'] <= tsl_price: exit_p, exit_r = tsl_price, "Trailing Stop"
                         elif signal == -1: exit_p, exit_r = row['close'], "Signal Flip"
                     else:
                         if row['low'] <= tp_price: exit_p, exit_r = tp_price, "Take Profit"
                         elif row['high'] >= tsl_price: exit_p, exit_r = tsl_price, "Trailing Stop"
                         elif signal == 1: exit_p, exit_r = row['close'], "Signal Flip"

                     if exit_p:
                         pnl_v = (exit_p - entry_price)/entry_price if position == 'long' else (entry_price - exit_p)/entry_price
                         balance *= (1 + (pnl_v * self.risk_mult) - 0.0016)
                         trades.append({
                             "type": "exit", "reason": exit_r, "price": round(exit_p, 2), "time": current_time,
                             "pnl": round(pnl_v * 100, 2), "balance": round(balance, 2)
                         })
                         position = None
                         continue

                    # Update Trailing Stop (High-water mark)
                     if position == 'long':
                         new_tsl = row['high'] * (1 - self.ts_pct)
                         if new_tsl > tsl_price: tsl_price = new_tsl
                     else:
                         new_tsl = row['low'] * (1 + self.ts_pct)
                         if new_tsl < tsl_price: tsl_price = new_tsl

                # Entry Execution
                 if position is None and gate_passed and signal != 0:
                     position = 'long' if signal == 1 else 'short'
                     entry_price = row['close']
                     tp_price = entry_price * (1 + self.tp_pct) if position == 'long' else entry_price * (1 - self.tp_pct)
                     tsl_price = entry_price * (1 - self.ts_pct) if position == 'long' else entry_price * (1 + self.ts_pct)
                     balance *= (1 - 0.0006)
                     trades.append({
                         "type": "buy" if position == 'long' else "sell", "price": entry_price, 
                         "time": current_time, "ai_score": round(conf_score, 4)
                     })

                # Log Equity every 4 hours to keep the UI charts smooth but fast
                 if i % 4 == 0:
                     equity_curve.append({"time": current_time, "balance": round(balance, 2)})

             return {
                 "status": "success",
                 "metrics": {
                     "finalBalance": round(balance, 2),
                     "roi": round(((balance - self.initial_balance) / self.initial_balance) * 100, 2),
                     "totalTrades": len(trades),
                     "netProfit": round(balance - self.initial_balance, 2)
                 },
                 # Return only the last 500 candles to keep the Browser from lagging
                 "candleData": df_final.reset_index().rename(columns={'timestamp': 'time', 'index': 'time'}).tail(500).to_dict('records'),
                 "trades": trades,
                 "equityCurve": equity_curve,
                 "vetoed_signals": vetoed_logs[:50],
                 "initialBalance": self.initial_balance
             }
         except Exception as e:
             logger.error(f"Backtest Error: {e}")
             return {"status": "failed", "error": str(e)}
