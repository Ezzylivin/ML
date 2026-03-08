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
        if isinstance(model_payload, dict):
            self.model = model_payload.get('model')
            # 🎯 CRITICAL: This must match the 8 features the model was trained on
            self.feature_names = model_payload.get('feature_names', 
                ['sma_50', 'sma_200', 'st_trend', 'rsi', 'atr', 'adx', 'BBL_20_2.0_2.0', 'BBU_20_2.0_2.0'])
            self.scaler = model_payload.get('scaler', None)
        else:
            self.model = model_payload
            self.feature_names = []
            self.scaler = None

    def predict_direction(self, df_history):
        try:
            # 1. Grab the latest row for prediction
            last_row = df_history.iloc[[-1]].copy()
            
            # 2. Filter to EXACT features expected by the model
            # This prevents the "training data did not have these fields" error
            if self.feature_names:
                X = last_row[self.feature_names]
            else:
                X = last_row.select_dtypes(include=[np.number])

            # 3. Scale
            if self.scaler and not X.empty:
                X = self.scaler.transform(X)

            # 4. Extract Probability (Nuance)
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0]
                conf = float(probs[-1]) # Probability of UP
                
                # 🎯 THE SQUEEZE: Make the gate actually work even if overfitted
                if conf > 0.99: conf = 0.85 
                return conf
            
            return float(self.model.predict(X)[0])
                
        except Exception as e:
            logger.error(f"❌ AI Prediction Crash: {e}")
            return 0.5

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
        if len(df) < 50: return df 
        
        # Core Indicators
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['ema_9'] = ta.ema(df['close'], length=9)
        df['ema_21'] = ta.ema(df['close'], length=21)
        df['ema_20'] = ta.ema(df['close'], length=20)
        
        # ADX (Naming for AI parity)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None: df['adx'] = adx['ADX_14']
            
        # Bollinger (Naming for AI parity)
        # Inside calculate_indicators in backtest2.py
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None:
            # Standard names for Strategy Logic
            df['BBL_20_2.0'] = bb['BBL_20_2.0']
            df['BBU_20_2.0'] = bb['BBU_20_2.0']
            # Extra naming for AI Model Parity
            df['BBL_20_2.0_2.0'] = bb['BBL_20_2.0']
            df['BBU_20_2.0_2.0'] = bb['BBU_20_2.0']

        # Supertrend (Naming for AI parity)
        st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
        if st is not None: df['st_trend'] = st['SUPERTd_10_3.0']

        # Stochastic & MACD
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None: df = pd.concat([df, stoch], axis=1)
        macd = ta.macd(df['close'])
        if macd is not None: df = pd.concat([df, macd], axis=1)

        # Price Action
        df['pa_high'] = df['high'].rolling(window=20).max()
        df['pa_low'] = df['low'].rolling(window=20).min()
        df['vol_ma'] = ta.sma(df['volume'], length=20)

        return df.fillna(0)


    def get_signal(self, row, strategies):
        """ Calculates the combined 'Vote' of all active strategies """
        votes = 0
        
        for strat in strategies:
            code = strat.get('code')
            p = strat.get('params', {})
            
            # 1. Stochastic
            if code == "stoch":
                k_val = row.get('STOCHk_14_3_3', 50)
                if k_val < 20: votes += 1
                elif k_val > 80: votes -= 1
            
            # 2. Bollinger Fade
            if row['close'] < row.get('BBL_20_2.0_2.0', 0): 
                votes += 1
            elif row['close'] > row.get('BBU_20_2.0_2.0', 999999): 
                votes -= 1

            # 3. RSI Threshold
            elif code == "rsi_threshold":
                rsi = row.get('rsi', 50)
                if rsi < p.get('oversold', 30): votes += 1
                elif rsi > p.get('overbought', 70): votes -= 1

            # 4. SMA Crossover
            elif code == "sma_crossover":
                if row.get('sma_50', 0) > row.get('sma_200', 0) and row.get('sma_200', 0) > 0:
                    votes += 1
                elif row.get('sma_50', 0) < row.get('sma_200', 0) and row.get('sma_200', 0) > 0:
                    votes -= 1

            # 5. Supertrend
            elif code == "supertrend":
                # SUPERTd indicates direction: 1 for Up, -1 for Down
                if row.get('st_trend', 0) == 1: votes += 1
                elif row.get('st_trend', 0) == -1: votes -= 1

            # 6. MACD Crossover
            elif code == "macd_crossover":
                # MACD_12_26_9 > MACDs_12_26_9 (Signal line)
                if row.get('MACD_12_26_9', 0) > row.get('MACDs_12_26_9', 0): votes += 1
                else: votes -= 1

            # 7. ATR Breakout
            elif code == "atr_breakout":
                upper = row.get('ema_20', 0) + (row.get('atr', 0) * p.get('multiplier', 1.5))
                lower = row.get('ema_20', 0) - (row.get('atr', 0) * p.get('multiplier', 1.5))
                if row['close'] > upper: votes += 1
                elif row['close'] < lower: votes -= 1

            # 8. EMA Cloud
            elif code == "ema_cloud":
                if row.get('ema_9', 0) > row.get('ema_21', 0): votes += 1
                else: votes -= 1

            # 9. Price Action Breakout
            elif code == "pa_breakout":
                if row['close'] >= row.get('pa_high', 999999): votes += 1
                elif row['close'] <= row.get('pa_low', 0): votes -= 1

            # 10. Volume Surge
            elif code == "vol_profile":
                if row.get('volume', 0) > (row.get('vol_ma', 0) * p.get('threshold', 1.5)):
                    votes += (1 if row['close'] > row.get('sma_50', 0) else -1)

        # --- APPLY COMBINATION RULE ---
        if self.combination_rule == "AND":
            # Requires absolute consensus
            if votes >= len(strategies): return 1
            if votes <= -len(strategies): return -1
            return 0
        else: 
            # OR Logic: Triggers if the net vote is positive/negative
            if votes > 0: return 1
            if votes < 0: return -1
            return 0
            

    async def load_data(self):
        from datetime import timedelta
    
        fetch_start = (pd.to_datetime(self.start_str) - timedelta(days=10)).strftime('%Y-%m-%d')
        """
        Refreshes data from US-based exchanges if local file is missing or outdated.
        """
        try:
            # 🟢 CALL THE FETCH-ON-DEMAND LOGIC
            # We 'await' this because ensure_full_data is an async function
            from main4 import ensure_full_data 
            df = await ensure_full_data(
                self.symbol, 
                self.timeframe, 
                self.start_str, 
                self.end_str
            )
            
            if df is None or df.empty:
                raise HTTPException(status_code=400, detail="Data gap could not be filled.")
            
            # Ensure index is clean
            df.index = df.index.tz_localize(None)
            return df
            
        except Exception as e:
            logger.error(f"Data Sync Error: {e}")
            raise HTTPException(status_code=500, detail=f"Data Sync Failed: {str(e)}")

    


    async def run(self):
        try:
            df = await self.load_data()
            df = self.calculate_indicators(df)
            
            ml_model = None
            if self.model_name and self.model_name != "off":
                payload = ModelFactory.load_model(self.model_name, self.symbol, self.timeframe)
                if payload: ml_model = RawModelAdapter(payload)

            balance, position, entry_price = self.initial_balance, None, 0
            tp_price, tsl_price = 0, 0
            equity_curve, trades, vetoed_logs = [], [], []

            for i in range(len(df)):
                row = df.iloc[i]
                current_time = str(row.name)
                signal = self.get_signal(row, self.strategies)

                # 1. AI Neural Gate
                gate_passed, conf_score = True, 1.0
                is_short_trend = False
                if 'sma_200' in row and row['sma_200'] > 0:
                    is_short_trend = row['close'] < row['sma_200']
                limit = self.ml_limit_short if is_short_trend else self.ml_limit_long

                if ml_model:
                    conf_score = ml_model.predict_direction(df.iloc[:i+1])
                    gate_passed = conf_score >= limit

                # Log Vetoes
                if signal != 0 and not gate_passed and position is None:
                    vetoed_logs.append({
                        "time": current_time, "signal": "Long" if signal == 1 else "Short",
                        "conf_score": round(conf_score, 4), "limit": round(limit, 4), "price": row['close']
                    })

                # 2. Exit Logic (Intra-candle wicks)
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

                    # Update TSL
                    if position == 'long':
                        new_tsl = row['high'] * (1 - self.ts_pct)
                        if new_tsl > tsl_price: tsl_price = new_tsl
                    else:
                        new_tsl = row['low'] * (1 + self.ts_pct)
                        if new_tsl < tsl_price: tsl_price = new_tsl

                # 3. Entry Logic
                if position is None and gate_passed and signal != 0:
                    position = 'long' if signal == 1 else 'short'
                    entry_price = row['close']
                    tp_price = entry_price * (1 + self.tp_pct) if position == 'long' else entry_price * (1 - self.tp_pct)
                    tsl_price = entry_price * (1 - self.ts_pct) if position == 'long' else entry_price * (1 + self.ts_pct)
                    balance *= (1 - 0.0006)
                    trades.append({
                        "type": "buy" if position == 'long' else "sell", "price": entry_price, 
                        "time": current_time, "ai_score": round(conf_score, 4), "gate_limit": round(limit, 4)
                    })

                equity_curve.append({"time": current_time, "balance": round(balance, 2)})

            return {
                "status": "success",
                "metrics": {
                    "finalBalance": round(balance, 2),
                    "final_balance": round(balance, 2), # 🟢 Add for Frontend
                    "roi": round(((balance - self.initial_balance) / self.initial_balance) * 100, 2),
                    "totalTrades": len(trades),
                    "total_trades": len(trades),         # 🟢 Fixes the 'undefined' error
                    "netProfit": round(balance - self.initial_balance, 2),
                    "net_profit": round(balance - self.initial_balance, 2)
                },
                "candleData": df.reset_index().rename(columns={'timestamp': 'time', 'index': 'time'}).to_dict('records'),
                "trades": trades,
                "equityCurve": equity_curve,
                "vetoed_signals": vetoed_logs,
                "initialBalance": self.initial_balance
            }
        except Exception as e:
            logger.error(f"Backtest Error: {e}")
            return {"status": "failed", "error": str(e)}
(venv) root@intelligent-mendel:~/Project/ML# 
