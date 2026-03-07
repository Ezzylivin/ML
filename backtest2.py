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
            self.feature_names = model_payload.get('feature_names', [])
            self.scaler = model_payload.get('scaler', None)
        else:
            self.model = model_payload
            self.feature_names = []
            self.scaler = None

    def predict_direction(self, df):
        try:
            last_row = df.iloc[[-1]].copy()

            # 🟢 AUTO-FIX: Map Backtester columns to Model columns
            # The model might want "BBL_20_2.0_2.0" but we have "BBL_20_2.0"
            if self.feature_names:
                # 1. Identify what we have
                available_cols = list(last_row.columns)
                
                # 2. Identify what is missing
                missing = [f for f in self.feature_names if f not in available_cols]
                
                # 3. Try to fuzzy match the missing ones
                if missing:
                    # Find our BBL/BBU columns
                    our_bbl = next((c for c in available_cols if c.startswith("BBL")), None)
                    our_bbu = next((c for c in available_cols if c.startswith("BBU")), None)
                    
                    # Find model's BBL/BBU requirements
                    model_bbl = next((f for f in missing if f.startswith("BBL")), None)
                    model_bbu = next((f for f in missing if f.startswith("BBU")), None)

                    # Rename if found
                    if our_bbl and model_bbl:
                        last_row.rename(columns={our_bbl: model_bbl}, inplace=True)
                    if our_bbu and model_bbu:
                        last_row.rename(columns={our_bbu: model_bbu}, inplace=True)

                # 4. Check again
                # If still missing, we really can't proceed
                final_missing = [f for f in self.feature_names if f not in last_row.columns]
                if final_missing:
                    # print(f"❌ Still missing: {final_missing}") # Debug line
                    return 0.5 
                
                X = last_row[self.feature_names]
            else:
                # Fallback for legacy models
                cols_to_exclude = ['datetime', 'timestamp', 'time', 'date', 'target']
                X = last_row.drop(columns=[c for c in cols_to_exclude if c in last_row.columns], errors='ignore')

            # Scale
            if self.scaler:
                X = self.scaler.transform(X)

            # Predict
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0]
                if len(probs) == 3:
                    return probs[2]
                else:
                    return probs[1]
            else:
                return float(self.model.predict(X)[0])
                
        except Exception as e:
            logger.error(f"Prediction Error: {e}")
            return 0.5

class Backtester:
    def __init__(self, config: dict):
        self.config = config # Keep original for reference
        self.symbol = config.get('symbol')
        self.timeframe = config.get('timeframe')
        
        # 🟢 Use the exact keys from your Pydantic model
        self.start_str = config.get('startDate')
        self.end_str = config.get('endDate')
        
        # Convert for internal logic
        self.start_date = pd.to_datetime(self.start_str).tz_localize(None) if self.start_str else None
        self.end_date = pd.to_datetime(self.end_str).tz_localize(None) if self.end_str else None
        
        self.initial_balance = float(config.get('initialBalance', 1000))
        self.model_name = config.get('mlModel')
        self.combination_rule = config.get('combinationRule', 'OR').upper()
        
        self.strategies = config.get('strategies', [])
        if not self.strategies and config.get('code'):
            self.strategies = [{"code": config.get('code'), "params": config.get('params', {})}]

        # 🟢 Risk Parameters (Synced with fix_pct)
        p = config.get('params', {})
        self.params = p
        
        self.tp_pct = p.get('take_profit', 0.06)
        self.sl_pct = p.get('stop_loss', 0.03)
        self.ts_pct = p.get('trailing_stop', 0.0)
        
        self.ml_limit_long = float(config.get('mlThresholdLong', 0.80))
        self.ml_limit_short = float(config.get('mlThresholdShort', 0.90))
        self.model_name = config.get('mlModel', 'stacking')


    def get_signal(self, row, strategies):
        """ Calculates the combined 'Vote' of all active strategies """
        votes = 0
        for strat in strategies:
            code = strat.get('code')
            p = strat.get('params', {})
            
            # 1. RSI Threshold
            if code == "rsi_threshold":
                rsi = row.get('rsi', 50)
                if rsi < p.get('oversold', 30): votes += 1
                elif rsi > p.get('overbought', 70): votes -= 1
            
            # 2. Bollinger Fade
            elif code == "bb_fade" or code == "bollinger_bands":
                if row['close'] < row.get('BBL_20_2.0', 0): votes += 1
                elif row['close'] > row.get('BBU_20_2.0', 999999): votes -= 1

            # 3. SMA Crossover
            elif code == "sma_crossover":
                if row.get('sma_50', 0) > row.get('sma_200', 0): votes += 1
                elif row.get('sma_50', 0) < row.get('sma_200', 0): votes -= 1

            # 4. MACD Crossover
            elif code == "macd_crossover":
                if row.get('MACD_12_26_9', 0) > row.get('MACDs_12_26_9', 0): votes += 1
                else: votes -= 1

            # 5. Supertrend
            elif code == "supertrend":
                if row.get('SUPERTd_7_3.0', 0) == 1: votes += 1
                else: votes -= 1

            # 6. ATR Breakout
            elif code == "atr_breakout":
                target = row.get('ema_20', 0) + (row.get('atr', 0) * p.get('multiplier', 1.5))
                if row['close'] > target: votes += 1
                elif row['close'] < (row.get('ema_20', 0) - (row.get('atr', 0) * p.get('multiplier', 1.5))): votes -= 1

            # 7. Stochastic
            elif code == "stoch":
                if row.get('STOCHk_14_3_3', 50) < 20: votes += 1
                elif row.get('STOCHk_14_3_3', 50) > 80: votes -= 1

            # 8. EMA Cloud
            elif code == "ema_cloud":
                if row.get('ema_9', 0) > row.get('ema_21', 0): votes += 1
                else: votes -= 1

            # 9. Price Action Breakout (20-period high/low)
            elif code == "pa_breakout":
                if row['close'] >= row.get('pa_high', 999999): votes += 1
                elif row['close'] <= row.get('pa_low', 0): votes -= 1

            # 10. Volume Profile (Volume Surge)
            elif code == "vol_profile":
                if row.get('volume', 0) > (row.get('vol_ma', 0) * p.get('threshold', 1.5)):
                    votes += (1 if row['close'] > row.get('sma_50', 0) else -1)

        # Combination Logic
        if self.combination_rule == "AND":
            if votes >= len(strategies): return 1
            if votes <= -len(strategies): return -1
            return 0
        else: # OR Logic
            if votes > 0: return 1
            if votes < 0: return -1
            return 0

    async def run(self):
        try:
            df = await self.load_data()
            
            # --- PRE-CALCULATE ALL INDICATORS FOR ALL 10 STRATEGIES ---
            df['sma_50'] = ta.sma(df['close'], 50)
            df['sma_200'] = ta.sma(df['close'], 200)
            df['ema_9'] = ta.ema(df['close'], 9)
            df['ema_20'] = ta.ema(df['close'], 20)
            df['ema_21'] = ta.ema(df['close'], 21)
            df['rsi'] = ta.rsi(df['close'], 14)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], 14)
            
            # Complex Indicators
            df = pd.concat([df, ta.bbands(df['close'], 20, 2)], axis=1)
            df = pd.concat([df, ta.macd(df['close'])], axis=1)
            df = pd.concat([df, ta.stoch(df['high'], df['low'], df['close'])], axis=1)
            df = pd.concat([df, ta.supertrend(df['high'], df['low'], df['close'], 7, 3)], axis=1)
            
            # Price Action & Volume
            df['pa_high'] = df['high'].rolling(20).max()
            df['pa_low'] = df['low'].rolling(20).min()
            df['vol_ma'] = ta.sma(df['volume'], 20)
            
            df = df.dropna()

            ml_model = None
            if self.model_name and self.model_name != "off":
                from app.predictors.model_factory import ModelFactory
                payload = ModelFactory.load_model(self.model_name, self.symbol, self.timeframe)
                if payload:
                    from app.backtest2 import RawModelAdapter
                    ml_model = RawModelAdapter(payload)

            balance, position, entry_price = self.initial_balance, None, 0
            equity_curve, trades = [], []

            for i in range(len(df)):
                row = df.iloc[i]
                current_time = str(row.name)
                signal = self.get_signal(row, self.strategies)

                gate_passed = True
                conf_score = 1.0
                if ml_model:
                    conf_score = ml_model.predict_direction(df_slice)
                    # Use 200EMA to determine which gate to use
                    is_short_trend = row['close'] < row['sma_200']
                    limit = self.ml_limit_short if is_short_trend else self.ml_limit_long
                    if conf_score < limit:
                        gate_passed = False


                if position == 'long':
                    pnl = (row['close'] - entry_price) / entry_price
                    if pnl >= self.tp_pct or pnl <= -self.sl_pct or signal == -1:
                        balance *= (1 + pnl - 0.0006)
                        trades.append({"type": "exit", "side": "long", "price": row['close'], "time": current_time, "pnl": round(pnl*100, 2)})
                        position = None
                
                elif position == 'short':
                    pnl = (entry_price - row['close']) / entry_price
                    if pnl >= self.tp_pct or pnl <= -self.sl_pct or signal == 1:
                        balance *= (1 + pnl - 0.0006)
                        trades.append({"type": "exit", "side": "short", "price": row['close'], "time": current_time, "pnl": round(pnl*100, 2)})
                        position = None

                

                if position is None and gate_passed:
                    if signal == 1:
                        position = 'long'; entry_price = row['close']
                        trades.append({"type": "buy", "price": entry_price, "time": current_time})
                    elif signal == -1:
                        position = 'short'; entry_price = row['close']
                        trades.append({"type": "sell", "price": entry_price, "time": current_time})
                
                elif position == 'long':
                    pnl = (row['close'] - entry_price) / entry_price
                    if pnl >= self.tp_pct or pnl <= -self.sl_pct or signal == -1:
                        balance *= (1 - 0.0006)
                        trades.append({"type": "exit", "side": "long", "price": row['close'], "time": current_time, "pnl": round(pnl*100, 2)})
                        position = None
                
                elif position == 'short':
                    pnl = (entry_price - row['close']) / entry_price
                    if pnl >= self.tp_pct or pnl <= -self.sl_pct or signal == 1:
                        balance *= (1 - 0.0006)
                        trades.append({"type": "exit", "side": "short", "price": row['close'], "time": current_time, "pnl": round(pnl*100, 2)})
                        position = None

                equity_curve.append({"time": current_time, "balance": round(balance, 2)})

            chart_df = df.reset_index().rename(columns={'index': 'time', 'datetime': 'time'})
            chart_df['time'] = chart_df['time'].astype(str)
            
            return {
                "status": "success",
                "metrics": {
                    "finalBalance": round(balance, 2),
                    "roi": round(((balance - self.initial_balance) / self.initial_balance) * 100, 2),
                    "totalTrades": len(trades)
                },
                "candleData": chart_df[['time', 'open', 'high', 'low', 'close']].to_dict('records'),
                "equityCurve": equity_curve,
                "trades": trades
            }
        except Exception as e:
            logger.error(f"Universal Engine Error: {e}")
            return {"status": "failed", "error": str(e)}
            

    async def load_data(self):
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

    

    def calculate_indicators(self, df):
        if len(df) < 50: return df 
        
        # --- MATCHING YOUR TRAINING LOGIC ---
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)
        
        st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
        if st is not None:
            df['st_trend'] = st.iloc[:, 1]
        else:
            df['st_trend'] = 0

        df['rsi'] = ta.rsi(df['close'], length=14)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None:
            df['adx'] = adx.iloc[:, 0]
        else:
            df['adx'] = 0
            
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None:
            df = pd.concat([df, bb], axis=1)
            
        return df.dropna()

    async def run(self):
        try:
            df = await self.load_data()
            df = self.calculate_indicators(df)
            
            if df is None or len(df) < 10:
                return {"status": "failed", "error": "Not enough data", "metrics": {"roi": -100}}

            ml_model = None
            if self.model_name and self.model_name != "off":
                payload = ModelFactory.load_model(self.model_name, self.symbol, self.timeframe)
                if payload:
                    ml_model = RawModelAdapter(payload)

            balance = self.initial_balance
            position = None
            entry_price = 0
            trades = []
            equity_curve = []

            for i in range(len(df)):
                row = df.iloc[i]
                df_slice = df.iloc[:i+1]
                current_time = str(row.name)

                is_short_trend = row['close'] < row.get('sma_200', row['close'])
                active_limit = self.ml_limit_short if is_short_trend else self.ml_limit_long
                
                # 1. Regime Detection
                regime = "TREND"
                gate_passed = True
                conf_score = 1.0
                
                if ml_model:
                    conf_score = ml_model.predict_direction(df_slice)
                    # Determine trend (Below 200EMA = Short Trend)
                    is_short_trend = row['close'] < row.get('sma_200', row['close'])
                    # Pick the right threshold from your config
                    limit = self.ml_limit_short if is_short_trend else self.ml_limit_long
                    
                    if conf_score < limit:
                        gate_passed = False

                # 2. GET STRATEGY VOTES (Uses your 10 strategies)
                signal = self.get_signal(row, self.strategies)
                
                # 3. Execution
                if position is None and gate_passed:
                    if signal == 1 and self.params.get('trade_direction') != 'SHORT':
                        position = 'long'
                        entry_price = row['close']
                        trades.append({'type': 'buy', 'price': entry_price, 'time': str(row.name)})
                    elif signal == -1 and self.params.get('trade_direction') != 'LONG':
                        position = 'short'
                        entry_price = row['close']
                        trades.append({'type': 'sell', 'price': entry_price, 'time': str(row.name)})
                
                elif position == 'long' and signal == -1:
                    balance *= (1 + (row['close'] - entry_price)/entry_price)
                    position = None
                    trades.append({'type': 'close_long', 'price': row['close'], 'time': str(row.name), 'balance': balance})

                elif position == 'short' and signal == 1:
                    balance *= (1 + (entry_price - row['close'])/entry_price)
                    position = None
                    trades.append({'type': 'close_short', 'price': row['close'], 'time': str(row.name), 'balance': balance})

                equity_curve.append({"time": current_time, "balance": round(balance, 2)})

            # 🟢 STEP 3: PREPARE FINAL STRUCTURE FOR NODE.JS
            # We must include 'candleData' and 'metrics' as top-level keys
            chart_df = df.reset_index().rename(columns={'index': 'time', 'datetime': 'time', 'timestamp': 'time'})
            chart_df['time'] = chart_df['time'].astype(str)
            candle_data = chart_df[['time', 'open', 'high', 'low', 'close']].to_dict('records')

            roi = ((balance - self.initial_balance) / self.initial_balance) * 100
            
            return {
                "status": "success",
                "metrics": {
                    "final_balance": round(balance, 2),
                    "roi": round(roi, 2),
                    "total_trades": len(trades) // 2
                },
                "candleData": candle_data,
                "trades": trades,
                "equityCurve": equity_curve,
                "initialBalance": self.initial_balance
            }
        except Exception as e:
            logger.error(f"Backtest Error: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "failed", "error": str(e), "metrics": {"roi": -100}}
(venv) root@intelligent-mendel:~/Project/ML# 
