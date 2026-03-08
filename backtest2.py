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

        self.params = config.get('params', {})
        
        self.initial_balance = float(config.get('initialBalance', 1000))
        self.model_name = config.get('mlModel')
        self.combination_rule = config.get('combinationRule', 'OR').upper()

        if 'trade_direction' not in self.params:
            self.params['trade_direction'] = 'BOTH'
        
        self.strategies = config.get('strategies', [])
        if not self.strategies and config.get('code'):
            self.strategies = [{"code": config.get('code'), "params": config.get('params', {})}]

        # 🟢 Risk Parameters (Synced with fix_pct)
        p = config.get('params', {})
        self.params = p
        
        raw_risk = config.get('risk_percentage', 100) # Default to 100 if missing
        self.risk_mult = float(raw_risk) / 100.0

        if self.risk_mult > 1.0: 
            self.risk_mult = 1.0
        
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
                is_short_trend = row['close'] < row.get('sma_200', row['close'])
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
                    "roi": round(((balance - self.initial_balance) / self.initial_balance) * 100, 2),
                    "totalTrades": len(trades),
                    "netProfit": round(balance - self.initial_balance, 2)
                },
                "candleData": df.reset_index().rename(columns={'index': 'time'}).to_dict('records'),
                "trades": trades,
                "equityCurve": equity_curve,
                "vetoed_signals": vetoed_logs,
                "initialBalance": self.initial_balance
            }
        except Exception as e:
            logger.error(f"Backtest Error: {e}")
            return {"status": "failed", "error": str(e)}
(venv) root@intelligent-mendel:~/Project/ML# 
