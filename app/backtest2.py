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
        self.params = config.get('params', {})
        self.ml_conf_threshold = float(config.get('ml_confidence_threshold', 0.10))

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

    def run(self):
        try:
            df = self.load_data()
            df = self.calculate_indicators(df)
            
            if len(df) < 10:
                return {"status": "failed", "error": "Not enough data", "metrics": {"roi": -100}}

            ml_model = None
            if self.model_name and self.model_name != "off":
                payload = ModelFactory.load_model(self.model_name, self.symbol, self.timeframe)
                if payload:
                    ml_model = RawModelAdapter(payload)

            balance = self.initial_balance
            position = None
            trades = []

            for i in range(len(df)):
                row = df.iloc[i]
                df_slice = df.iloc[:i+1]
                
                # 1. Regime Detection
                regime = "TREND"
                if ml_model:
                    buy_prob = ml_model.predict_direction(df_slice)
                    # Use confidence interval
                    if abs(buy_prob - 0.5) * 2 < self.ml_conf_threshold:
                        regime = "RANGE"
                
                # 2. Strategy Signal
                signal = 0 
                if regime == "TREND":
                    if row['close'] > row['sma_50'] and row['rsi'] > 50: signal = 1
                    elif row['close'] < row['sma_50'] and row['rsi'] < 50: signal = -1
                else:
                    try:
                        # Use fuzzy lookup for BBL/BBU here too just in case
                        bbl = next((row[c] for c in row.index if c.startswith("BBL")), 0)
                        bbu = next((row[c] for c in row.index if c.startswith("BBU")), 999999)
                        
                        if row['close'] <= bbl: signal = 1
                        elif row['close'] >= bbu: signal = -1
                    except: pass

                # 3. Execution
                if position is None:
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
