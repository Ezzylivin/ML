import pandas as pd
import pandas_ta as ta
import os
import logging
from fastapi import HTTPException
from app.predictors.model_factory import ModelFactory

logger = logging.getLogger("BacktestEngine")

class Backtester:
    def __init__(self, config: dict):
        self.symbol = config.get('symbol')
        self.timeframe = config.get('timeframe')
        self.start_date = pd.to_datetime(config.get('startDate')).tz_localize(None) if config.get('startDate') else None
        self.end_date = pd.to_datetime(config.get('endDate')).tz_localize(None) if config.get('endDate') else None
        self.initial_balance = config.get('initialBalance', 1000)
        self.risk_per_trade = config.get('riskPercentage', 1.0)
        self.model_name = config.get('mlModel')
        self.trend_strat = config.get('trend_strategy', 'atr_breakout')
        self.range_strat = config.get('range_strategy', 'bollinger_reversal')
        self.ml_conf_threshold = config.get('ml_confidence_threshold', 0.10)
        self.params = config.get('params', {})

    def load_data(self):
        # 🟢 1. Construct Path
        clean_symbol = self.symbol.replace("/", "-")
        # Try primary path
        file_path = f"data/{clean_symbol}-{self.timeframe}.csv"
        
        if not os.path.exists(file_path):
            # Try fallback path (e.g. data/SOL/USD-1h.csv or similar variants)
            fallback = f"data/{self.symbol.split('/')[0]}-USD-{self.timeframe}.csv"
            if os.path.exists(fallback):
                file_path = fallback
            else:
                logger.error(f"❌ File not found: {file_path}")
                raise HTTPException(status_code=404, detail=f"Missing data file: {file_path}")

        # 🟢 2. Load CSV
        logger.info(f"📂 Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Standardize columns
        df.columns = [c.lower() for c in df.columns]
        
        # 🟢 3. Parse Dates Robustly
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
            df.set_index('datetime', inplace=True)
        elif 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
            df.set_index('datetime', inplace=True)
        
        total_rows = len(df)
        
        # 🟢 4. Filter Date Range
        if self.start_date and self.end_date:
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
        
        filtered_rows = len(df)
        logger.info(f"📊 Data Loaded: {total_rows} rows -> {filtered_rows} rows after filter.")

        if df.empty:
            raise HTTPException(status_code=400, detail=f"No data found between {self.start_date} and {self.end_date}. File range: {total_rows} rows.")

        return df

    def calculate_indicators(self, df):
        # 🟢 Safe Indicator Calculation
        if len(df) < 50:
            return df # Not enough data for indicators

        # SMA
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)
        
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # MACD (With None Check)
        macd = ta.macd(df['close'])
        if macd is not None:
            df['macd'] = macd.iloc[:, 0]
            df['macds'] = macd.iloc[:, 2]
        else:
            df['macd'] = 0
            df['macds'] = 0
        
        # ATR
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None:
            df = pd.concat([df, bb], axis=1)
        
        return df.dropna()

    def run(self):
        try:
            df = self.load_data()
            df = self.calculate_indicators(df)
            
            if df.empty:
                return {"status": "failed", "error": "Not enough data for indicators", "metrics": {"roi": 0}}

            # Load ML Model
            ml_model = ModelFactory.load_model(self.model_name, self.symbol, self.timeframe)
            if not ml_model:
                logger.warning(f"⚠️ ML Model {self.model_name} not found. Running pure strategy.")

            balance = self.initial_balance
            position = None
            trades = []
            equity_curve = []

            logger.info(f"🚀 Starting Backtest Loop on {len(df)} candles...")

            for i in range(len(df)):
                row = df.iloc[i]
                
                # Slicing for ML input
                df_slice = df.iloc[:i+1]
                
                # 1. Determine Regime
                regime = "TREND"
                if ml_model:
                    buy_prob = ml_model.predict_direction(df_slice)
                    # Simple confidence logic: distance from 0.5
                    confidence = abs(buy_prob - 0.5) * 2 
                    
                    if confidence < self.ml_conf_threshold:
                        regime = "RANGE"
                    else:
                        regime = "TREND"
                
                # 2. Strategy Logic
                signal = 0 
                
                if regime == "TREND":
                    # ATR Breakout
                    if row['close'] > row['sma_50'] and row['rsi'] > 50:
                        signal = 1
                    elif row['close'] < row['sma_50'] and row['rsi'] < 50:
                        signal = -1
                else:
                    # Range Reversal
                    try:
                        # Dynamic column names from pandas_ta
                        lower = row.get("BBL_20_2.0", 0)
                        upper = row.get("BBU_20_2.0", 0)
                        if row['close'] <= lower: signal = 1
                        elif row['close'] >= upper: signal = -1
                    except: pass

                # 3. Execution
                if position is None:
                    if signal == 1 and (self.params.get('trade_direction') in ['BOTH', 'LONG']):
                        entry_price = row['close']
                        position = 'long'
                        trades.append({'type': 'buy', 'price': entry_price, 'time': str(row.name), 'regime': regime})
                    elif signal == -1 and (self.params.get('trade_direction') in ['BOTH', 'SHORT']):
                        entry_price = row['close']
                        position = 'short'
                        trades.append({'type': 'sell', 'price': entry_price, 'time': str(row.name), 'regime': regime})
                
                elif position == 'long':
                    if signal == -1: 
                        exit_price = row['close']
                        pnl = (exit_price - entry_price) / entry_price
                        balance *= (1 + pnl)
                        position = None
                        trades.append({'type': 'close_long', 'price': exit_price, 'pnl': pnl, 'balance': balance})
                
                elif position == 'short':
                    if signal == 1:
                        exit_price = row['close']
                        pnl = (entry_price - exit_price) / entry_price
                        balance *= (1 + pnl)
                        position = None
                        trades.append({'type': 'close_short', 'price': exit_price, 'pnl': pnl, 'balance': balance})

                equity_curve.append(balance)

            roi = ((balance - self.initial_balance) / self.initial_balance) * 100
            
            return {
                "status": "completed",
                "metrics": {
                    "final_balance": round(balance, 2),
                    "roi": round(roi, 2),
                    "total_trades": len(trades) // 2
                },
                "trades": trades[-5:]
            }
        except Exception as e:
            logger.error(f"Backtest Runtime Error: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "failed", "error": str(e), "metrics": {"roi": -100.0}}
