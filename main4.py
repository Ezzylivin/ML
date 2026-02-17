(venv) root@intelligent-mendel:~/Project/ML# cat main4.py 
import asyncio
import logging
import os
import joblib
import json
import re
import aiofiles
import sqlite3
import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt 
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Union
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
from fastapi.exceptions import RequestValidationError # <--- ADD THIS
# Update this line to include JSONResponse
from fastapi.responses import JSONResponse
# 🟢 SOCKET HELPERS
from app.services.socket_emitter import emit_log, emit_status

from app.backtest2 import Backtester 
from app.config2 import MODEL_DIR


MODEL_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEO-Engine")

ACTIVE_BOTS = {} 

# ==========================================
# 🗄️ 0. DATABASE HANDLER (Persistence Layer)
# ==========================================
# In main3.py / main4.py

class DatabaseHandler:
    DB_FILE = "bot_state.db"

    @classmethod
    def init_db(cls):
        """Initialize the SQLite database for persistence."""
        try:
            conn = sqlite3.connect(cls.DB_FILE)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS bot_sessions
                         (user_id TEXT PRIMARY KEY, config TEXT, balance REAL, 
                          positions TEXT, trade_history TEXT, equity_curve TEXT, logs TEXT,
                          status TEXT, last_update TIMESTAMP)''')
            conn.commit()
            conn.close()
            print("✅ Database initialized successfully.")
        except Exception as e:
            print(f"❌ Database Init Error: {e}")

    @classmethod
    def save_state(cls, user_id, bot_data):
        """Save the current bot state to DB."""
        # 🟢 SELF-HEALING: Ensure table exists before saving
         
        
        conn = sqlite3.connect(cls.DB_FILE)
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO bot_sessions 
                     (user_id, config, balance, positions, trade_history, equity_curve, logs, status, last_update)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (user_id, 
                   json.dumps(bot_data['config']), 
                   bot_data['balance'], 
                   json.dumps(bot_data['positions']), 
                   json.dumps(bot_data['trade_history']), 
                   json.dumps(bot_data.get('equityCurve', [])),
                   json.dumps(bot_data.get('logs', [])), 
                   bot_data['status'],
                   datetime.now().isoformat()))
        conn.commit()
        conn.close()

    @classmethod
    def load_state(cls, user_id):
        """Load bot state from DB if exists."""
        # 🟢 SELF-HEALING: Ensure table exists before loading
        
            
        conn = sqlite3.connect(cls.DB_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        try:
            c.execute("SELECT * FROM bot_sessions WHERE user_id = ?", (user_id,))
        except sqlite3.OperationalError:
            # 🟢 Table missing? Create it and return None (fresh start)
            conn.close()
            cls.init_db()
            return None

        row = c.fetchone()
        conn.close()
        
        if row:
            try:
                return {
                    "config": json.loads(row['config']),
                    "balance": row['balance'],
                    "positions": json.loads(row['positions']),
                    "trade_history": json.loads(row['trade_history']),
                    "equityCurve": json.loads(row['equity_curve']),
                    "logs": json.loads(row['logs']),
                    "status": "stopped" # Always load as stopped initially
                }
            except Exception as e:
                logger.error(f"DB Load Error: {e}")
                return None
        return None

# Ensure this is called at the module level
DatabaseHandler.init_db()
# --- REQUEST MODELS ---
class BotStartRequest(BaseModel):
    userId: str
    config: Dict[str, Any]

class BotStopRequest(BaseModel):
    userId: str

class BotClosePositionRequest(BaseModel):
    userId: str
    symbol: str


 
class Config:
        populate_by_name = True


class StrategyConfig(BaseModel):
    code: str
    params: Dict[str, Any]

class BacktestRequest(BaseModel):
    symbol: str
    timeframe: str
    startDate: str
    endDate: str
    initialBalance: float = 1000.0
    riskPercentage: float = 1.0
    mlModel: Optional[str] = None
    trend_strategy: Optional[str] = "atr_breakout"
    range_strategy: Optional[str] = "bollinger_reversal"
    ml_confidence_threshold: Optional[float] = 0.10
    trade_direction: Optional[str] = "BOTH"
    params: Optional[Dict[str, Any]] = {}

class ComboRequest(BaseModel):
    symbol: str
    timeframe: str
    startDate: str
    endDate: str
    initialBalance: float
    strategies: List[StrategyConfig]
    combinationRule: str = "OR"
    risk_percentage: float = 1.0
    take_profit: Optional[float] = 0.06
    stop_loss: Optional[float] = 0.03
    trailing_stop: Optional[float] = 0.02
    mlMode: Optional[str] = None 
    advanced_filters: Optional[Dict] = {}
    params: Optional[Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for user_id, bot in ACTIVE_BOTS.items():
        bot["status"] = "stopped"
        DatabaseHandler.save_state(user_id, bot)

app = FastAPI(title="NEO-V25.14 Sovereign Engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    # This prints the EXACT error to your nohup log
    print(f"❌ DATA ERROR: {exc.errors()}")
    print(f"❌ RECEIVED BODY: {exc.body}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

# ==========================================
# BACK TEST DATA
# ==========================================
def load_data_robust(symbol, timeframe):
    possible_paths = [
        f"data/{symbol.replace('/', '-')}-{timeframe}.csv",
        f"Project/ML/data/{symbol.replace('/', '-')}-{timeframe}.csv",
        f"{symbol.replace('/', '-')}-{timeframe}.csv"
    ]
    
    file_path = None
    for p in possible_paths:
        if os.path.exists(p):
            file_path = p
            break
            
    if not file_path:
        logger.error(f"❌ File not found. Searched: {possible_paths}")
        return None

    try:
        df = pd.read_csv(file_path)
        time_col = next((c for c in df.columns if c.lower() in ['timestamp', 'time', 'date']), None)
        
        if time_col:
            first_val = df[time_col].iloc[0]
            if isinstance(first_val, (int, float, np.number)):
                unit = 'ms' if first_val > 1e11 else 's'
                df[time_col] = pd.to_datetime(df[time_col], unit=unit, utc=True)
            else:
                df[time_col] = pd.to_datetime(df[time_col], utc=True, errors='coerce')
            
            df.rename(columns={time_col: 'timestamp'}, inplace=True)
            df.set_index('timestamp', inplace=True)
            df.columns = [c.lower() for c in df.columns]
            df.sort_index(inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            return df
            
    except Exception as e:
        logger.error(f"Data Load Error: {e}")
    return None

def calculate_strategy_signal(df, code, params):
    close = df['close']
    high = df['high']
    low = df['low']
    vol = df['volume']
    signal = pd.Series(0, index=df.index)
    
    try:
        if code == "rsi_threshold":
            length = int(params.get('rsi_length', 14))
            rsi = ta.rsi(close, length=length)
            signal[rsi < params.get('oversold', 30)] = 1
            signal[rsi > params.get('overbought', 70)] = -1
            
        elif code == "sma_crossover":
            fast = ta.sma(close, length=int(params.get('fast_sma', 50)))
            slow = ta.sma(close, length=int(params.get('slow_sma', 200)))
            signal[fast > slow] = 1
            signal[fast < slow] = -1
            
        elif code == "bollinger_bands" or code == "bb_fade":
            length = int(params.get('bb_period', 20))
            std = float(params.get('bb_std', 2.0))
            bb = ta.bbands(close, length=length, std=std)
            if bb is not None:
                lower = bb.iloc[:, 0]
                upper = bb.iloc[:, 2]
                signal[close < lower] = 1
                signal[close > upper] = -1

        elif code == "atr_breakout":
            length = int(params.get('atr_length', 14))
            mult = float(params.get('multiplier', 1.5))
            atr = ta.atr(high, low, close, length=length)
            sma = ta.sma(close, length=20)
            signal[close > (sma + atr * mult)] = 1
            signal[close < (sma - atr * mult)] = -1

        elif code == "macd_crossover":
            fast = int(params.get('fast', 12))
            slow = int(params.get('slow', 26))
            sig = int(params.get('signal', 9))
            macd = ta.macd(close, fast=fast, slow=slow, signal=sig)
            if macd is not None:
                macd_line = macd.iloc[:, 0]
                sig_line = macd.iloc[:, 2]
                signal[macd_line > sig_line] = 1
                signal[macd_line < sig_line] = -1
        
        elif code == "stoch":
            k_len = int(params.get('k_period', 14))
            d_len = int(params.get('d_period', 3))
            stoch = ta.stoch(high, low, close, k=k_len, d=d_len)
            if stoch is not None:
                k, d = stoch.iloc[:, 0], stoch.iloc[:, 1]
                signal[(k > d) & (k < 30)] = 1
                signal[(k < d) & (k > 70)] = -1

        elif code == "supertrend":
            length = int(params.get('st_atr', 10))
            factor = float(params.get('st_factor', 3.0))
            st = ta.supertrend(high, low, close, length=length, multiplier=factor)
            if st is not None:
                direction = st.iloc[:, 1] 
                signal[direction == 1] = 1
                signal[direction == -1] = -1

        elif code == "ema_cloud":
            fast_len = int(params.get('fast_ema', 9))
            slow_len = int(params.get('slow_ema', 21))
            fast_ema = ta.ema(close, length=fast_len)
            slow_ema = ta.ema(close, length=slow_len)
            signal[fast_ema > slow_ema] = 1
            signal[fast_ema < slow_ema] = -1

        elif code == "pa_breakout":
            lookback = int(params.get('lookback', 20))
            highest = high.rolling(lookback).max().shift(1)
            lowest = low.rolling(lookback).min().shift(1)
            signal[close > highest] = 1
            signal[close < lowest] = -1

        elif code == "vol_profile":
            vol_ma_len = int(params.get('vol_ma', 20))
            threshold = float(params.get('threshold', 1.5))
            vol_ma = ta.sma(vol, length=vol_ma_len)
            vol_spike = vol > (vol_ma * threshold)
            price_up = close > close.shift(1)
            signal[vol_spike & price_up] = 1
            signal[vol_spike & ~price_up] = -1

    except Exception as e:
        logger.error(f"Strategy Error ({code}): {e}")
    
    return signal.fillna(0)


# ==========================================
# 🧠 1. NEURAL PREDICTOR
# ==========================================
class DiagnosticLayer:
    @staticmethod
    def render_progress(current, target, reverse=False):
        """Generates a [|||||.....] visual bar."""
        try:
            # Calculate completion percentage
            pct = (target / current) if not reverse else (current / target)
            pct = min(1.0, max(0.0, pct))
            filled = int(pct * 10)
            bar = "┃" + "█" * filled + "░" * (10 - filled) + "┃"
            return f"{bar} {int(pct * 100)}%"
        except: return "[----------] 0%"

    @staticmethod
    def get_pending_conditions(df, config, conf, ui_limit):
        try:
            current_price = float(df['close'].iloc[-1])
            ema200_val = ta.ema(df['close'], length=200).iloc[-1]
            is_uptrend = current_price > ema200_val
            
            if conf < ui_limit:
                return f"🛑 AI VETO: Confidence {DiagnosticLayer.render_progress(conf, ui_limit, True)}"

            pending = []
            strategies = config.get('strategies', [])
            for strat in strategies:
                code = strat.get('code')
                p = strat.get('params', {})
                if code == "bb_fade":
                    bb = ta.bbands(df['close'], length=int(p.get('bb_period', 20)), std=float(p.get('bb_std', 2.0)))
                    target = bb.iloc[-1, 0] if is_uptrend else bb.iloc[-1, 2]
                    bar = DiagnosticLayer.render_progress(current_price, target, is_uptrend)
                    pending.append(f"Price to BB: {bar}")
                elif code == "stoch":
                    k = ta.stoch(df['high'], df['low'], df['close']).iloc[-1, 0]
                    target = 20 if is_uptrend else 80
                    bar = DiagnosticLayer.render_progress(k, target, not is_uptrend)
                    pending.append(f"Stoch to Trigg: {bar}")
            
            rule = config.get('combinationRule', 'OR')
            return f"🔍 TARGETS ({rule}): " + " | ".join(pending)
        except Exception: return "🔍 Scanning Market Conditions..."


class NeuralPredictor:
    # 🟢 1. Initialize an in-memory cache
    _model_cache = {}

    @staticmethod
    def get_prediction(model_id: str, df: pd.DataFrame) -> float:
        try:
            # Feature Engineering: Use a 20-bar window for better stability
            recent = df.tail(20)
            if len(recent) < 10:
                return 0.5
            
            momentum = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
            
            # 🟢 2. Caching Logic
            # Only load from disk if the model isn't already in memory
            if model_id not in NeuralPredictor._model_cache:
                model_path = f"./models/{model_id}_model.pkl"
                
                if os.path.exists(model_path):
                    logger.info(f"🧠 Loading Neural Model into Cache: {model_id}")
                    NeuralPredictor._model_cache[model_id] = joblib.load(model_path)
                else:
                    # If file doesn't exist, use None to trigger fallback
                    NeuralPredictor._model_cache[model_id] = None

            # 🟢 3. Execute Prediction
            cached_model = NeuralPredictor._model_cache.get(model_id)
            
            if cached_model:
                # Scikit-learn expects 2D array for predictions
                prediction = cached_model.predict_proba([[momentum]])[0][1]
                return float(prediction)
            
            # 🟢 4. Intelligent Fallback (Sigmoid)
            # Ensures the bot remains operational even if the .pkl is missing
            base = 1.0 / (1.0 + np.exp(-momentum * 100))
            return float(min(0.99, max(0.01, base))) # Clamp between 1% and 99%
            
        except Exception as e:
            logger.error(f"🧠 Neural Predictor Error: {e}")
            return 0.5

    @classmethod
    def clear_cache(cls):
        """Call this if you upload a new model file to refresh it."""
        cls._model_cache = {}
        logger.info("🧠 Neural Cache Purged.")


async def execute_backtest_logic(data: BacktestRequest):
    """Reusable backtest engine that prioritizes CSV files."""
    try:
        df = None
        # 1. Try Loading from Local CSV
        csv_path = f"data/{data.symbol.replace('/', '-')}_{data.timeframe}.csv"
        
        if os.path.exists(csv_path):
            logger.info(f"📂 Loading historical data from {csv_path}")
            df = pd.read_csv(csv_path)
            # Normalize column names
            df.columns = [c.lower() for c in df.columns]
            rename_map = {'timestamp': 'time', 'date': 'time', 'volume': 'vol'}
            df.rename(columns=rename_map, inplace=True)
            
            # Ensure time is datetime for filtering
            df['time'] = pd.to_datetime(df['time'])
            
            # Filter by Date Range
            start_dt = pd.to_datetime(data.start_date).replace(tzinfo=None)
            end_dt = pd.to_datetime(data.end_date).replace(tzinfo=None)
            
            # If CSV time is TZ-aware, strip it for comparison or ensure match
            if df['time'].dt.tz is not None:
                df['time'] = df['time'].dt.tz_localize(None)
                
            df = df[(df['time'] >= start_dt) & (df['time'] <= end_dt)]
            
            # Convert time back to milliseconds for logic consistency if needed, or keep as is
            # StrategyBrain expects DataFrame. The indices might need reset.
            df.reset_index(drop=True, inplace=True)

        # 2. Fallback to CCXT if CSV missing
        if df is None or df.empty:
            logger.info("⚠️ CSV not found or empty. Falling back to CCXT.")
            async with ccxt.coinbase() as exchange:
                since = exchange.parse8601(data.start_date)
                ohlcv = await exchange.fetch_ohlcv(data.symbol.replace('-', '/'), data.timeframe, since=since, limit=1000)
                df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
                df['time'] = pd.to_datetime(df['time'], unit='ms')

        if df.empty:
            return {"status": "error", "message": "No data found for backtest range"}

        balance, position, trades, curve = data.initial_capital, None, [], []
        
        # Prepare Config for StrategyBrain
        sim_config = {
            "strategies": data.strategies,
            "comboConfig": data.comboConfig or {},
            "mlModel": "stacking" 
        }

        # 🚀 THE SIMULATION LOOP
        for i in range(50, len(df)): # Start at 50 to allow indicators to warm up
            window = df.iloc[:i+1].copy()
            
            # Calculate Indicators on the window
            # Note: For speed, you might want to pre-calc indicators on full DF, but this mimics live behavior
            sig, _, _, _ = StrategyBrain.calculate_signals(window, sim_config, 0.5, 0.5)
            
            row = df.iloc[i]
            price = row['close']
            ts = row['time'].isoformat()
            
            # Execution Logic
            if sig == 1 and position is None:
                position = {"type": "long", "entry": price, "size": balance / price}
                trades.append({"type": "buy", "price": price, "time": ts})
            elif sig == -1 and position is None:
                position = {"type": "short", "entry": price, "size": balance / price}
                trades.append({"type": "short", "price": price, "time": ts})
            elif (sig == -1 and position and position['type'] == 'long') or (sig == 1 and position and position['type'] == 'short'):
                pnl = (price - position['entry']) * position['size'] if position['type'] == 'long' else (position['entry'] - price) * position['size']
                balance += pnl; position = None
                trades.append({"type": "exit", "price": price, "time": ts, "pnl": pnl})
            
            curve.append({"time": ts, "equity": balance})
            
        return {
            "status": "success", 
            "metrics": {"final_balance": round(balance, 2), "trade_count": len(trades)}, 
            "trades": trades, 
            "equity_curve": curve
        }
    except Exception as e: 
        logger.error(f"Backtest Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# Packet Data
# ==========================================

async def process_data_packet(df: pd.DataFrame, strategies: list) -> list:
    """Calculates all indicators and sanitizes candle objects for the chart."""
    for strat in strategies:
        code = strat.get('code')
        p = strat.get('params', {})
        try:
            if code == 'bb_fade':
                bb = ta.bbands(df['close'], length=int(p.get('bb_period', 20)), std=float(p.get('bb_std', 2.0)))
                df['bb_lower'], df['bb_upper'] = bb.iloc[:, 0], bb.iloc[:, 2]
            elif code == 'ema_cloud':
                df['ema_fast'] = ta.ema(df['close'], length=int(p.get('fast_ema', 9)))
                df['ema_slow'] = ta.ema(df['close'], length=int(p.get('slow_ema', 21)))
            elif code == 'sma_crossover':
                df['sma_fast'] = ta.sma(df['close'], length=int(p.get('fast_sma', 50)))
                df['sma_slow'] = ta.sma(df['close'], length=int(p.get('slow_sma', 200)))
            elif code == 'supertrend':
                st = ta.supertrend(df['high'], df['low'], df['close'], length=int(p.get('st_atr', 10)), multiplier=float(p.get('st_factor', 3.0)))
                df['supertrend'] = st.iloc[:, 0]
            elif code == 'rsi_threshold':
                df['rsi'] = ta.rsi(df['close'], length=int(p.get('rsi_length', 14)))
        except Exception: continue

    # Package into JSON-ready list
    keys = ['bb_lower', 'bb_upper', 'ema_fast', 'ema_slow', 'sma_fast', 'sma_slow', 'supertrend', 'rsi']
    candles_to_send = []
    for _, row in df.tail(100).iterrows():
        # Ensure time is an integer (Unix seconds)
        ts = int(row['time']) if 'time' in row else int(row.name.timestamp())
        c_obj = {"time": ts, "open": row['open'], "high": row['high'], "low": row['low'], "close": row['close']}
        for k in keys:
            if k in row and not pd.isna(row[k]): c_obj[k] = float(row[k])
        candles_to_send.append(c_obj)
    return candles_to_send

# ==========================================
# 🧠 2. SHARED STRATEGY BRAIN
# ==========================================
class StrategyBrain:
    @staticmethod
    def calculate_signals(df: pd.DataFrame, config: Dict[str, Any], l_thresh: float, s_thresh: float):
        active_thoughts, votes = [], 0
        strategies = config.get('strategies', [])
        current_price = df['close'].iloc[-1]

        # 🟢 1. GLOBAL INDICATORS (Foundational)
        ema20 = ta.ema(df['close'], length=20).iloc[-1]
        ema50 = ta.ema(df['close'], length=50).iloc[-1]
        ema200 = ta.ema(df['close'], length=200).iloc[-1]
        bb = ta.bbands(df['close'], length=20, std=2.0)
        lower, mid, upper = bb.iloc[-1, 0], bb.iloc[-1, 1], bb.iloc[-1, 2]
        pr = int((current_price - lower) / (upper - lower) * 100)

        # 🟢 2. FULL 10-STRATEGY DYNAMIC EVALUATION
        for strat in strategies:
            code = strat.get('code')
            p = strat.get('params', {})
            try:
                # 1. RSI Threshold
                if code == "rsi_threshold":
                    rsi = ta.rsi(df['close'], length=int(p.get('rsi_length', 14))).iloc[-1]
                    if rsi < p.get('oversold', 30): 
                        votes += 1; active_thoughts.append(f"RSI Low ({int(rsi)})")
                    elif rsi > p.get('overbought', 70): 
                        votes -= 1; active_thoughts.append(f"RSI High ({int(rsi)})")

                # 2. Stochastic Oscillator
                elif code == "stoch":
                    stoch = ta.stoch(df['high'], df['low'], df['close'], k=int(p.get('k_period', 14)))
                    k_val = stoch.iloc[-1, 0]
                    if k_val < 20: 
                        votes += 1; active_thoughts.append(f"Stoch Low ({int(k_val)})")
                    elif k_val > 80: 
                        votes -= 1; active_thoughts.append(f"Stoch High ({int(k_val)})")

                # 3. MACD Crossover (Standardized Indexing)
                elif code == "macd_crossover":
                    macd_df = ta.macd(df['close'], fast=int(p.get('fast', 12)), slow=int(p.get('slow', 26)), signal=int(p.get('signal', 9)))
                    if macd_df.iloc[-1, 0] > macd_df.iloc[-1, 2]: 
                        votes += 1; active_thoughts.append("MACD Bullish")
                    else:
                        votes -= 1

                # 4. Supertrend
                elif code == "supertrend":
                    st_df = ta.supertrend(df['high'], df['low'], df['close'], length=int(p.get('st_atr', 10)), multiplier=float(p.get('st_factor', 3.0)))
                    if st_df.iloc[-1, 1] == 1: 
                        votes += 1; active_thoughts.append("SuperTrend Long")
                    else:
                        votes -= 1

                # 5. EMA Cloud
                elif code == "ema_cloud":
                    if current_price > ema50: 
                        votes += 1; active_thoughts.append("Above EMA Cloud")
                    else:
                        votes -= 1

                # 6. SMA Crossover
                elif code == "sma_crossover":
                    sma_fast = ta.sma(df['close'], length=int(p.get('fast_sma', 50))).iloc[-1]
                    sma_slow = ta.sma(df['close'], length=int(p.get('slow_sma', 200))).iloc[-1]
                    if sma_fast > sma_slow: 
                        votes += 1; active_thoughts.append("SMA Golden Cross")

                # 7. Bollinger Band Fade
                elif code == "bb_fade":
                    if current_price < lower: 
                        votes += 1; active_thoughts.append("Price < BB Floor")
                    elif current_price > upper: 
                        votes -= 1; active_thoughts.append("Price > BB Ceiling")

                # 8. ATR Breakout
                elif code == "atr_breakout":
                    atr = ta.atr(df['high'], df['low'], df['close'], length=int(p.get('atr_length', 14))).iloc[-1]
                    if current_price > (ema20 + atr * float(p.get('multiplier', 1.5))):
                        votes += 1; active_thoughts.append("ATR Breakout")

                # 9. Price Action Breakout
                elif code == "pa_breakout":
                    lookback = int(p.get('lookback', 20))
                    if current_price >= df['high'].tail(lookback).max():
                        votes += 1; active_thoughts.append("PA High Break")

                # 10. Volume Profile
                elif code == "vol_profile":
                    vol_ma = ta.sma(df['volume'], length=int(p.get('vol_ma', 20))).iloc[-1]
                    if df['volume'].iloc[-1] > vol_ma * float(p.get('threshold', 1.5)):
                        votes += (1 if current_price > mid else -1)
                        active_thoughts.append("Volume Surge")

            except Exception: continue

        # 🚀 3. DYNAMIC DUAL-GATE LOGIC (ML Filtering)
        is_short = current_price < ema200
        ui_limit = float(config.get('mlThresholdShort', 0.90)) if is_short else float(config.get('mlThresholdLong', 0.80))
        
        conf = NeuralPredictor.get_prediction(config.get('mlModel', 'stacking'), df)
        gate_passed = conf >= ui_limit
        
        logic_desc = f"📊 LOGIC: {'SHORT' if is_short else 'LONG'} GATE {'PASSED' if gate_passed else 'VETOED'} ({int(conf*100)}% vs {int(ui_limit*100)}% UI Limit) {'🟢' if gate_passed else '🔴'}"

        # 🎯 4. INTENT LOGIC
        signal_names = " + ".join(active_thoughts) if active_thoughts else "Scanning Setup"
        gap = int(abs(current_price - ema50))
        intent_desc = f"🎯 INTENT: STALKING {'SHORT' if is_short else 'LONG'} ({signal_names} | Gap: ${gap}) {'🔴' if is_short else '🟢'}"

        # 📡 5. MARKET CONTEXT (Rich Descriptions)
        trend_dist = current_price - ema200
        trend_pct = (trend_dist / ema200) * 100
        if trend_dist > 0:
            trend_str = "STRONG UPTREND" if trend_pct > 1.0 else "WEAK UPTREND"
        else:
            trend_str = "STRONG DOWNTREND" if trend_pct < -1.0 else "WEAK DOWNTREND"
            
        trend_text = f"📡 TREND: {trend_str} (Price is ${int(abs(trend_dist))} {'above' if trend_dist > 0 else 'below'} 200EMA)"

        spread = ema20 - ema50
        bias_str = "BULLISH EXPANSION" if spread > 0 else "BEARISH CONTRACTION"
        bias_text = f"⚖️ BIAS: {bias_str} (Fast EMA is ${int(abs(spread))} {'above' if spread > 0 else 'below'} Slow EMA)"

        if pr >= 80: mindset_str = "⚠️ OVEREXTENDED (Expensive)"
        elif pr <= 20: mindset_str = "🎯 ACCUMULATION ZONE (Cheap)"
        else: mindset_str = "⚖️ EQUILIBRIUM (Balanced)"
        
        mindset_text = f"🤖 MINDSET: {mindset_str} - Price is at {pr}% of Bollinger Range"

        numeric_details = {
            "market": {
                "logic": logic_desc, 
                "intent": intent_desc,
                "trend": trend_text,
                "bias": bias_text,
                "mindset": mindset_text
            }
        }

        # 🎯 6. FINAL SIGNAL CALCULATION
        # Must pass Gate AND have positive/negative votes matching trend direction
        final_sig = 0
        if gate_passed:
            if votes > 0 and not is_short: 
                final_sig = 1
            elif votes < 0 and is_short: 
                final_sig = -1
        
        return final_sig, active_thoughts, numeric_details, conf





# ==========================================
# 🚀 3. THE HEARTBEAT (Dynamic Calculation Loop)
# ==========================================
# 🟢 1. Initialize Global Exchange at the top of main4.py
global_exchange = ccxt.coinbase({'enableRateLimit': True})

async def live_neural_heartbeat(user_id: str):
    last_log = 0
    
    # 🟢 2. INITIAL SETUP
    if user_id in ACTIVE_BOTS:
        if "equityCurve" not in ACTIVE_BOTS[user_id]: 
            ACTIVE_BOTS[user_id]["equityCurve"] = [{"time": datetime.now().isoformat(), "balance": ACTIVE_BOTS[user_id]["balance"], "confidence": 50}]
        if "logs" not in ACTIVE_BOTS[user_id]: ACTIVE_BOTS[user_id]["logs"] = []

    try:
        while user_id in ACTIVE_BOTS and ACTIVE_BOTS[user_id]["status"] == "running":
            bot = ACTIVE_BOTS[user_id]
            config = bot.get('config', {})
            strategies = config.get('strategies', [])
            upnl = 0
            current_equity = bot['balance']
            markers = []
            candles_to_send = []
            current_confidence = 50
            new_data_point = {}
            
            try:
                # 🟢 3. DATA FETCHING (Zero-Drift Patch)
                ohlcv_raw = await fetch_live_candles_ccxt(config['symbol'], config.get('timeframe', '1h'), 250)
                
                if not ohlcv_raw:
                    logger.warning(f"⚠️ Market feed unstable for {config['symbol']}")
                    emit_log(user_id, "⚠️ Market Data Feed Unstable - Retrying...")
                    await asyncio.sleep(10); continue

                # Patch last candle with absolute real-time price
                ticker = await global_exchange.fetch_ticker(config['symbol'].replace('-', '/'))
                current_price = float(ticker['last'])
                ohlcv_raw[-1]['close'] = current_price

                upnl = sum([(current_price - p['entry']) * p['size'] if p['type'] == 'long' else (p['entry'] - current_price) * p['size'] for p in bot['positions']])
                current_equity = bot['balance'] + upnl

                # Real-time drift log
                last_ts = datetime.fromtimestamp(ohlcv_raw[-1]['time'], tz=timezone.utc)
                drift = (datetime.now(timezone.utc) - last_ts).total_seconds()
                logger.info(f"📡 PULSE: ${current_price} | Drift: {drift}s")

                # 🟢 4. DYNAMIC INDICATOR CALCULATION
                df = pd.DataFrame(ohlcv_raw)
                for strat in strategies:
                    code, p = strat.get('code'), strat.get('params', {})
                    try:
                        if code == 'bb_fade':
                            bb = ta.bbands(df['close'], length=int(p.get('bb_period', 20)), std=float(p.get('bb_std', 2.0)))
                            df['bb_lower'], df['bb_upper'] = bb.iloc[:, 0], bb.iloc[:, 2]
                        elif code == 'ema_cloud':
                            df['ema_fast'] = ta.ema(df['close'], length=int(p.get('fast_ema', 9)))
                            df['ema_slow'] = ta.ema(df['close'], length=int(p.get('slow_ema', 21)))
                        elif code == 'sma_crossover':
                            df['sma_fast'] = ta.sma(df['close'], length=int(p.get('fast_sma', 50)))
                            df['sma_slow'] = ta.sma(df['close'], length=int(p.get('slow_sma', 200)))
                        elif code == 'supertrend':
                            st = ta.supertrend(df['high'], df['low'], df['close'], length=int(p.get('st_atr', 10)), multiplier=float(p.get('st_factor', 3.0)))
                            df['supertrend'] = st.iloc[:, 0]
                        elif code == 'pa_breakout':
                            lb = int(p.get('lookback', 20))
                            df['pa_high'], df['pa_low'] = df['high'].rolling(lb).max(), df['low'].rolling(lb).min()
                        elif code == 'atr_breakout':
                            atr = ta.atr(df['high'], df['low'], df['close'], length=int(p.get('atr_length', 14)))
                            ema20 = ta.ema(df['close'], 20)
                            df['atr_upper'], df['atr_lower'] = ema20 + (atr * float(p.get('multiplier', 1.5))), ema20 - (atr * float(p.get('multiplier', 1.5)))
                        elif code == 'rsi_threshold':
                            df['rsi'] = ta.rsi(df['close'], length=int(p.get('rsi_length', 14)))
                        elif code == 'stoch':
                            stoch = ta.stoch(df['high'], df['low'], df['close'], k=int(p.get('k_period', 14)))
                            df['stoch_k'], df['stoch_d'] = stoch.iloc[:, 0], stoch.iloc[:, 1]
                        elif code == 'macd_crossover':
                            macd = ta.macd(df['close'], fast=int(p.get('fast', 12)), slow=int(p.get('slow', 26)), signal=int(p.get('signal', 9)))
                            df['macd'], df['macd_signal'] = macd.iloc[:, 0], macd.iloc[:, 2]
                        elif code == 'vol_profile':
                            df['vol_ma'] = ta.sma(df['volume'], length=int(p.get('vol_ma', 20)))
                    except Exception: continue

                # 🟢 5. SIGNAL PROCESSING & DIAGNOSTICS
                sig, thoughts, nums, score = StrategyBrain.calculate_signals(df, config, 0.5, 0.5)
                ui_limit = float(config.get('mlThresholdLong', 0.5)) if current_price > ta.ema(df['close'], 200).iloc[-1] else float(config.get('mlThresholdShort', 0.5))
                waiting_msg = DiagnosticLayer.get_pending_conditions(df, config, score, ui_limit)


                emit_log(user_id, nums['market']['trend'])
                emit_log(user_id, waiting_msg) # Shows INTENT (e.g., Stalking Long)
                emit_log(user_id, nums['market']['logic']) # Shows GATE status

                clean_curve = [p for p in bot.get("equityCurve", []) if p and isinstance(p, dict) and 'time' in p]

                if datetime.now().timestamp() - last_log >= 60:
                    
                    bot["equityCurve"].append({"time": datetime.now().isoformat(), "balance": round(current_equity, 2), "confidence": int(score * 100)})
                    if len(bot["equityCurve"]) > 100: bot["equityCurve"].pop(0)
                    bot["logs"] = ([{"time": datetime.now().isoformat(), "message": m} for m in [nums['market']['trend'], waiting_msg]] + bot["logs"])[:300]
                    DatabaseHandler.save_state(user_id, bot)
                    last_log = datetime.now().timestamp()

                # 🟢 6. TRADE EXECUTION LOGIC (Standardized PnL Fix)
               


                # EXECUTION: Open Trades
                if len(bot['positions']) < int(config.get('maxPyramiding', 1)):
                    risk_val = float(config.get('riskPercentage', 1)) / 100
                    size = (bot['balance'] * risk_val) / current_price
                    
                    if sig == 1: # LONG
                        pos = {"type": "long", "entry": current_price, "size": size, "time": datetime.now().isoformat(),
                               "sl": current_price * (1 - float(config['params'].get('stop_loss', 0.05))),
                               "tp": current_price * (1 + float(config['params'].get('take_profit', 0.10)))}
                        bot['positions'].append(pos)
                        bot['trade_history'].append({"type": "buy", "price": current_price, "time": datetime.now().isoformat()})
                        emit_log(user_id, f"🚀 LONG EXECUTED @ ${current_price}")
                        DatabaseHandler.save_state(user_id, bot)
                    elif sig == -1 and config.get('enable_shorting', True): # SHORT
                        pos = {"type": "short", "entry": current_price, "size": size, "time": datetime.now().isoformat(),
                               "sl": current_price * (1 + float(config['params'].get('stop_loss', 0.05))),
                               "tp": current_price * (1 - float(config['params'].get('take_profit', 0.10)))}
                        bot['positions'].append(pos)
                        bot['trade_history'].append({"type": "short", "price": current_price, "time": datetime.now().isoformat()})
                        emit_log(user_id, f"🔻 SHORT EXECUTED @ ${current_price}")
                        DatabaseHandler.save_state(user_id, bot)

                # EXECUTION: Check Exits
                active_pos = bot['positions'][:]
                for pos in active_pos:
                    pnl, closed = 0, False
                    if pos['type'] == 'long':
                        if current_price >= pos['tp']: pnl = (current_price - pos['entry']) * pos['size']; closed = True; emit_log(user_id, f"💰 TP HIT: +${round(pnl, 2)}")
                        elif current_price <= pos['sl']: pnl = (current_price - pos['entry']) * pos['size']; closed = True; emit_log(user_id, f"🛑 SL HIT: -${round(abs(pnl), 2)}")
                    elif pos['type'] == 'short':
                        if current_price <= pos['tp']: pnl = (pos['entry'] - current_price) * pos['size']; closed = True; emit_log(user_id, f"💰 TP HIT: +${round(pnl, 2)}")
                        elif current_price >= pos['sl']: pnl = (pos['entry'] - current_price) * pos['size']; closed = True; emit_log(user_id, f"🛑 SL HIT: -${round(abs(pnl), 2)}")

                    if closed:
                        bot['balance'] += pnl; bot['positions'].remove(pos)
                        bot['trade_history'].append({"type": "exit", "price": current_price, "pnl": pnl, "time": datetime.now().isoformat()})
                        DatabaseHandler.save_state(user_id, bot)

                

                # 🟢 7. PACKAGING FOR UI
                candles_to_send = []
                keys = ['bb_lower', 'bb_upper', 'ema_fast', 'ema_slow', 'sma_fast', 'sma_slow', 'supertrend', 'pa_high', 'pa_low', 'atr_upper', 'atr_lower', 'rsi', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'vol_ma']
                for _, row in df.tail(100).iterrows():
                    c_obj = {"time": int(row['time']), "open": row['open'], "high": row['high'], "low": row['low'], "close": row['close']}
                    for k in keys:
                        if k in row and not pd.isna(row[k]): c_obj[k] = float(row[k])
                    candles_to_send.append(c_obj)

                markers = [{"time": int(c['time']), "position": "belowBar", "color": "#10b981", "text": thoughts[0] if thoughts else ""} for c in ohlcv_raw[-1:] if thoughts]

                emit_status(user_id, {
                    "status": "running", 
                    "currentBalance": round(current_equity, 2), 
                    "unrealizedPnl": round(upnl, 2),
                    "activePositions": bot['positions'], 
                    "tradeMarkers": bot['trade_history'] + markers,
                    "equityCurve": clean_curve, 
                    "startedAt": bot.get("startedAt"), 
                    "candles": candles_to_send
                })

                if datetime.now().timestamp() - last_log >= 60:
    # This new point adds to the history shown on your line charts
                   new_data_point = {
                   "time": datetime.now().isoformat(), 
                   "balance": round(current_equity, 2), 
                   "confidence": current_confidence # This draws the historical confidence line
                }
    
                bot["equityCurve"].append(new_data_point)
    
    # Prune history to keep the chart snappy (last 100 points)
                if len(bot["equityCurve"]) > 100: 
                    bot["equityCurve"].pop(0)

    # Persist to disk
                DatabaseHandler.save_state(user_id, bot)
                last_log = datetime.now().timestamp()


                await asyncio.sleep(15)

            except Exception as e:
                logger.error(f"❌ Loop Sync Error: {e}"); await asyncio.sleep(10)
    finally:
        logger.info(f"🔌 Heartbeat loop terminated for {user_id}")
        
# =============================================================
# ENDPOINTS
# =============================================================

async def fetch_live_candles_ccxt(symbol: str, timeframe: str, limit: int):
    async with ccxt.coinbase() as ex:
        try:
            ohlcv = await ex.fetch_ohlcv(symbol.replace('-', '/'), timeframe, limit=limit)
            return [{"time": c[0]/1000, "open": c[1], "high": c[2], "low": c[3], "close": c[4], "vol": c[5] if len(c) > 5 else 0} for c in ohlcv]
        except: return []

@app.get("/ml/available-models")
@app.get("/api/models")
def list_models():
    if not os.path.exists(MODEL_DIR): return {"status": "success", "models": []}
    models = [{"id": f.rsplit('.', 1)[0], "name": f.rsplit('.', 1)[0]} 
              for f in os.listdir(MODEL_DIR) if f.endswith(('.keras', '.joblib'))]
    return {"status": "success", "models": models}

@app.post("/api/bot/start")
async def start_bot(data: BotStartRequest, background_tasks: BackgroundTasks):
    user_id = data.userId.strip()
    
    # 🟢 1. GET CAPITAL FROM UI (Consolidated Greedy Search)
    # This checks for every possible naming convention to ensure the $300 is captured
    raw_cap = (
        data.config.get("capitalAllocation") or 
        data.config.get("capital_allocation") or 
        data.config.get("initialBalance")
    )
    ui_capital = float(raw_cap) if raw_cap else 200.0 

    # 2. Check if already active
    if user_id in ACTIVE_BOTS and ACTIVE_BOTS[user_id]["status"] == "running":
        return {"status": "running", "message": "Bot already active"}

    # 3. 🚀 PRE-FETCH CANDLES (Critical Chart Fix)
    initial_ohlcv = await fetch_live_candles_ccxt(data.config['symbol'], data.config.get('timeframe', '1h'), 150)
    processed_candles = []
    if initial_ohlcv:
        df_init = pd.DataFrame(initial_ohlcv)
        processed_candles = await process_data_packet(df_init, data.config.get('strategies', []))

    # 4. Load State & Merge Logic
    saved_state = DatabaseHandler.load_state(user_id)

    if saved_state:
        # 🟢 RESUME EXISTING SESSION
        ACTIVE_BOTS[user_id] = saved_state
        ACTIVE_BOTS[user_id]["status"] = "running"
        ACTIVE_BOTS[user_id]["config"] = data.config
        
        # 🚀 FORCE OVERWRITE: Balance Reset Fix
        # This officially unlocks the $200 lock by forcing the UI's $300 into the state
        ACTIVE_BOTS[user_id]["balance"] = ui_capital 
        
        # 🚀 RESET TIMER
        ACTIVE_BOTS[user_id]["startedAt"] = datetime.now(timezone.utc).isoformat()
        
        emit_log(user_id, f"♻️ SESSION RESET: Balance updated to ${ui_capital}")

    else:
        # 🟢 FRESH START
        ACTIVE_BOTS[user_id] = {
            "status": "running",
            "config": data.config,
            "balance": ui_capital, # Use the UI capital ($300)
            "positions": [],
            "trade_history": [],
            "equityCurve": [],
            "logs": [],
            "startedAt": datetime.now(timezone.utc).isoformat()
        }
        emit_log(user_id, f"🚀 Engine Started. Portfolio: ${ui_capital}")

    # 5. PRE-FILL CHART DATA (Prevents "Charts not showing")
    if not ACTIVE_BOTS[user_id].get("equityCurve"):
        ACTIVE_BOTS[user_id]["equityCurve"] = [{
            "time": datetime.now().isoformat(), 
            "balance": ui_capital, 
            "confidence": 50
        }]

    # 6. SAVE IMMEDIATELY
    # Persist the new $300 balance to disk right now
    DatabaseHandler.save_state(user_id, ACTIVE_BOTS[user_id])
    
    # 7. 🚀 EMIT IMMEDIATELY (Sends both Balance & Candles)
    emit_status(user_id, {
        "status": "running", 
        "currentBalance": ACTIVE_BOTS[user_id]["balance"],
        "candles": processed_candles,
        "startedAt": ACTIVE_BOTS[user_id]["startedAt"]
    })

    background_tasks.add_task(live_neural_heartbeat, user_id)
    return {"status": "running"}

@app.post("/api/bot/stop")
async def stop_bot(data: BotStopRequest):
    user_id = data.userId
    if user_id in ACTIVE_BOTS:
        # 1. Kill the loop flag
        ACTIVE_BOTS[user_id]["status"] = "stopped"
        
        # 2. Reset the bot data (The "Reset" part)
        ACTIVE_BOTS[user_id]["positions"] = []
        ACTIVE_BOTS[user_id]["trade_history"] = []
        ACTIVE_BOTS[user_id]["equityCurve"] = []
        ACTIVE_BOTS[user_id]["logs"] = []
        
        # 3. Tell Frontend to clear everything NOW
        emit_status(user_id, {
            "status": "stopped",
            "currentBalance": ACTIVE_BOTS[user_id]['balance'],
            "activePositions": [],
            "tradeMarkers": [],
            "equityCurve": [],
            "startedAt": None 
        })
        
        # 4. Wipe the Database state for this user
        DatabaseHandler.save_state(user_id, ACTIVE_BOTS[user_id])
        
        # 5. Remove from active memory to ensure total death
        del ACTIVE_BOTS[user_id]
        
        emit_log(user_id, "💀 SYSTEM PURGED: Engine stopped and session reset.")
        return {"status": "stopped", "message": "Bot killed and reset"}
    
    return {"status": "stopped"}

@app.post("/api/bot/close_position")
async def close_position(data: BotClosePositionRequest):
    if data.userId in ACTIVE_BOTS and ACTIVE_BOTS[data.userId]['positions']:
        pos = ACTIVE_BOTS[data.userId]['positions'].pop(0)
        emit_log(data.userId, f"⚠️ Manual Exit: {pos['type'].upper()} closed.")
        DatabaseHandler.save_state(data.userId, ACTIVE_BOTS[data.userId])
        return {"status": "closed"}
    raise HTTPException(status_code=400, detail="No active positions")

@app.post('/api/backtest/run')
async def run_backtest(request: BacktestRequest):
    """
    Atomic/Single Strategy Endpoint (Upgraded with Risk Fix)
    """
    try:
        config = request.dict()
        
        # 🛡️ RISK FIX FOR ATOMIC
        # Apply the same auto-correction logic here so single strategies behave safely
        p_dict = config.get('params', {})
        
        # Helper to fix percentages
        def fix_pct(val, default):
            if val is None: return default
            return val / 100 if val > 1.0 else val

        # Extract and Fix
        tp = fix_pct(p_dict.get('take_profit'), 0.06)
        sl = fix_pct(p_dict.get('stop_loss'), 0.03)
        ts = fix_pct(p_dict.get('trailing_stop'), 0.02)
        
        # Inject back into config for Backtester class to use
        config['params']['take_profit'] = tp
        config['params']['stop_loss'] = sl
        config['params']['trailing_stop'] = ts
        
        logger.info(f"🛡️ Atomic Risk Corrected: TP={tp}, SL={sl}, TS={ts}")

        # Run Bot
        bot = Backtester(config)
        result = bot.run()
        
        # 📊 CANDLE DATA FIX FOR ATOMIC
        # If the legacy Backtester didn't return candleData, we load it manually here
        if 'candleData' not in result or not result['candleData']:
            df = load_data_robust(config['symbol'], config['timeframe'])
            if df is not None:
                # Filter if needed (simplified)
                try:
                    df = df.loc[config['startDate']:config['endDate']]
                except: pass
                
                chart_df = df.reset_index()[['timestamp', 'open', 'high', 'low', 'close']].copy()
                chart_df['timestamp'] = chart_df['timestamp'].astype(str)
                result['candleData'] = chart_df.to_dict('records')
                result['initialBalance'] = config['initialBalance']
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Atomic API Error: {e}")
        return JSONResponse(content={"status": "failed", "error": str(e)}, status_code=500)

@app.post('/api/backtest/combo')
async def run_combo_backtest(req: ComboRequest):
    """
    Hybrid/Combo Engine (Fully Loaded)
    """
    try:
        logger.info(f"🔥 COMBO Request: {len(req.strategies)} strategies on {req.symbol}")
        
        # 1. Load Data
        df = load_data_robust(req.symbol, req.timeframe)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Data file not found for {req.symbol}")
            
        try:
            df = df.loc[req.startDate:req.endDate]
            if df.empty: raise Exception("No data in date range")
        except:
            pass

        # 2. Calculate Signals
        signals_list = []
        for strat in req.strategies:
            sig_series = calculate_strategy_signal(df, strat.code, strat.params)
            signals_list.append(sig_series)
        
        if not signals_list:
            return {"status": "error", "message": "No strategies executed"}

        # 3. Vote Logic
        sig_df = pd.concat(signals_list, axis=1).fillna(0)
        vote_sum = sig_df.sum(axis=1)
        
        final_signal = pd.Series(0, index=df.index)
        
        if req.combinationRule == "AND": 
            n = len(req.strategies)
            final_signal[vote_sum == n] = 1
            final_signal[vote_sum == -n] = -1
        else: 
            final_signal[vote_sum > 0] = 1
            final_signal[vote_sum < 0] = -1

        # 4. Event-Driven Simulation
        balance = req.initialBalance
        position = 0
        entry_price = 0.0
        best_price = 0.0
        equity_curve = []
        trades_log = []
        
        prices = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        times = df.index
        sigs = final_signal.values
        
        comm = 0.0006 
        
        # 🛡️ RISK PARSING (Robust)
        p_dict = req.params or {}
        raw_tp = p_dict.get('take_profit', req.take_profit)
        raw_sl = p_dict.get('stop_loss', req.stop_loss)
        raw_ts = p_dict.get('trailing_stop', req.trailing_stop)
        
        raw_tp = raw_tp if raw_tp is not None else 0.06
        raw_sl = raw_sl if raw_sl is not None else 0.03
        raw_ts = raw_ts if raw_ts is not None else 0.0
        
        tp_pct = raw_tp / 100 if raw_tp > 1.0 else raw_tp
        sl_pct = raw_sl / 100 if raw_sl > 1.0 else raw_sl
        ts_pct = raw_ts / 100 if raw_ts > 1.0 else raw_ts

        logger.info(f"🛡️ Combo Risk Loaded: TP={tp_pct}, SL={sl_pct}, TS={ts_pct}")
        
        for i in range(1, len(prices)):
            price = prices[i]
            high = highs[i]
            low = lows[i]
            date = str(times[i])
            signal = sigs[i-1] 
            
            # --- CHECK EXITS ---
            if position == 1: # LONG
                if high > best_price: best_price = high
                
                stop_price = entry_price * (1 - sl_pct)
                take_price = entry_price * (1 + tp_pct)
                trail_price = best_price * (1 - ts_pct) if ts_pct > 0 else 0
                
                exit_reason = None
                exit_p = price
                
                if low <= stop_price: 
                    exit_reason = "SL"
                    exit_p = stop_price
                elif ts_pct > 0 and low <= trail_price:
                    exit_reason = "Trail"
                    exit_p = trail_price
                elif high >= take_price:
                    exit_reason = "TP"
                    exit_p = take_price
                
                if exit_reason:
                    pnl_pct = (exit_p - entry_price) / entry_price
                    balance *= (1 + pnl_pct - comm)
                    trades_log.append({"type": "close_long", "price": exit_p, "time": date, "balance": balance, "reason": exit_reason})
                    position = 0

            elif position == -1: # SHORT
                if low < best_price: best_price = low
                
                stop_price = entry_price * (1 + sl_pct)
                take_price = entry_price * (1 - tp_pct)
                trail_price = best_price * (1 + ts_pct) if ts_pct > 0 else 9999999
                
                exit_reason = None
                exit_p = price
                
                if high >= stop_price:
                    exit_reason = "SL"
                    exit_p = stop_price
                elif ts_pct > 0 and high >= trail_price:
                    exit_reason = "Trail"
                    exit_p = trail_price
                elif low <= take_price:
                    exit_reason = "TP"
                    exit_p = take_price
                    
                if exit_reason:
                    pnl_pct = (entry_price - exit_p) / entry_price
                    balance *= (1 + pnl_pct - comm)
                    trades_log.append({"type": "close_short", "price": exit_p, "time": date, "balance": balance, "reason": exit_reason})
                    position = 0
            
            # --- CHECK ENTRIES ---
            if position == 0:
                if signal == 1:
                    position = 1
                    entry_price = price
                    best_price = price
                    balance *= (1 - comm)
                    trades_log.append({"type": "buy", "price": price, "time": date})
                elif signal == -1:
                    position = -1
                    entry_price = price
                    best_price = price
                    balance *= (1 - comm)
                    trades_log.append({"type": "sell", "price": price, "time": date})
            
            elif position == 1 and signal == -1:
                pnl_pct = (price - entry_price) / entry_price
                balance *= (1 + pnl_pct - comm)
                trades_log.append({"type": "flip_to_short", "price": price, "time": date, "balance": balance, "reason": "Signal Flip"})
                position = -1
                entry_price = price
                best_price = price
                balance *= (1 - comm)
                
            elif position == -1 and signal == 1:
                pnl_pct = (entry_price - price) / entry_price
                balance *= (1 + pnl_pct - comm)
                trades_log.append({"type": "flip_to_long", "price": price, "time": date, "balance": balance, "reason": "Signal Flip"})
                position = 1
                entry_price = price
                best_price = price
                balance *= (1 - comm)

            equity_curve.append({"time": date, "balance": balance})

        roi = ((balance - req.initialBalance) / req.initialBalance) * 100
        
        # 📊 CANDLE DATA FIX FOR COMBO
        chart_df = df.reset_index()[['timestamp', 'open', 'high', 'low', 'close']].copy()
        chart_df['timestamp'] = chart_df['timestamp'].astype(str)
        candle_data = chart_df.to_dict('records')

        print(f"DEBUG: Returning {len(candle_data)} candles to Frontend")

        return {
            "status": "completed",
            "metrics": {
                "final_balance": balance,
                "roi": roi,
                "total_trades": len(trades_log)
            },
            "equityCurve": equity_curve,
            "trades": trades_log,
            "candleData": candle_data,
            "initialBalance": req.initialBalance
        }

    except Exception as e:
        logger.error(f"Combo API Error: {e}")
        return JSONResponse(content={"status": "failed", "error": str(e)}, status_code=500)


@app.get("/api/bot/status")
async def get_status(userId: str):
    bot = ACTIVE_BOTS.get(userId.strip())
    if not bot:
        bot = DatabaseHandler.load_state(userId.strip())
    
    if bot:
        return {
            "status": bot["status"], 
            "balance": bot["balance"],
            "equityCurve": bot.get("equityCurve", []),
            "logs": bot.get("logs", []),
            "positions": bot.get("positions", []),
            # 🟢 NEW: Return start time
            "startedAt": bot.get("startedAt"),
            "config": bot.get("config"),
            "candles": bot.get("candles", [])
        }
    return {"status": "inactive", "balance": 0}

@app.post("/api/bot/reset")
async def reset_bot(data: BotStopRequest): # Uses same model as stop
    if data.userId in ACTIVE_BOTS:
        ACTIVE_BOTS[data.userId]["status"] = "stopped"
        ACTIVE_BOTS[data.userId]["positions"] = []
        ACTIVE_BOTS[data.userId]["trade_history"] = []
        ACTIVE_BOTS[data.userId]["equityCurve"] = []
        ACTIVE_BOTS[data.userId]["logs"] = []
        DatabaseHandler.save_state(data.userId, ACTIVE_BOTS[data.userId])
        return {"status": "reset"}
    return {"status": "not_found"}



if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
(venv) root@intelligent-mendel:~/Project/ML# 
