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
import time
import pandas_ta as ta
import ccxt.async_support as ccxt
import ccxt.pro as ccxtpro
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Union
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
import aiohttp
from app.verify.engineer_and_train import apply_mega_features

# 🟢 SOCKET HELPERS (Must be async/await)
from app.services.socket_emitter import emit_log, emit_status
from app.config2 import MODEL_DIR
from app.predictors.stacking_predictor import StackingPredictor

MODEL_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEO-Engine")

GLOBAL_SESSION: Optional[aiohttp.ClientSession] = None
ACTIVE_BOTS = {}
TASK_REGISTRY = {}

# ==========================================
# 🗄️ 0. DATABASE HANDLER
# ==========================================
class DatabaseHandler:
    DB_FILE = "bot_state.db"

    @classmethod
    def init_db(cls):
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
        conn = sqlite3.connect(cls.DB_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        try:
            c.execute("SELECT * FROM bot_sessions WHERE user_id = ?", (user_id,))
        except sqlite3.OperationalError:
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
                    "status": "stopped"
                }
            except Exception as e:
                logger.error(f"DB Load Error: {e}")
                return None
        return None

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
    code: str  
    mlModel: str = "stacking"
    # 🎯 FIX: Make this Optional so the API doesn't crash if it's missing
    # We will map 'mlThresholdLong' to this inside the endpoint logic
    ml_confidence_threshold: Optional[float] = 0.8 
    
    # 🎯 ADD: Support for the actual keys being sent by the frontend
    mlThresholdLong: Optional[float] = 0.8
    mlThresholdShort: Optional[float] = 0.8
    
    trend_strategy: Optional[str] = "atr_breakout"
    range_strategy: Optional[str] = "bollinger_reversal"
    trade_direction: Optional[str] = "BOTH"
    params: Optional[Dict[str, Any]] = {}
    
BacktestRequest.model_rebuild()


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
    mlModel: Optional[str] = "stacking"
    # 🎯 FIX: Use defaults here as well to prevent "Field Required" errors
    # if the frontend user leaves a slider untouched.
    mlThresholdLong: float = 0.8
    mlThresholdShort: float = 0.8
    mlMode: Optional[str] = None
    advanced_filters: Optional[Dict] = {}
    params: Optional[Dict[str, Any]] = {}

ComboRequest.model_rebuild()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_SESSION
    GLOBAL_SESSION = aiohttp.ClientSession()
    # 🗑️ Removed global_exchange initialization here
    yield

    await GLOBAL_SESSION.close()
    # 🗑️ Removed global_exchange.close() here
    
    # 🛡️ This keeps your database safe on server restart
    for user_id, bot in ACTIVE_BOTS.items():
        bot["status"] = "stopped"
        DatabaseHandler.save_state(user_id, bot)
        
app = FastAPI(title="NEO-V25.14 Sovereign Engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    print(f"❌ DATA ERROR: {exc.errors()}")
    print(f"❌ RECEIVED BODY: {exc.body}")
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "body": exc.body})

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


async def ensure_full_data(symbol, timeframe, start_str, end_str,*args, **kwargs):
    """
    Checks local DB for data gaps and fetches from Coinbase (Async).
    """
    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_str).timestamp() * 1000)

    df = load_data_robust(symbol, timeframe)
    if df is None: df = pd.DataFrame()

    needs_fetch = False
    if df.empty:
        needs_fetch = True
        current_since = start_ts
    else:
        local_start = int(df.index.min().timestamp() * 1000)
        if local_start > start_ts + 3600000:
            needs_fetch = True
            current_since = start_ts
        else:
            current_since = int(df.index.max().timestamp() * 1000)

    if needs_fetch or current_since < end_ts:
        print(f"📡 US-DATA GAP: Fetching {symbol} from Coinbase...")
        
        # 🟢 Use Async Coinbase
        exchange = ccxt.coinbase({'enableRateLimit': True}) 
        all_new_candles = []
        fetch_symbol = symbol.replace("-", "/") 

        try:
            while current_since < end_ts:
                # 🟢 MUST AWAIT THIS CALL
                new_batch = await exchange.fetch_ohlcv(fetch_symbol, timeframe, since=current_since, limit=300)
                if not new_batch: break
                
                all_new_candles.extend(new_batch)
                current_since = new_batch[-1][0] + 1 
                # No need for time.sleep() with enableRateLimit in async
            
            # 🔴 CRITICAL: Close the async session
            await exchange.close()
        except Exception as e:
            print(f"⚠️ Coinbase Fetch Error: {e}")
            await exchange.close()

        if all_new_candles:
            new_df = pd.DataFrame(all_new_candles, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            new_df['timestamp'] = pd.to_datetime(new_df['ts'], unit='ms', utc=True)
            new_df.set_index('timestamp', inplace=True)
            new_df.drop(columns=['ts'], inplace=True)
            
            df = pd.concat([df, new_df]).sort_index()
            df = df[~df.index.duplicated(keep='first')]
            
            try:
                save_to_local_db(df, symbol, timeframe)
            except: pass

    target_start = pd.to_datetime(start_str, utc=True)
    target_end = pd.to_datetime(end_str, utc=True)
    return df



# ==========================================
# 🧠 1. NEURAL PREDICTOR
# ==========================================
class DiagnosticLayer:
    @staticmethod
    def render_progress(current, target, reverse=False):
        try:
            pct = (target / current) if not reverse else (current / target)
            pct = min(1.0, max(0.0, pct))
            filled = int(pct * 10)
            bar = "┃" + "█" * filled + "░" * (10 - filled) + "┃"
            return f"{bar} {int(pct * 100)}%"
        except: return "[----------] 0%"

    @staticmethod
    def get_pending_conditions(df, config, conf, ui_limit):
        strategies = config.get('strategies', [])
        try:
            current_price = float(df['close'].iloc[-1])
            ema200_val = ta.ema(df['close'], length=200).iloc[-1]
            is_uptrend = current_price > ema200_val
            if conf < ui_limit:
                return f"🛑 AI VETO: Confidence {DiagnosticLayer.render_progress(conf, ui_limit, True)}"
            
            pending = []
            for strat in strategies:
                code = strat.get('code')
                p = strat.get('params', {})
                if code == "rsi_threshold":
                    val = ta.rsi(df['close']).iloc[-1]
                    target = 30 if current_price < ta.ema(df['close'], 200).iloc[-1] else 70
                    pending.append(f"RSI: {DiagnosticLayer.render_progress(val, target)}")

                elif code == "sma_crossover":
                    fast = ta.sma(df['close'], 50).iloc[-1]
                    slow = ta.sma(df['close'], 200).iloc[-1]
                    pending.append(f"SMA Cross: {DiagnosticLayer.render_progress(fast, slow)}")

                elif code == "supertrend":
                    st = ta.supertrend(df['high'], df['low'], df['close']).iloc[-1, 0]
                    pending.append(f"ST Dist: {DiagnosticLayer.render_progress(current_price, st)}")

                elif code == "macd_crossover":
                    macd = ta.macd(df['close'])
                    pending.append(f"MACD Gap: {DiagnosticLayer.render_progress(macd.iloc[-1, 0], macd.iloc[-1, 2])}")

                elif code == "atr_breakout":
                    atr = ta.atr(df['high'], df['low'], df['close']).iloc[-1]
                    ema20 = ta.ema(df['close'], 20).iloc[-1]
                    pending.append(f"ATR Break: {DiagnosticLayer.render_progress(current_price, ema20 + atr)}")

                elif code == "bb_fade":
                    bb = ta.bbands(df['close']).iloc[-1]
                    target = bb[0] if current_price < bb[1] else bb[2]
                    pending.append(f"BB Wall: {DiagnosticLayer.render_progress(current_price, target)}")

                elif code == "stoch":
                    k = ta.stoch(df['high'], df['low'], df['close']).iloc[-1, 0]
                    pending.append(f"Stoch: {DiagnosticLayer.render_progress(k, 20)}")

                elif code == "ema_cloud":
                    fast = ta.ema(df['close'], 9).iloc[-1]
                    slow = ta.ema(df['close'], 21).iloc[-1]
                    pending.append(f"Cloud: {DiagnosticLayer.render_progress(fast, slow)}")

                elif code == "pa_breakout":
                    high_20 = df['high'].rolling(20).max().iloc[-1]
                    pending.append(f"PA High: {DiagnosticLayer.render_progress(current_price, high_20)}")

                elif code == "vol_profile":
                    vol_ma = ta.sma(df['volume'], 20).iloc[-1]
                    pending.append(f"Vol Surge: {DiagnosticLayer.render_progress(df['volume'].iloc[-1], vol_ma * 1.5)}")
            
            rule = config.get('comboConfig', {}).get('combinationRule', 'OR')
            if not pending: return f"🔍 TARGETS ({rule}): Scanning Setup..."
            return f"🔍 TARGETS ({rule}): " + " | ".join(pending[:2])
        except Exception: return "🔍 Scanning Market Conditions..."

class NeuralPredictor:
    @staticmethod
    def get_prediction(model_id: str, df: pd.DataFrame, symbol: str = "BTC-USD") -> float:
        """
        Swaps the old placeholder logic for the Full Council Stacking Judge.
        """
        try:
            # 1. Initialize the Stacking Predictor (The Council)
            # Standardized for our 25-feature logic
            council = StackingPredictor(symbol=symbol, timeframe="1h")
            
            # 2. Get the Final Probability from the Stacking Judge
            # This internal call will also PRINT the debate to your terminal console
            prediction = council.predict_direction(df)
            
            return float(prediction)
        except Exception as e:
            logger.error(f"🧠 Council Predictor Error: {e}")
            return 0.5 # Neutral fallback


# ==========================================
# Packet Data
# ==========================================
async def process_data_packet(df: pd.DataFrame, strategies: list) -> list:
    # 🎯 V25 UPGRADE: One call to rule them all
    df, feats = apply_mega_features(df)
    
    
    keys = ['bb_lower', 'bb_upper', 'ema_fast', 'ema_slow', 'sma_fast', 'sma_slow', 
            'supertrend', 'rsi', 'macd', 'macd_signal', 'stoch_k', 'stoch_d', 
            'atr_upper', 'atr_lower', 'pa_high', 'pa_low', 'vol_ma']

    candles_to_send = []
    for _, row in df.tail(100).iterrows():
        ts = int(row['time']) if 'time' in row else int(row.name.timestamp())
        c_obj = {"time": ts, "open": row['open'], "high": row['high'], "low": row['low'], "close": row['close']}
        for k in keys:
            if k in row and not pd.isna(row[k]): c_obj[k] = round(float(row[k]), 2)
        candles_to_send.append(c_obj)
    return candles_to_send

# ==========================================
# 🧠 2. SHARED STRATEGY BRAIN
# ==========================================
class StrategyBrain:
   @staticmethod
   def calculate_signals(df: pd.DataFrame, config: Dict[str, Any], l_thresh: float, s_thresh: float, symbol="BTC-USD"):
       active_thoughts, votes = [], 0
        # 🟢 ADD THIS: Create a dictionary to store raw values for the logs
       signals_map = {} 
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
               # 1. RSI (Distance to 30/70)
               if code == "rsi_threshold":
                   rsi = ta.rsi(df['close'], length=int(p.get('rsi_length', 14))).iloc[-1]
                   dist = min(abs(rsi - 30), abs(rsi - 70))
                   signals_map[code] = max(0.1, min(1.0, 1.0 - (dist / 40)))
                   if rsi < p.get('oversold', 30): votes += 1; active_thoughts.append("RSI Low")
                   elif rsi > p.get('overbought', 70): votes -= 1; active_thoughts.append("RSI High")
                # 2. SMA CROSSOVER (Proximity of Fast to Slow)
               elif code == "sma_crossover":
                   f = ta.sma(df['close'], length=int(p.get('fast_sma', 50))).iloc[-1]
                   s = ta.sma(df['close'], length=int(p.get('slow_sma', 200))).iloc[-1]
                   gap = abs(f - s) / s
                   signals_map[code] = 1.0 if f > s else max(0.1, min(0.95, 1.0 - (gap * 50)))
                   if f > s: votes += 1
                # 3. MACD CROSSOVER (Histogram Intensity)
               elif code == "macd_crossover":
                   macd = ta.macd(df['close'], fast=int(p.get('fast', 12))).iloc[-1]
                   hist = macd[1] # Histogram
                   norm_hist = abs(hist) / (current_price * 0.0005)
                   signals_map[code] = max(0.1, min(1.0, norm_hist))
                   votes += (1 if macd[0] > macd[2] else -1)
                # 4. SUPERTREND (Price proximity to ST Line)
               elif code == "supertrend":
                   st_data = ta.supertrend(df['high'], df['low'], df['close']).iloc[-1]
                   st_line = st_data[0]
                   dist = abs(current_price - st_line) / current_price
                   signals_map[code] = 1.0 if st_data[1] == 1 else max(0.1, min(0.95, 1.0 - (dist * 20)))
                   votes += (1 if st_data[1] == 1 else -1)

                # 5. BOLLINGER FADE (Distance to Wall)
               elif code == "bb_fade":
                   signals_map[code] = max(0.1, min(1.0, pr / 100.0))
                   if current_price < lower: votes += 1
                   elif current_price > upper: votes -= 1
                # 6. ATR BREAKOUT (Distance to EMA+ATR Channel)
               elif code == "atr_breakout":
                   atr = ta.atr(df['high'], df['low'], df['close']).iloc[-1]
                   target = ema20 + (atr * float(p.get('multiplier', 1.5)))
                   signals_map[code] = max(0.1, min(1.0, current_price / target))
                   if current_price > target: votes += 1
                # 7. PRICE ACTION BREAKOUT (Distance to 20-candle High)
               elif code == "pa_breakout":
                   lb = int(p.get('lookback', 20))
                   high_lb = df['high'].tail(lb).max()
                   signals_map[code] = max(0.1, min(1.0, current_price / high_lb))
                   if current_price >= high_lb: votes += 1
                # 8. VOL PROFILE (Volume Surge Ratio)
               elif code == "vol_profile":
                   v_ma = ta.sma(df['volume'], length=int(p.get('vol_ma', 20))).iloc[-1]
                   ratio = df['volume'].iloc[-1] / (v_ma * float(p.get('threshold', 1.5)))
                   signals_map[code] = max(0.1, min(1.0, ratio))
                   if ratio >= 1.0: votes += (1 if current_price > mid else -1)
                # 9. STOCHASTIC (Proximity to 20/80 threshold)
               elif code == "stoch":
                   stoch_df = ta.stoch(df['high'], df['low'], df['close']).iloc[-1]
                   k = stoch_df[0]
                   dist = min(abs(k - 20), abs(k - 80))
                   signals_map[code] = max(0.1, min(1.0, 1.0 - (dist / 40)))
                   if k < 20: votes += 1
                   elif k > 80: votes -= -1
                # 10. EMA CLOUD (Distance between Fast and Slow EMA)
               elif code == "ema_cloud":
                   f_ema = ta.ema(df['close'], length=int(p.get('fast_ema', 9))).iloc[-1]
                   s_ema = ta.ema(df['close'], length=int(p.get('slow_ema', 21))).iloc[-1]
                   gap = abs(f_ema - s_ema) / s_ema
                   signals_map[code] = 1.0 if f_ema > s_ema else max(0.1, min(0.95, 1.0 - (gap * 100)))
                   if f_ema > s_ema: votes += 1
               else:
                   signals_map[code] = 0.5
           except Exception: signals_map[code] = 0.0
        
       # 🚀 3. DYNAMIC DUAL-GATE LOGIC (ML Filtering)
       is_short = current_price < ema200
       ui_limit = float(config.get('mlThresholdShort', 0.90)) if is_short else float(config.get('mlThresholdLong', 0.80))
       conf = NeuralPredictor.get_prediction(config.get('mlModel', 'stacking'), df, symbol=symbol)
       gate_passed = conf >= ui_limit

       logic_desc = f"📊 LOGIC: {'SHORT' if is_short else 'LONG'} GATE {'PASSED' if gate_passed else 'VETOED'} ({int(conf*100)}% vs {int(ui_limit*100)}% UI Limit) {'🟢' if gate_passed else '🔴'}"
       signal_names = " + ".join(active_thoughts) if active_thoughts else "Scanning Setup"
       gap = int(abs(current_price - ema50))
       intent_desc = f"🎯 INTENT: STALKING {'SHORT' if is_short else 'LONG'} ({signal_names} | Gap: ${gap}) {'🔴' if is_short else '🟢'}"

       trend_dist = current_price - ema200
       trend_text = f"📡 TREND: {'UP' if trend_dist > 0 else 'DOWN'} (Price is ${int(abs(trend_dist))} {'above' if trend_dist > 0 else 'below'} 200EMA)"
       spread = ema20 - ema50
       bias_str = "BULLISH EXPANSION" if spread > 0 else "BEARISH CONTRACTION"
       bias_text = f"⚖️ BIAS: {bias_str} (Fast EMA is ${int(abs(spread))} {'above' if spread > 0 else 'below'} Slow EMA)"
       mindset_str = "⚠️ OVEREXTENDED" if pr >= 80 else "🎯 ACCUMULATION" if pr <= 20 else "⚖️ EQUILIBRIUM"
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
        # 🎯 6. FINAL SIGNAL CALCULATION (Directional Unlock Upgrade)
       rule = config.get('comboConfig', {}).get('combinationRule', 'OR')
       final_sig = 0
       
       if gate_passed:
           # Check for LONG: Votes must be positive
           if rule == "AND":
               if votes >= len(strategies): final_sig = 1
           else: # "OR" logic
               if votes > 0: final_sig = 1
           
            # Check for SHORT: Votes must be negative
            # Note: We check this separately so a Long can flip to Short and vice versa
           if rule == "AND":
               if votes <= -len(strategies): final_sig = -1
           elif votes < 0: # "OR" logic
               final_sig = -1

        # 🚀 LOGIC OVERRIDE: 
        # If the ML says we can trade (gate_passed), we take the signal 
        # provided by the strategies, ignoring the trend bias.
       return final_sig, active_thoughts, numeric_details, conf, signals_map

# ==========================================
# 🚀 3. THE HEARTBEAT (Dynamic Calculation Loop)
# ==========================================

async def live_neural_heartbeat(user_id: str):
    last_log = 0
    last_ui_update = 0
    
    if user_id in ACTIVE_BOTS:
        if "equityCurve" not in ACTIVE_BOTS[user_id]: 
            ACTIVE_BOTS[user_id]["equityCurve"] = [{"time": datetime.now().isoformat(), "balance": ACTIVE_BOTS[user_id]["balance"], "confidence": 50}]
        if "logs" not in ACTIVE_BOTS[user_id]: ACTIVE_BOTS[user_id]["logs"] = []

    try:
        while user_id in ACTIVE_BOTS and ACTIVE_BOTS[user_id]["status"] == "running":
            bot = ACTIVE_BOTS[user_id]
            config = bot.get('config', {})
            params = config.get('params', {})
            strategies = config.get('strategies', [])
            symbol = config['symbol'].replace('-', '/')
            ticker_symbol = config['symbol'] 
            
            # 🔐 STATE ROUTING
            api_keys = bot.get('api_keys') or config.get('api_keys', {}) 
            trading_mode = config.get('trading_mode', 'paper').lower()
            use_margin = config.get('enable_shorting', False)
            leverage_val = float(config.get('leverage', 1.0))
            
            target_exchange = "kraken" if use_margin else "coinbase"
            
            # Note: For Kraken, the UI should send 'krakenKey'. For Coinbase, 'apiKey'.
            has_valid_keys = bool(api_keys.get('krakenKey') if use_margin else api_keys.get('apiKey'))
            is_live_trading = (trading_mode == 'live') and has_valid_keys

            if trading_mode == 'live' and not is_live_trading:
                logger.warning(f"User {user_id} selected LIVE but lacks API keys. Falling back to PAPER.")
                await emit_log(user_id, "⚠️ LIVE MODE FAILED: Missing API Keys. Forcing PAPER MODE.")
                config['trading_mode'] = 'paper'
                bot['config'] = config 
                DatabaseHandler.save_state(user_id, bot)

            try:
                # 🟢 2. DATA FETCHING (Dynamic Exchange Pulse)
                ohlcv_raw = await fetch_live_candles_ccxt(config['symbol'], config.get('timeframe', '1h'), 350, exchange_id=target_exchange)
                
                if not ohlcv_raw:
                    await emit_log(user_id, f"⚠️ {target_exchange.upper()} Feed Unstable - Retrying...")
                    await asyncio.sleep(5); continue
                
                # 🚀 THE FIX: Dynamically fetch the ticker from the correct exchange
                exchange_class = getattr(ccxt, target_exchange)
                async with exchange_class({'enableRateLimit': True}) as ex:
                    ticker = await ex.fetch_ticker(ticker_symbol.replace('-', '/'))
                    current_price = float(ticker['last'])
                
                # Zero-Drift Patch using the live price
                ohlcv_raw[-1]['close'] = current_price
                ohlcv_raw[-1]['time'] = int(datetime.now(timezone.utc).timestamp())
                
                # 🟢 3. THE ANTI-REPAINTING LOCK & THREAD FIX
                df_raw = pd.DataFrame(ohlcv_raw)
                df_closed_history = df_raw.iloc[:-1].copy()
                
                # 🚀 CRITICAL FIX: Offload heavy math to background thread
                df_ai, _ = await asyncio.to_thread(apply_mega_features, df_closed_history)

                # 🟢 4. COUNCIL SIGNAL PROCESSING
                sig, thoughts, nums, score, signals_map = StrategyBrain.calculate_signals(
                    df_ai, config, 0.5, 0.5, symbol=ticker_symbol
                )

                sentiment = "STRONG BUY" if score > 0.85 else "BUY" if score > 0.70 else "NEUTRAL"
                if score < 0.20: sentiment = "STRONG SELL"
                elif score < 0.35: sentiment = "SELL"

                # 🟢 5. HUD CONSTRUCTION
                upnl = sum([(current_price - p['entry']) * p['size'] if p['type'] == 'long' else (p['entry'] - current_price) * p['size'] for p in bot['positions']])
                current_equity = bot['balance'] + upnl

                active_summary = ""
                buffer_bar = "[----------]" 
                if bot['positions']:
                    p = bot['positions'][-1]
                    dist_to_stop = abs(current_price - p['tsl'])
                    stop_pct = round((dist_to_stop / current_price) * 100, 2)
                    buffer_bar = DiagnosticLayer.render_progress(stop_pct, 2.0, reverse=True)
                    active_summary = f"⚡ ACTIVE: {len(bot['positions'])} POS (${upnl:,.2f}) | TSL {buffer_bar} {stop_pct}% | "

                ui_limit = float(config.get('mlThresholdLong', 0.80)) if current_price > ta.ema(df_ai['close'], 200).iloc[-1] else float(config.get('mlThresholdShort', 0.90))
                waiting_msg = DiagnosticLayer.get_pending_conditions(df_ai, config, score, ui_limit)

                # ==========================================
                # 🛡️ NEURAL VETO TRACKER (Inserted Here)
                # ==========================================
                if "vetoed_signals" not in bot:
                    bot["vetoed_signals"] = []

                # Reconstruct what the atomic strategies WANTED to do
                rule = config.get('comboConfig', {}).get('combinationRule', 'OR')
                raw_sig = 0
                votes = sum([1 if signals_map.get(s['code'], 0) > 0.5 else -1 for s in strategies])
                
                if rule == "AND":
                    if votes >= len(strategies): raw_sig = 1
                    elif votes <= -len(strategies): raw_sig = -1
                else:
                    if votes > 0: raw_sig = 1
                    elif votes < 0: raw_sig = -1

                # If they wanted to trade, but the AI stopped them, log it!
                if raw_sig != 0 and score < ui_limit:
                    # Prevent spamming the same veto every 0.5 seconds (5-minute cooldown)
                    if not bot["vetoed_signals"] or (datetime.now(timezone.utc) - datetime.fromisoformat(bot["vetoed_signals"][-1]["time"])).total_seconds() > 300:
                        bot["vetoed_signals"].append({
                            "time": datetime.now(timezone.utc).isoformat(),
                            "signal": "Long" if raw_sig == 1 else "Short",
                            "conf_score": round(score, 4),
                            "limit": round(ui_limit, 4),
                            "price": current_price
                        })
                        # Keep memory clean (last 100 vetoes)
                        if len(bot["vetoed_signals"]) > 100: bot["vetoed_signals"].pop(0)
                
                if len(bot['positions']) < int(config.get('maxPyramiding', 1)):
                    hunting_summary = f"🏹 STALKING LEG {len(bot['positions'])+1}: {int(score*100)}% ({sentiment}) | {waiting_msg}"
                else:
                    hunting_summary = "✅ PYRAMID FULL: Managing Exits"

                targets_str = " | ".join([f"{s['code'].upper()}: {DiagnosticLayer.render_progress(int(signals_map.get(s['code'], 0) * 100), 100)}" for s in strategies])
                combined_status = f"{active_summary}{hunting_summary} | 🔍 {targets_str}"

                # 🟢 6. EMIT LOGS & UI UPDATES
                now_ts = datetime.now().timestamp()
                if (now_ts - last_log >= 15):
                    await emit_log(user_id, combined_status)
                    last_log = now_ts

                if (now_ts - last_ui_update >= 10):
                    exposure_pct = round((sum([p['entry'] * p['size'] for p in bot['positions']]) / bot['balance']) * 100, 1) if bot['balance'] > 0 else 0
                    await emit_status(user_id, {
                        "status": "running", "currentBalance": round(current_equity, 2), "exposure": exposure_pct,
                        "activePositions": bot['positions'], "unrealizedPnl": round(upnl, 2),
                        "tradeMarkers": bot['trade_history'], "candles": await process_data_packet(df_raw, strategies), 
                        "currentConfidence": int(score * 100), "signalsMap": signals_map
                    })
                    bot["equityCurve"].append({"time": datetime.now().isoformat(), "balance": round(current_equity, 2), "confidence": int(score * 100)})
                    if len(bot["equityCurve"]) > 300: bot["equityCurve"].pop(0)
                    last_ui_update = now_ts

                # 🟢 7. TRADE EXECUTION LOGIC (Live Router)
                max_p = min(5, int(config.get('maxPyramiding', 5)))
                current_time = datetime.now(timezone.utc)
                
                time_gate_passed = not bot.get('last_trade_time') or (current_time - datetime.fromisoformat(bot['last_trade_time'])).total_seconds() >= 3600
                last_pos = bot['positions'][-1] if bot['positions'] else None
                climb_satisfied = (score >= last_pos.get('entry_conf', 0) + 0.10) if last_pos else True

                # ENTRY
                if len(bot['positions']) < max_p and time_gate_passed:
                    if (sig == 1 and climb_satisfied) or (sig == -1 and climb_satisfied):
                        trade_type = "long" if sig == 1 else "short"
                        size_in_fiat = bot['balance'] * (float(config.get('riskPercentage', 10.0)) / 100 / max_p)
                        size_in_crypto = size_in_fiat / current_price
                        
                        # 🚀 CRITICAL FIX: Prevent Scientific Notation (1e-5) rejection
                        safe_size = float(f"{size_in_crypto:.6f}") 
                        actual_entry_price = current_price
                        
                        # 🔴 REAL API EXECUTION
                        if is_live_trading:
                            try:
                                exchange_class = getattr(ccxt, target_exchange)
                                async with exchange_class({
                                    'apiKey': api_keys.get('krakenKey', api_keys.get('apiKey')),
                                    'secret': api_keys.get('krakenSecret', api_keys.get('secret')),
                                    'enableRateLimit': True
                                }) as user_exchange:
                                    
                                    side = "buy" if sig == 1 else "sell"
                                    order_params = {'leverage': leverage_val} if use_margin else {}
                                    
                                    await emit_log(user_id, f"🔗 ROUTING {side.upper()} TO {target_exchange.upper()}...")
                                    order = await user_exchange.create_market_order(symbol, side, safe_size, params=order_params)
                                    
                                    actual_entry_price = order.get('average') or order.get('price') or current_price
                                    fee_rate = 0.0026 if use_margin else 0.006
                                    bot['balance'] -= (safe_size * actual_entry_price) * fee_rate
                                    
                            except Exception as ex_err:
                                await emit_log(user_id, f"❌ {target_exchange.upper()} ORDER FAILED: {str(ex_err)}")
                                await asyncio.sleep(5)
                                continue 
                        
                        bot['positions'].append({
                            "symbol": symbol, "type": trade_type, "entry": actual_entry_price, "size": safe_size,
                            "time": current_time.isoformat(), "entry_conf": score,
                            "tp": actual_entry_price * (1 + float(params.get('take_profit', 0.1))) if sig == 1 else actual_entry_price * (1 - float(params.get('take_profit', 0.1))),
                            "sl": actual_entry_price * (1 - float(params.get('stop_loss', 0.05))) if sig == 1 else actual_entry_price * (1 + float(params.get('stop_loss', 0.05))),
                            "tsl": actual_entry_price * (1 - float(params.get('stop_loss', 0.05))) if sig == 1 else actual_entry_price * (1 + float(params.get('stop_loss', 0.05)))
                        })
                        bot['last_trade_time'] = current_time.isoformat()
                        await emit_log(user_id, f"🚀 ENTERED {trade_type.upper()} LEG {len(bot['positions'])} @ ${actual_entry_price:,.2f}")
                        DatabaseHandler.save_state(user_id, bot)

                # 🟢 8. EXIT MONITORING
                tsl_pct = float(params.get('trailing_stop', 0.01))
                for pos in bot['positions'][:]:
                    closed = False
                    if pos['type'] == 'long':
                        if current_price * (1 - tsl_pct) > pos['tsl']: pos['tsl'] = current_price * (1 - tsl_pct)
                        if current_price >= pos['tp'] or current_price <= pos['tsl']: closed = True
                    else:
                        if current_price * (1 + tsl_pct) < pos['tsl']: pos['tsl'] = current_price * (1 + tsl_pct)
                        if current_price <= pos['tp'] or current_price >= pos['tsl']: closed = True
                    
                    if closed:
                        actual_close_price = current_price
                        
                        # 🔴 REAL API EXECUTION
                        if is_live_trading:
                            try:
                                exchange_class = getattr(ccxt, target_exchange)
                                async with exchange_class({
                                    'apiKey': api_keys.get('krakenKey', api_keys.get('apiKey')),
                                    'secret': api_keys.get('krakenSecret', api_keys.get('secret')),
                                    'enableRateLimit': True
                                }) as user_exchange:
                                    
                                    close_side = "sell" if pos['type'] == 'long' else "buy"
                                    order_params = {'leverage': leverage_val} if use_margin else {}
                                    
                                    await emit_log(user_id, f"🔗 ROUTING {close_side.upper()} EXIT TO {target_exchange.upper()}...")
                                    order = await user_exchange.create_market_order(symbol, close_side, pos['size'], params=order_params)
                                    actual_close_price = order.get('average') or order.get('price') or current_price
                                    
                            except Exception as ex_err:
                                await emit_log(user_id, f"❌ {target_exchange.upper()} EXIT FAILED: {str(ex_err)}")
                                continue 
                        
                        gross_pnl = (actual_close_price - pos['entry']) * pos['size'] if pos['type'] == 'long' else (pos['entry'] - actual_close_price) * pos['size']
                        fee_rate = 0.0026 if use_margin else 0.006
                        net_pnl = gross_pnl - ((pos['size'] * actual_close_price) * fee_rate)

                        bot['balance'] += net_pnl
                        bot['positions'].remove(pos)
                        bot['trade_history'].append({"type": "exit", "side": pos['type'], "price": actual_close_price, "pnl": round(net_pnl, 2), "time": datetime.now(timezone.utc).isoformat()})
                        await emit_log(user_id, f"💰 CLOSED {pos['type'].upper()} @ ${actual_close_price:,.2f} | Net PnL: ${round(net_pnl, 2)}")
                        DatabaseHandler.save_state(user_id, bot)

                if datetime.now().timestamp() - last_log >= 60:
                    DatabaseHandler.save_state(user_id, bot)
                
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"❌ WS Stream Error: {e}"); await asyncio.sleep(5)
    finally:
        logger.info(f"🔌 Heartbeat loop terminated for {user_id}")

# =============================================================
# ENDPOINTS
# =============================================================
async def fetch_live_candles_ccxt(symbol: str, timeframe: str, limit: int, exchange_id: str = "coinbase"):
    """
    Dynamically fetches data from the correct exchange to prevent price mismatch.
    """
    # Initialize the correct exchange class dynamically
    exchange_class = getattr(ccxt, exchange_id)
    
    async with exchange_class({'enableRateLimit': True}) as ex:
        try:
            # Kraken uses standard format, Coinbase uses dashed. CCXT handles most, but we ensure '/'
            fetch_symbol = symbol.replace('-', '/')
            ohlcv = await ex.fetch_ohlcv(fetch_symbol, timeframe, limit=limit)
            return [{"time": int(c[0]/1000), "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5] if len(c) > 5 else 0} for c in ohlcv]
        except Exception as e:
            logger.error(f"❌ {exchange_id.upper()} Fetch Error: {e}")
            return []

@app.get("/api/ml/available-models")
@app.get("/ml/available-models")
@app.get("/api/models")
def list_models():
    if not os.path.exists(MODEL_DIR): return {"status": "success", "models": []}
    models = [{"id": f.rsplit('.', 1)[0], "name": f.rsplit('.', 1)[0]} 
              for f in os.listdir(MODEL_DIR) if f.endswith(('.keras', '.joblib', '.pkl'))]
    return {"status": "success", "models": models}

@app.post("/api/bot/start")
async def start_bot(data: BotStartRequest):
    user_id = data.userId.strip()
    raw_cap = data.config.get("capitalAllocation") or data.config.get("capital_allocation") or data.config.get("initialBalance")
    ui_capital = float(raw_cap) if raw_cap else 200.0 

    if user_id in TASK_REGISTRY:
        try:
            TASK_REGISTRY[user_id].cancel()
            logger.info(f"♻️ Registry: Cleaned old loop for {user_id}")
        except Exception as e:
            logger.error(f"⚠️ Registry Cleanup Error: {e}")
           
    initial_ohlcv = await fetch_live_candles_ccxt(data.config['symbol'], data.config.get('timeframe', '1h'), 150)
    processed_candles = []
    if initial_ohlcv:
        df_init = pd.DataFrame(initial_ohlcv)
        processed_candles = await process_data_packet(df_init, data.config.get('strategies', []))

    saved_state = DatabaseHandler.load_state(user_id)
    if saved_state:
        ACTIVE_BOTS[user_id] = saved_state
        ACTIVE_BOTS[user_id]["status"] = "running"
        ACTIVE_BOTS[user_id]["config"] = data.config
        ACTIVE_BOTS[user_id]["balance"] = ui_capital 
        ACTIVE_BOTS[user_id]["startedAt"] = datetime.now(timezone.utc).isoformat()
        await emit_log(user_id, f"♻️ SESSION RESET: Balance updated to ${ui_capital}")
    else:
        ACTIVE_BOTS[user_id] = {
            "status": "running",
            "config": data.config,
            "balance": ui_capital, 
            "positions": [],
            "trade_history": [],
            "equityCurve": [],
            "logs": [],
            "startedAt": datetime.now(timezone.utc).isoformat()
        }
        if not ACTIVE_BOTS[user_id].get("equityCurve"):
            ACTIVE_BOTS[user_id]["equityCurve"] = [{"time": datetime.now().isoformat(), "balance": ui_capital, "confidence": 50}]
        await emit_log(user_id, f"🚀 Engine Started. Portfolio: ${ui_capital}")

        if ACTIVE_BOTS[user_id]["positions"]:
            pos = ACTIVE_BOTS[user_id]["positions"][0]
            side = pos['type'].upper()
            # This forces the "Activated" message into the flow on every start/refresh
            await emit_log(user_id, f"⚡ TRADE DURING STARTUP: {side} Position Detected @ ${pos['entry']}")
        
        await emit_log(user_id, f"♻️ SESSION STARTED: Balance updated to ${ui_capital}")
    DatabaseHandler.save_state(user_id, ACTIVE_BOTS[user_id])
    
    await emit_status(user_id, {
        "status": "running", 
        "currentBalance": ACTIVE_BOTS[user_id]["balance"],
        "candles": processed_candles,
        "startedAt": ACTIVE_BOTS[user_id]["startedAt"]
    })

    loop = asyncio.get_event_loop()
    task = loop.create_task(live_neural_heartbeat(user_id))
    TASK_REGISTRY[user_id] = task
    return {"status": "running"}

@app.post("/api/bot/stop")
async def stop_bot(data: BotStopRequest):
    user_id = data.userId

    if user_id in TASK_REGISTRY:
        TASK_REGISTRY[user_id].cancel()
        del TASK_REGISTRY[user_id]
        logger.info(f"💀 Registry: Task killed for {user_id}")

    if user_id in ACTIVE_BOTS:
        ACTIVE_BOTS[user_id]["status"] = "stopped"
        ACTIVE_BOTS[user_id]["positions"] = []
        ACTIVE_BOTS[user_id]["trade_history"] = []
        ACTIVE_BOTS[user_id]["equityCurve"] = []
        ACTIVE_BOTS[user_id]["logs"] = []
        
        await emit_status(user_id, {
            "status": "stopped",
            "currentBalance": ACTIVE_BOTS[user_id]['balance'],
            "activePositions": [],
            "tradeMarkers": [],
            "equityCurve": [],
            "startedAt": None 
        })
        
        DatabaseHandler.save_state(user_id, ACTIVE_BOTS[user_id])
        del ACTIVE_BOTS[user_id]
        await emit_log(user_id, "💀 SYSTEM PURGED: Engine stopped and session reset.")
        return {"status": "stopped", "message": "Bot killed and reset"}
    return {"status": "stopped"}

@app.post("/api/bot/close_position")
async def close_position(data: BotClosePositionRequest):
    if data.userId in ACTIVE_BOTS and ACTIVE_BOTS[data.userId]['positions']:
        pos = ACTIVE_BOTS[data.userId]['positions'].pop(0)
        await emit_log(data.userId, f"⚠️ Manual Exit: {pos['type'].upper()} closed.")
        DatabaseHandler.save_state(data.userId, ACTIVE_BOTS[data.userId])
        return {"status": "closed"}
    raise HTTPException(status_code=400, detail="No active positions")

@app.post('/api/backtest/run')
async def run_backtest(request: BacktestRequest):
    try:
        # 1. Standardize UI request keys
        config = request.dict()
        
        # 🎯 KEY BRIDGING: Map UI names to Engine names
        final_thresh = config.get('mlThresholdLong') or config.get('ml_confidence_threshold') or 0.8
        config['mlThresholdLong'] = final_thresh
        config['mlThresholdShort'] = config.get('mlThresholdShort') or final_thresh
        config['risk_percentage'] = config.get('riskPercentage', 1.0)
        
        # 2. Format strategies list
        config['strategies'] = [{"code": request.code, "params": request.params}]

        # 3. Initialize and Run
        from app.backtest2 import Backtester
        tester = Backtester(config)
        result = await tester.run()

        # 🏁 THE FIX: Use a custom encoder or JSONResponse with manual dump to handle Timestamps
        # This prevents the 'Timestamp is not JSON serializable' error in Atomic runs
        return json.loads(json.dumps(result, default=str))

    except Exception as e:
        logger.error(f"❌ Simplified Atomic Run Error: {e}")
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})

        
@app.post('/api/backtest/combo')
async def run_combo_backtest(req: ComboRequest):
    async def event_generator():
        try:
            # 1. Setup Config
            config = req.dict()
            config['mlThresholdLong'] = config.get('mlThresholdLong', 0.8)
            config['mlThresholdShort'] = config.get('mlThresholdShort', 0.8)
            config['risk_percentage'] = config.get('risk_percentage', 1.0)

            # 📈 10%: Progress
            yield f"{json.dumps({'status': 'progress', 'percentage': 10, 'message': 'Assembling AI Council...'}, default=str)}\n"

            logger.info(f"⚖️ COMBO RUN START: {len(config.get('strategies', []))} Strategies | AI Limit: {config['mlThresholdLong']}")
            
            # 2. Initialize Engine
            from app.backtest2 import Backtester
            tester = Backtester(config)
            
            # 📈 30%: Progress
            yield f"{json.dumps({'status': 'progress', 'percentage': 30, 'message': 'Fetching Market History...'}, default=str)}\n"

            # 3. Execute Run
            result = await tester.run()
            
            # 📈 90%: Progress
            yield f"{json.dumps({'status': 'progress', 'percentage': 90, 'message': 'Finalizing Analytics...'}, default=str)}\n"

            # 🏁 100%: SUCCESS HANDSHAKE
            # 🚀 CRITICAL: default=str converts Pandas Timestamps to strings so JSON doesn't crash
            final_payload = {
                "status": "success",
                "result": result
            }
            yield f"{json.dumps(final_payload, default=str)}\n"

        except Exception as e:
            logger.error(f"❌ Combo Stream Error: {e}")
            # Ensure error messages are also string-serialized
            yield f"{json.dumps({'status': 'error', 'message': str(e)}, default=str)}\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",  # Prevents Nginx/Render buffering
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


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
async def reset_bot(data: BotStopRequest):
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
