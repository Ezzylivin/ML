import sys
import os

# ============================================================
# ⚙️ RUNTIME ENVIRONMENT PATH INJECTION
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "Project", "ML"))
if "/root/Project/ML" not in sys.path:
    sys.path.append("/root/Project/ML")

import asyncio
import logging
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
from fastapi.exceptions import RequestValidationError  # ✅ FIX 1: removed duplicate import
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
import aiohttp
from app.verify.engineer_and_train import apply_mega_features

# 🟢 SOCKET HELPERS (Must be async/await)
from app.services.socket_emitter import emit_log, emit_status

# ============================================================
# 🔧 IMPORT CONFIGURATION
# ============================================================
from app.config2 import (
    MODEL_DIR,
    MODEL_STORAGE_DIR,
    RESULTS_DIR,
    DEFAULT_TAKER_FEE,
    KRAKEN_TAKER_FEE,
    FEATURE_COLUMNS,
)
from app.predictors.stacking_predictor import StackingPredictor

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEO-Engine")

GLOBAL_SESSION: Optional[aiohttp.ClientSession] = None
ACTIVE_BOTS = {}
TASK_REGISTRY = {}

# ============================================================
# 🧠 PREDICTOR MODEL IN-MEMORY CACHE
# ============================================================
_PREDICTOR_CACHE: Dict[str, StackingPredictor] = {}

def get_cached_predictor(symbol: str, timeframe: str = "1h") -> StackingPredictor:
    """Load a StackingPredictor once and cache it in memory."""
    cache_key = f"{symbol}_{timeframe}"
    if cache_key not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[cache_key] = StackingPredictor(symbol=symbol, timeframe=timeframe)
        logger.info(f"🧠 Cached predictor for {cache_key}")
    return _PREDICTOR_CACHE[cache_key]


# ==========================================
# 🗄️ DATABASE HANDLER STATE STORAGE
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
    ml_confidence_threshold: Optional[float] = 0.8 
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
    riskPercentage: float = 1.0
    take_profit: Optional[float] = 0.06
    stop_loss: Optional[float] = 0.03
    trailing_stop: Optional[float] = 0.02
    mlModel: Optional[str] = "stacking"
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
    yield
    await GLOBAL_SESSION.close()
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
# HISTORICAL DATA MANAGEMENT LAYER
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


async def ensure_full_data(symbol, timeframe, start_str, end_str, *args, **kwargs):
    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_str).timestamp() * 1000)

    df = load_data_robust(symbol, timeframe)
    if df is None:
        df = pd.DataFrame()

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
        print(f"📡 DATA GAP: Fetching {symbol} from Coinbase...")
        exchange = ccxt.coinbase({'enableRateLimit': True})
        all_new_candles = []
        fetch_symbol = symbol.replace("-", "/")

        try:
            while current_since < end_ts:
                new_batch = await exchange.fetch_ohlcv(fetch_symbol, timeframe, since=current_since, limit=300)
                if not new_batch:
                    break
                all_new_candles.extend(new_batch)
                current_since = new_batch[-1][0] + 1
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
                os.makedirs("data", exist_ok=True)
                save_path = f"data/{symbol.replace('/', '-')}-{timeframe}.csv"
                df.to_csv(save_path)
                print(f"💾 Cached {len(df)} candles → {save_path}")
            except Exception as e:
                print(f"⚠️ Could not save candle cache: {e}")

    target_start = pd.to_datetime(start_str, utc=True)
    target_end = pd.to_datetime(end_str, utc=True)
    return df

# ==========================================
# DIAGNOSTIC LAYOUT METRIC COMPILER
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
            if conf < ui_limit:
                return f"🛑 AI VETO: Needs {int(ui_limit*100)}% (At {int(conf*100)}%)"
            
            pending = []
            for strat in strategies:
                code = strat.get('code')
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


class PredictiveRegimeOptimizer:
    @staticmethod
    def dynamically_tune_strategies(bot_config: Dict[str, Any], ai_score: float, current_adx: float) -> List[Dict[str, Any]]:
        if bot_config.get("comboConfig", {}).get("combinationRule") == "AND":
            return bot_config.get("strategies", [])
        if ai_score >= 0.68 or (ai_score > 0.55 and current_adx > 30):
            return [
                {"code": "supertrend",    "params": {"st_atr": 10, "st_factor": 3.0}},
                {"code": "pa_breakout",   "params": {"lookback": 20, "buffer": 0.01}},
                {"code": "ema_cloud",     "params": {"fast_ema": 9, "slow_ema": 21}},
                {"code": "sma_crossover", "params": {"fast_sma": 20, "slow_sma": 100}}
            ]
        elif ai_score <= 0.32 or (ai_score < 0.45 and current_adx > 30):
            return [
                {"code": "supertrend",     "params": {"st_atr": 10, "st_factor": 2.5}},
                {"code": "atr_breakout",   "params": {"atr_length": 14, "multiplier": 1.5}},
                {"code": "macd_crossover", "params": {"fast": 12, "slow": 26, "signal": 9}},
                {"code": "vol_profile",    "params": {"vol_ma": 20, "threshold": 1.2}}
            ]
        else:
            return [
                {"code": "bb_fade",        "params": {"bb_period": 20, "bb_std": 2.0}},
                {"code": "rsi_threshold",  "params": {"rsi_length": 14, "oversold": 30, "overbought": 70}},
                {"code": "stoch",          "params": {"k_period": 14, "d_period": 3}}
            ]


class NeuralPredictor:
    @staticmethod
    def get_prediction(model_id: str, df: pd.DataFrame, symbol: str = "BTC-USD") -> float:
        try:
            council = get_cached_predictor(symbol=symbol, timeframe="1h")
            prediction = council.get_prediction_score(df) if hasattr(council, 'get_prediction_score') else council.predict_direction(df)
            return float(prediction)
        except Exception as e:
            logger.error(f"❌ Council Predictor Error: {e}")
            return 0.5

async def process_data_packet(df: pd.DataFrame, strategies: list) -> list:
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


class StrategyBrain:
    @staticmethod
    def calculate_signals(df_raw: pd.DataFrame, df_ai: pd.DataFrame, config: Dict[str, Any], l_thresh: float, s_thresh: float, symbol="BTC-USD"):
        active_thoughts, votes = [], 0
        signals_map = {}
        strategies = config.get('strategies', [])
        current_price = df_raw['close'].iloc[-1]

        ema20 = ta.ema(df_raw['close'], length=20).iloc[-1]
        ema200 = ta.ema(df_raw['close'], length=200).iloc[-1]
        bb = ta.bbands(df_raw['close'], length=20, std=2.0)
        lower, mid, upper = bb.iloc[-1, 0], bb.iloc[-1, 1], bb.iloc[-1, 2]
        pr = int((current_price - lower) / (upper - lower) * 100)

        weighted_votes = 0.0

        for strat in strategies:
            code = strat.get('code')
            p = strat.get('params', {})
            raw_vote = 0
            confidence = 0.5

            try:
                if code == "rsi_threshold":
                    digits = int(p.get('rsi_length', 14))
                    rsi = ta.rsi(df_raw['close'], length=digits).iloc[-1]
                    dist = min(abs(rsi - 30), abs(rsi - 70))
                    confidence = max(0.1, min(1.0, 1.0 - (dist / 40)))
                    if rsi < p.get('oversold', 30): raw_vote = 1; active_thoughts.append("RSI Low")
                    elif rsi > p.get('overbought', 70): raw_vote = -1; active_thoughts.append("RSI High")

                elif code == "sma_crossover":
                    f = ta.sma(df_raw['close'], length=int(p.get('fast_sma', 50))).iloc[-1]
                    s = ta.sma(df_raw['close'], length=int(p.get('slow_sma', 200))).iloc[-1]
                    gap = abs(f - s) / s
                    confidence = 1.0 if f > s else max(0.1, min(0.95, 1.0 - (gap * 50)))
                    raw_vote = 1 if f > s else -1

                elif code == "macd_crossover":
                    macd = ta.macd(df_raw['close'], fast=int(p.get('fast', 12))).iloc[-1]
                    hist = macd[1]
                    norm_hist = abs(hist) / (current_price * 0.0005)
                    confidence = max(0.1, min(1.0, norm_hist))
                    raw_vote = 1 if macd[0] > macd[2] else -1

                elif code == "supertrend":
                    st_data = ta.supertrend(df_raw['high'], df_raw['low'], df_raw['close']).iloc[-1]
                    st_line = st_data[0]
                    dist = abs(current_price - st_line) / current_price
                    confidence = 1.0 if st_data[1] == 1 else max(0.1, min(0.95, 1.0 - (dist * 20)))
                    raw_vote = 1 if st_data[1] == 1 else -1

                elif code == "bb_fade":
                    confidence = max(0.1, min(1.0, pr / 100.0))
                    if current_price < lower: raw_vote = 1
                    elif current_price > upper: raw_vote = -1

                elif code == "atr_breakout":
                    atr = ta.atr(df_raw['high'], df_raw['low'], df_raw['close']).iloc[-1]
                    target = ema20 + (atr * float(p.get('multiplier', 1.5)))
                    confidence = max(0.1, min(1.0, current_price / target))
                    raw_vote = 1 if current_price > target else -1

                elif code == "pa_breakout":
                    lb = int(p.get('lookback', 20))
                    high_lb = df_raw['high'].tail(lb).max()
                    confidence = max(0.1, min(1.0, current_price / high_lb))
                    raw_vote = 1 if current_price >= high_lb else -1

                elif code == "vol_profile":
                    v_ma = ta.sma(df_raw['volume'], length=int(p.get('vol_ma', 20))).iloc[-1]
                    ratio = df_raw['volume'].iloc[-1] / (v_ma * float(p.get('threshold', 1.5)))
                    confidence = max(0.1, min(1.0, ratio))
                    raw_vote = (1 if current_price > mid else -1) if ratio >= 1.0 else 0

                elif code == "stoch":
                    stoch_df = ta.stoch(df_raw['high'], df_raw['low'], df_raw['close']).iloc[-1]
                    k = stoch_df[0]
                    dist = min(abs(k - 20), abs(k - 80))
                    confidence = max(0.1, min(1.0, 1.0 - (dist / 40)))
                    if k < 20: raw_vote = 1
                    elif k > 80: raw_vote = -1

                elif code == "ema_cloud":
                    f_ema = ta.ema(df_raw['close'], length=int(p.get('fast_ema', 9))).iloc[-1]
                    s_ema = ta.ema(df_raw['close'], length=int(p.get('slow_ema', 21))).iloc[-1]
                    gap = abs(f_ema - s_ema) / s_ema
                    confidence = 1.0 if f_ema > s_ema else max(0.1, min(0.95, 1.0 - (gap * 100)))
                    raw_vote = 1 if f_ema > s_ema else -1

                signals_map[code] = confidence
                votes += raw_vote  
                weighted_votes += raw_vote * confidence  

            except Exception:
                signals_map[code] = 0.0

        is_short = current_price < ema200
        ui_limit = float(config.get('mlThresholdShort', 0.55)) if is_short else float(config.get('mlThresholdLong', 0.55))
        conf = NeuralPredictor.get_prediction(config.get('mlModel', 'stacking'), df_ai, symbol=symbol)

        if config.get('mlMode') == 'off':
            gate_passed = True
        else:
            gate_passed = conf >= ui_limit if not is_short else (1.0 - conf) >= ui_limit

        rule = config.get('comboConfig', {}).get('combinationRule', 'OR')
        final_sig = 0

        if gate_passed:
            if rule == "AND":
                if votes >= len(strategies) and weighted_votes > 0: final_sig = 1
                elif votes <= -len(strategies) and weighted_votes < 0: final_sig = -1
            else:
                min_weighted = float(config.get('minWeightedSignal', 0.3))
                if votes > 0 and weighted_votes >= min_weighted: final_sig = 1
                elif votes < 0 and weighted_votes <= -min_weighted: final_sig = -1

        return final_sig, active_thoughts, {}, conf, signals_map


# ============================================================================
# 🚀 CORE ENGINE HEARTBEAT CALCULATOR (TIME-COOLDOWNS DEACTIVATED)
# ============================================================================
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

            max_p = min(5, int(config.get('maxPyramiding', 5)))
            start_capital = float(config.get('capitalAllocation', config.get('initialBalance', 200.0)))
            
            api_keys = bot.get('api_keys') or config.get('api_keys', {}) 
            trading_mode = config.get('trading_mode', 'paper').lower()
            use_margin = config.get('enable_shorting', False)
            leverage_val = float(config.get('leverage', 1.0))
            
            target_exchange = "kraken" if use_margin else "coinbase"
            has_valid_keys = bool(api_keys.get('krakenKey') if use_margin else api_keys.get('apiKey'))
            is_live_trading = (trading_mode == 'live') and has_valid_keys

            fee_rate = KRAKEN_TAKER_FEE if use_margin else DEFAULT_TAKER_FEE

            try:
                current_time = datetime.now(timezone.utc)
                # ✅ FIX 2: Define now_ts at the top of the inner try block so UI-update
                #           throttle and all downstream references resolve correctly.
                now_ts = time.time()

                # ------------------------------------------------------------
                # TIER 1: RAW MARKET PACKETS
                # ------------------------------------------------------------
                ohlcv_raw = await fetch_live_candles_ccxt(config['symbol'], config.get('timeframe', '1h'), 500, exchange_id=target_exchange)
                if not ohlcv_raw:
                    await asyncio.sleep(1); continue
                
                exchange_class = getattr(ccxt, target_exchange)
                async with exchange_class({'enableRateLimit': True}) as ex:
                    ticker = await ex.fetch_ticker(ticker_symbol.replace('-', '/'))
                    current_price = float(ticker['last'])
                
                ohlcv_raw[-1]['close'] = current_price
                df_raw = pd.DataFrame(ohlcv_raw)
                df_closed_history = df_raw.iloc[:-1].copy()
                df_ai, _ = await asyncio.to_thread(apply_mega_features, df_closed_history)

                current_ema200 = float(ta.ema(df_raw['close'], length=200).iloc[-1])
                current_atr = float(ta.atr(df_raw['high'], df_raw['low'], df_raw['close'], length=14).iloc[-1])
                
                atr_tp_mult = float(params.get('atr_tp_mult', config.get('atrTpMultiplier', 3.0)))
                atr_sl_mult = float(params.get('atr_sl_mult', config.get('atrSlMultiplier', 1.5)))
                
                try: current_adx = float(ta.adx(df_raw['high'], df_raw['low'], df_raw['close'], length=14).iloc[-1, 0])
                except Exception: current_adx = 25.0

                try:
                    vol_ma = ta.sma(df_raw['volume'], length=20).iloc[-1]
                    vol_ratio = df_raw['volume'].iloc[-1] / vol_ma if vol_ma > 0 else 1.0
                except Exception: vol_ratio = 1.0

                # ------------------------------------------------------------
                # TIER 2: SIGNAL COMPILATION
                # ------------------------------------------------------------
                conf_score = NeuralPredictor.get_prediction(config.get('mlModel', 'stacking'), df_ai, symbol=ticker_symbol)

                if config.get('mlMode') == 'on':
                    active_strategies = PredictiveRegimeOptimizer.dynamically_tune_strategies(config, conf_score, current_adx)
                else:
                    active_strategies = config.get('strategies', strategies)

                temp_config = {**config, 'strategies': active_strategies}

                sig, thoughts, nums, score, signals_map = StrategyBrain.calculate_signals(
                    df_raw, df_ai, temp_config, 0.5, 0.5, symbol=ticker_symbol
                )

                sentiment = "STRONG BUY" if score > 0.85 else "BUY" if score > 0.70 else "NEUTRAL"
                if score < 0.20: sentiment = "STRONG SELL"
                elif score < 0.35: sentiment = "SELL"

                last_pos = bot['positions'][-1] if bot['positions'] else None
                if last_pos:
                    pos_is_profitable = (
                        (last_pos['type'] == 'long'  and current_price > last_pos['entry']) or
                        (last_pos['type'] == 'short' and current_price < last_pos['entry'])
                    )
                    climb_satisfied = score >= last_pos.get('entry_conf', 0) + 0.05 and pos_is_profitable
                else:
                    climb_satisfied = True

                # ✅ FIX 3: Simplified is_short_allowed — previous ternary was logically equivalent
                #           to "always True when sig != -1" but reversed: allowed short only when
                #           shorting is enabled AND signal is short; all other signals pass through.
                is_short_allowed = (sig != -1) or config.get('enable_shorting', False)

                upnl = sum([
                    (current_price - p['entry']) * p['size'] if p['type'] == 'long' else (p['entry'] - current_price) * p['size']
                    for p in bot['positions']
                ])
                current_equity = bot['balance'] + upnl

                active_summary = ""
                if bot['positions']:
                    p = bot['positions'][-1]
                    stop_pct = round((abs(current_price - p['tsl']) / current_price) * 100, 2)
                    buffer_bar = DiagnosticLayer.render_progress(stop_pct, 2.0, reverse=True)
                    active_summary = f"⚡ ACTIVE: {len(bot['positions'])} POS (${upnl:,.2f}) | TSL {buffer_bar} {stop_pct}% | "

                # ------------------------------------------------------------
                # TIER 3: CRITICAL TIME-CONSTRAINT OVERRIDE (DEACTIVATED)
                # ------------------------------------------------------------
                ui_limit = float(config.get('mlThresholdShort', 0.55)) if current_price < current_ema200 else float(config.get('mlThresholdLong', 0.55))
                waiting_msg = DiagnosticLayer.get_pending_conditions(df_raw, temp_config, score, ui_limit)

                # ⚡ [TIME LOCK DEACTIVATED] -> Set instantly to True for real-time scaling
                time_gate_passed = True 

                atr_pct = (current_atr / current_price) * 100
                is_volatility_safe = float(config.get('minAtrPct', 0.1)) <= atr_pct <= float(config.get('maxAtrPct', 5.0))
                is_circuit_breaker_tripped = bot['balance'] <= (start_capital * (1.0 - (float(config.get('maxDailyLoss', 5.0)) / 100.0)))
                is_trend_aligned = (sig == 1 and current_price > current_ema200) or (sig == -1 and current_price < current_ema200) or (sig == 0)
                adx_trending = current_adx >= float(config.get('minAdx', 10.0))
                volume_confirmed = vol_ratio >= float(config.get('minVolRatio', 0.4))
                adaptive_gate = score >= ui_limit if config.get('mlMode') == 'on' else True
                market_gate_passed = True # Secondary reset buffer bypassed for instant entries

                all_filters_pass = (
                    is_volatility_safe and not is_circuit_breaker_tripped and
                    is_trend_aligned and adx_trending and volume_confirmed and
                    adaptive_gate and market_gate_passed and is_short_allowed and time_gate_passed
                )

                if len(bot['positions']) < max_p and all_filters_pass:
                    if (sig == 1 and climb_satisfied) or (sig == -1 and climb_satisfied):
                        trade_type = "long" if sig == 1 else "short"
                        raw_risk = config.get('riskPercentage', 1.0)
                        size_in_fiat = bot['balance'] * (float(raw_risk) / 100.0 / max_p)
                        size_in_crypto = size_in_fiat / current_price
                        safe_size = float(f"{size_in_crypto:.6f}")
                        actual_entry_price = current_price

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
                                    await user_exchange.load_markets()
                                    formatted_size = float(user_exchange.amount_to_precision(symbol, size_in_crypto))
                                    order = await user_exchange.create_market_order(symbol, side, formatted_size, params=order_params)
                                    actual_entry_price = order.get('average') or order.get('price') or current_price
                                    safe_size = order.get('filled') or formatted_size
                            except Exception as ex_err:
                                await emit_log(user_id, f"❌ ORDER FAILED: {str(ex_err)}")
                                await asyncio.sleep(1); continue
                        else:
                            bot['balance'] -= (safe_size * actual_entry_price) * fee_rate

                        tp_price_calc = actual_entry_price + (current_atr * atr_tp_mult) if sig == 1 else actual_entry_price - (current_atr * atr_tp_mult)
                        sl_price_calc = actual_entry_price - (current_atr * atr_sl_mult) if sig == 1 else actual_entry_price + (current_atr * atr_sl_mult)
                        partial_tp = actual_entry_price + (current_atr * 1.5) if sig == 1 else actual_entry_price - (current_atr * 1.5)

                        bot['positions'].append({
                            "symbol": symbol, "type": trade_type,
                            "entry": actual_entry_price, "size": safe_size, "time": current_time.isoformat(), "entry_conf": score,
                            "tp": tp_price_calc, "sl": sl_price_calc, "tsl": sl_price_calc, "partial_tp": partial_tp, "partial_taken": False,
                            "atr_at_entry": current_atr  # ✅ FIX 4: persist ATR at entry so trail distance is stable
                        })
                        bot['last_trade_time'] = current_time.isoformat()
                        await emit_log(user_id, f"🚀 ENTERED {trade_type.upper()} @ ${actual_entry_price:,.2f} | Live Threshold: {ui_limit:.2f}")
                        DatabaseHandler.save_state(user_id, bot)

                # ------------------------------------------------------------
                # TIER 4: EXIT PROCESSING (CONDITIONAL PARTIAL TP COMPILER)
                # ------------------------------------------------------------
                for pos in bot['positions'][:]:
                    closed = False
                    exit_reason = ""
                    pos_atr = pos.get('atr_at_entry', current_atr)
                    trail_dist = pos_atr * atr_sl_mult

                    # ✅ FIX 5: Handle partial TP for both long AND short positions.
                    #           Removed the `continue` so the TSL still updates on the same
                    #           tick after the partial exit — the remaining half stays protected.
                    if config.get('enablePartialExit', False) and not pos.get('partial_taken', False):
                        if pos['type'] == 'long' and current_price >= pos.get('partial_tp', float('inf')):
                            half_size = pos['size'] / 2.0
                            partial_pnl = (current_price - pos['entry']) * half_size
                            if not is_live_trading:
                                bot['balance'] += partial_pnl - (half_size * current_price * fee_rate)
                            pos['size'] = half_size
                            pos['partial_taken'] = True
                            pos['tsl'] = max(pos['tsl'], pos['entry'])
                            await emit_log(user_id, f"💎 PARTIAL LONG EXIT: 50% locked at ${current_price:,.2f}. Trailing stop set to breakeven.")
                            # Fall through — TSL update runs this tick on the remaining half

                        elif pos['type'] == 'short' and current_price <= pos.get('partial_tp', float('-inf')):
                            half_size = pos['size'] / 2.0
                            partial_pnl = (pos['entry'] - current_price) * half_size
                            if not is_live_trading:
                                bot['balance'] += partial_pnl - (half_size * current_price * fee_rate)
                            pos['size'] = half_size
                            pos['partial_taken'] = True
                            pos['tsl'] = min(pos['tsl'], pos['entry'])
                            await emit_log(user_id, f"💎 PARTIAL SHORT EXIT: 50% locked at ${current_price:,.2f}. Trailing stop set to breakeven.")
                            # Fall through — TSL update runs this tick on the remaining half

                    if pos['type'] == 'long':
                        new_tsl = current_price - trail_dist
                        if new_tsl > pos['tsl']: pos['tsl'] = new_tsl
                        if current_price >= pos['tp']: closed = True; exit_reason = "Take Profit"
                        elif current_price <= pos['tsl']: closed = True; exit_reason = "Trailing Stop"
                    else:
                        new_tsl = current_price + trail_dist
                        if new_tsl < pos['tsl']: pos['tsl'] = new_tsl
                        if current_price <= pos['tp']: closed = True; exit_reason = "Take Profit"
                        elif current_price >= pos['tsl']: closed = True; exit_reason = "Trailing Stop"

                    if closed:
                        actual_close_price = current_price
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
                                    await user_exchange.load_markets()
                                    fmt_size = float(user_exchange.amount_to_precision(symbol, pos['size']))
                                    order = await user_exchange.create_market_order(symbol, close_side, fmt_size, params=order_params)
                                    actual_close_price = order.get('average') or order.get('price') or current_price
                            except Exception as ex_err:
                                await emit_log(user_id, f"❌ EXIT ROUTE FAILURE: {str(ex_err)}")
                                continue

                        gross_pnl = (actual_close_price - pos['entry']) * pos['size'] if pos['type'] == 'long' else (pos['entry'] - actual_close_price) * pos['size']
                        net_pnl = gross_pnl - ((pos['size'] * actual_close_price) * fee_rate)
                        if not is_live_trading: bot['balance'] += net_pnl

                        bot['positions'].remove(pos)
                        bot['trade_history'].append({
                            "type": "exit", "side": pos['type'], "entry": pos['entry'],
                            "price": actual_close_price, "pnl": round(net_pnl, 2), "reason": exit_reason, "time": int(time.time() * 1000)
                        })
                        await emit_log(user_id, f"💰 CLOSED {pos['type'].upper()} via {exit_reason.upper()} | Net PnL: ${net_pnl:+.2f}")
                        DatabaseHandler.save_state(user_id, bot)

                if (now_ts - last_ui_update >= 10):
                    exposure_pct = round((sum([p['entry'] * p['size'] for p in bot['positions']]) / current_equity) * 100, 1) if current_equity > 0 else 0
                    latest_candles = await process_data_packet(df_raw, active_strategies)
                    bot["currentBalance"] = round(current_equity, 2)
                    bot["unrealizedPnl"] = round(upnl, 2)
                    bot["exposure"] = exposure_pct
                    bot["currentConfidence"] = int(score * 100)
                    bot["signalsMap"] = signals_map
                    bot["candles"] = latest_candles
                    
                    await emit_status(user_id, {
                        "status": "running", "currentBalance": bot["currentBalance"], "exposure": bot["exposure"],
                        "activePositions": bot['positions'], "unrealizedPnl": bot["unrealizedPnl"], "tradeMarkers": bot['trade_history'], 
                        "candles": bot["candles"], "currentConfidence": bot["currentConfidence"], "signalsMap": bot["signalsMap"], "initialCapital": start_capital
                    })
                    bot["equityCurve"].append({"time": datetime.now().isoformat(), "balance": round(current_equity, 2), "confidence": int(score * 100)})
                    if len(bot["equityCurve"]) > 300: bot["equityCurve"].pop(0)
                    last_ui_update = now_ts

                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"❌ WS Stream Error: {e}"); await asyncio.sleep(2)
    finally:
        logger.info(f"🔌 Heartbeat loop terminated for {user_id}")


# =============================================================
# API ENDPOINTS
# =============================================================
async def fetch_live_candles_ccxt(symbol: str, timeframe: str, limit: int, exchange_id: str = "coinbase"):
    exchange_class = getattr(ccxt, exchange_id)
    async with exchange_class({'enableRateLimit': True}) as ex:
        try:
            fetch_symbol = symbol.replace('-', '/')
            ohlcv = await ex.fetch_ohlcv(fetch_symbol, timeframe, limit=limit)
            return [{"time": int(c[0]/1000), "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5] if len(c) > 5 else 0} for c in ohlcv]
        except Exception as e:
            logger.error(f"❌ {exchange_id.upper()} Fetch Error: {e}")
            return []

@app.get("/api/ml/available-models")
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
        try: TASK_REGISTRY[user_id].cancel()
        except Exception: pass
           
    initial_ohlcv = await fetch_live_candles_ccxt(data.config['symbol'], data.config.get('timeframe', '1h'), 350)
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
    else:
        ACTIVE_BOTS[user_id] = {
            "status": "running", "config": data.config, "balance": ui_capital, 
            "positions": [], "trade_history": [], "equityCurve": [], "logs": [],
            "startedAt": datetime.now(timezone.utc).isoformat()
        }
    if not ACTIVE_BOTS[user_id].get("equityCurve"):
        ACTIVE_BOTS[user_id]["equityCurve"] = [{"time": datetime.now().isoformat(), "balance": ui_capital, "confidence": 50}]

    DatabaseHandler.save_state(user_id, ACTIVE_BOTS[user_id])
    await emit_status(user_id, {
        "status": "running", "currentBalance": ACTIVE_BOTS[user_id]["balance"],
        "candles": processed_candles, "startedAt": ACTIVE_BOTS[user_id]["startedAt"]
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
    if user_id in ACTIVE_BOTS:
        ACTIVE_BOTS[user_id]["status"] = "stopped"
        await emit_status(user_id, {
            "status": "stopped", "currentBalance": ACTIVE_BOTS[user_id]['balance'],
            "activePositions": [], "tradeMarkers": [], "equityCurve": [], "startedAt": None 
        })
        DatabaseHandler.save_state(user_id, ACTIVE_BOTS[user_id])
        del ACTIVE_BOTS[user_id]
        return {"status": "stopped"}
    return {"status": "stopped"}

@app.post("/api/bot/close_position")
async def close_position(data: BotClosePositionRequest):
    if data.userId in ACTIVE_BOTS and ACTIVE_BOTS[data.userId]['positions']:
        pos = ACTIVE_BOTS[data.userId]['positions'].pop(0)
        DatabaseHandler.save_state(data.userId, ACTIVE_BOTS[data.userId])
        return {"status": "closed"}
    raise HTTPException(status_code=400, detail="No active positions")

@app.get("/api/bot/status")
async def get_status(userId: str):
    bot = ACTIVE_BOTS.get(userId.strip())
    if not bot: bot = DatabaseHandler.load_state(userId.strip())
    if bot:
        return {
            "status": bot.get("status", "stopped"), 
            "currentBalance": bot.get("currentBalance", bot.get("balance", 0)),
            "balance": bot.get("balance", 0), "unrealizedPnl": bot.get("unrealizedPnl", 0),
            "exposure": bot.get("exposure", 0), "currentConfidence": bot.get("currentConfidence", 50),
            "signalsMap": bot.get("signalsMap", {}), "equityCurve": bot.get("equityCurve", []),
            "logs": bot.get("logs", []), "activePositions": bot.get("positions", []),
            "positions": bot.get("positions", []), "startedAt": bot.get("startedAt"),
            "config": bot.get("config"), "candles": bot.get("candles", []),
            "trade_history": bot.get('trade_history', []), "tradeMarkers": bot.get('trade_history', []),
            "aiRegimeTitle": bot.get("aiRegimeTitle", "Mean-Reverting Consolidation"),
            "aiRegimeDesc": bot.get("aiRegimeDesc", "Sideways Range"),
            "aiDeployedGear": bot.get("aiDeployedGear", "Syncing Strategy Matrix Data...")
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
