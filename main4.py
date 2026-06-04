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

# ============================================================
# 🔧 FIX #1: Import from config2 instead of redefining
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
# 🔧 FIX #4: MODEL CACHE (Tier 3, but critical for performance)
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
# Diagnostic Layer
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
                return f"🛑 AI VETO: Needs {int(ui_limit*100)}% (At {int(conf*100)}%)"
            
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


# ============================================================
# 🔮 10-STRATEGY PREDICTIVE ENSEMBLE REGIME OPTIMIZER
# ============================================================
class PredictiveRegimeOptimizer:
    @staticmethod
    def dynamically_tune_strategies(bot_config: Dict[str, Any], ai_score: float, current_adx: float) -> List[Dict[str, Any]]:
        """
        Dynamically coordinates all 10 system indicators into highly specific, 
        synergetic engine subsets depending upon the forward projected AI market phase.
        """
        if bot_config.get("comboConfig", {}).get("combinationRule") == "AND":
            return bot_config.get("strategies", [])

        # REGIME 1: Parabolic Bullish Momentum Expansion (4 Modules Activated)
        if ai_score >= 0.68 or (ai_score > 0.55 and current_adx > 30):
            return [
                {"code": "supertrend", "params": {"st_atr": 10, "st_factor": 3.0}},
                {"code": "pa_breakout", "params": {"lookback": 20, "buffer": 0.01}},
                {"code": "ema_cloud", "params": {"fast_ema": 9, "slow_ema": 21}},
                {"code": "sma_crossover", "params": {"fast_sma": 20, "slow_sma": 100}}
            ]

        # REGIME 2: Violent Bearish Breakout / Liquidation (4 Modules Activated)
        elif ai_score <= 0.32 or (ai_score < 0.45 and current_adx > 30):
            return [
                {"code": "supertrend", "params": {"st_atr": 10, "st_factor": 2.5}},
                {"code": "atr_breakout", "params": {"atr_length": 14, "multiplier": 1.5}},
                {"code": "macd_crossover", "params": {"fast": 12, "slow": 26, "signal": 9}},
                {"code": "vol_profile", "params": {"vol_ma": 20, "threshold": 1.2}}
            ]

        # REGIME 3: Mean-Reverting Choppy / Range Bound Market Structure (3 Modules Activated)
        else:
            return [
                {"code": "bb_fade", "params": {"bb_period": 20, "bb_std": 2.0}},
                {"code": "rsi_threshold", "params": {"rsi_length": 14, "oversold": 30, "overbought": 70}},
                {"code": "stoch", "params": {"k_period": 14, "d_period": 3}}
            ]


# ==========================================
# 🧠 1. NEURAL PREDICTOR
# ==========================================
class NeuralPredictor:
    @staticmethod
    def get_prediction(model_id: str, df: pd.DataFrame, symbol: str = "BTC-USD") -> float:
        """Uses the module-level cache. First call loads, rest are free."""
        try:
            council = get_cached_predictor(symbol=symbol, timeframe="1h")
            prediction = council.predict_direction(df)
            return float(prediction)
        except Exception as e:
            logger.error(f"🧠 Council Predictor Error: {e}")
            return 0.5


# ==========================================
# Packet Data
# ==========================================
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


# ============================================================
# 🔧 UPGRADED 2 & 10-STRATEGY WEIGHTED SIGNAL SYSTEM
# ============================================================
class StrategyBrain:
    @staticmethod
    def calculate_signals(df_raw: pd.DataFrame, df_ai: pd.DataFrame, config: Dict[str, Any], l_thresh: float, s_thresh: float, symbol="BTC-USD"):
        active_thoughts, votes = [], 0
        signals_map = {}
        strategies = config.get('strategies', [])
        current_price = df_raw['close'].iloc[-1]

        # Global indicators
        ema20 = ta.ema(df_raw['close'], length=20).iloc[-1]
        ema50 = ta.ema(df_raw['close'], length=50).iloc[-1]
        ema200 = ta.ema(df_raw['close'], length=200).iloc[-1]
        bb = ta.bbands(df_raw['close'], length=20, std=2.0)
        lower, mid, upper = bb.iloc[-1, 0], bb.iloc[-1, 1], bb.iloc[-1, 2]
        pr = int((current_price - lower) / (upper - lower) * 100)

        # 🔧 UPGRADE 1: WEIGHTED VOTE SYSTEM
        weighted_votes = 0.0

        for strat in strategies:
            code = strat.get('code')
            p = strat.get('params', {})
            raw_vote = 0
            confidence = 0.5

            try:
                if code == "rsi_threshold":
                    rsi = ta.rsi(df_raw['close'], length=int(p.get('rsi_length', 14))).iloc[-1]
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
                else:
                    confidence = 0.5

                signals_map[code] = confidence
                votes += raw_vote  # Keep integer votes for AND/OR logic
                weighted_votes += raw_vote * confidence  # Weighted signal strength

            except Exception:
                signals_map[code] = 0.0

        # ML Gate
        is_short = current_price < ema200
        ui_limit = float(config.get('mlThresholdShort', 0.55)) if is_short else float(config.get('mlThresholdLong', 0.55))
        conf = NeuralPredictor.get_prediction(config.get('mlModel', 'stacking'), df_ai, symbol=symbol)

        if config.get('mlMode') == 'off':
            gate_passed = True
            logic_desc = f"📊 LOGIC: NEURAL GATE BYPASSED (AI Score: {int(conf*100)}%) 🟡"
        else:
            gate_passed = conf >= ui_limit
            logic_desc = f"📊 LOGIC: {'SHORT' if is_short else 'LONG'} GATE {'PASSED' if gate_passed else 'VETOED'} ({int(conf*100)}% vs {int(ui_limit*100)}%) {'🟢' if gate_passed else '🔴'}"

        signal_names = " + ".join(active_thoughts) if active_thoughts else "Scanning Setup"
        gap_val = int(abs(current_price - ema50))
        intent_desc = f"🎯 INTENT: STALKING {'SHORT' if is_short else 'LONG'} ({signal_names} | Gap: ${gap_val}) {'🔴' if is_short else '🟢'}"
        trend_dist = current_price - ema200
        trend_text = f"📡 TREND: {'UP' if trend_dist > 0 else 'DOWN'} (Price ${int(abs(trend_dist))} {'above' if trend_dist > 0 else 'below'} 200EMA)"
        spread = ema20 - ema50
        bias_text = f"⚖️ BIAS: {'BULLISH' if spread > 0 else 'BEARISH'} (EMA Gap: ${int(abs(spread))})"
        mindset_str = "⚠️ OVEREXTENDED" if pr >= 80 else "🎯 ACCUMULATION" if pr <= 20 else "⚖️ EQUILIBRIUM"
        mindset_text = f"🤖 MINDSET: {mindset_str} ({pr}% of BB Range)"

        numeric_details = {
            "market": {
                "logic": logic_desc, "intent": intent_desc, "trend": trend_text,
                "bias": bias_text, "mindset": mindset_text,
                "weighted_signal_strength": round(weighted_votes, 3)
            }
        }

        # Final signal using BOTH binary votes (for AND/OR rule) and weighted strength
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

        return final_sig, active_thoughts, numeric_details, conf, signals_map


# ============================================================================
# 🚀 3. THE HEARTBEAT (Perfect Top-Down Synchronized Calculation Loop)
# ============================================================================
async def live_neural_heartbeat(user_id: str):
    signal_has_reset = True
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

            if trading_mode == 'live' and not is_live_trading:
                logger.warning(f"User {user_id} selected LIVE but lacks API keys. Falling back to PAPER.")
                await emit_log(user_id, "⚠️ LIVE MODE FAILED: Missing API Keys. Forcing PAPER MODE.")
                config['trading_mode'] = 'paper'
                bot['config'] = config 
                DatabaseHandler.save_state(user_id, bot)

            fee_rate = KRAKEN_TAKER_FEE if use_margin else DEFAULT_TAKER_FEE

            try:

                current_time = datetime.now(timezone.utc)
                # ------------------------------------------------------------
                # TIER 1: CORE MARKET FEED FETCHING & RAW INDICATORS
                # ------------------------------------------------------------
                ohlcv_raw = await fetch_live_candles_ccxt(config['symbol'], config.get('timeframe', '1h'), 500, exchange_id=target_exchange)
                
                if not ohlcv_raw:
                    await emit_log(user_id, f"⚠️ {target_exchange.upper()} Feed Unstable - Retrying...")
                    await asyncio.sleep(5); continue
                
                exchange_class = getattr(ccxt, target_exchange)
                async with exchange_class({'enableRateLimit': True}) as ex:
                    ticker = await ex.fetch_ticker(ticker_symbol.replace('-', '/'))
                    current_price = float(ticker['last'])
                
                ohlcv_raw[-1]['close'] = current_price
                df_raw = pd.DataFrame(ohlcv_raw)
                df_closed_history = df_raw.iloc[:-1].copy()
                df_ai, _ = await asyncio.to_thread(apply_mega_features, df_closed_history)

                # Initialize deep mathematical indicators immediately to resolve scope dependencies
                current_ema200 = float(ta.ema(df_raw['close'], length=200).iloc[-1])
                current_atr = float(ta.atr(df_raw['high'], df_raw['low'], df_raw['close'], length=14).iloc[-1])

                atr_tp_mult = float(params.get('atr_tp_mult', config.get('atrTpMultiplier', 3.0)))
                atr_sl_mult = float(params.get('atr_sl_mult', config.get('atrSlMultiplier', 1.5)))
                
                try:
                    current_adx = float(ta.adx(df_raw['high'], df_raw['low'], df_raw['close'], length=14).iloc[-1, 0])
                except Exception:
                    current_adx = 25.0

                try:
                    vol_ma = float(ta.sma(df_raw['volume'], length=20).iloc[-1])
                    vol_ratio = df_raw['volume'].iloc[-1] / vol_ma if vol_ma > 0 else 1.0
                except Exception:
                    vol_ratio = 1.0

                # ------------------------------------------------------------
                # TIER 2: AI COUNCILS & STRATEGY SIGNAL GENERATION
                # ------------------------------------------------------------
                conf_score = NeuralPredictor.get_prediction(config.get('mlModel', 'stacking'), df_ai, symbol=ticker_symbol)

                if config.get('mlMode') == 'on':
                    optimized_modules = PredictiveRegimeOptimizer.dynamically_tune_strategies(config, conf_score, current_adx)
                    config['strategies'] = optimized_modules
                    bot['config']['strategies'] = optimized_modules
                    strategies = optimized_modules 
                
                sig, thoughts, nums, score, signals_map = StrategyBrain.calculate_signals(
                    df_raw, df_ai, config, 0.5, 0.5, symbol=ticker_symbol
                )

                # ------------------------------------------------------------
                # TIER 3: CHRONOLOGICAL PROP-MAPPING & SHORT FILTERS
                # ------------------------------------------------------------
                sentiment = "STRONG BUY" if score > 0.85 else "BUY" if score > 0.70 else "NEUTRAL"
                if score < 0.20: sentiment = "STRONG SELL"
                elif score < 0.35: sentiment = "SELL"

                last_pos = bot['positions'][-1] if bot['positions'] else None
                climb_satisfied = (score >= last_pos.get('entry_conf', 0) + 0.10) if last_pos else True

                # Interceptor checks for Spot account short restrictions
                is_short_allowed = True
                if sig == -1 and not config.get('enable_shorting', False):
                    is_short_allowed = False

                #if sig == -1 and len(bot['positions']) == 0 and not is_short_allowed:
                    #sig = 0

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

                # ------------------------------------------------------------
                # TIER 4: DYNAMIC EXECUTIONS RISKS FILTERS AND RISK GATEWAY
                # ------------------------------------------------------------
                ui_limit = float(config.get('mlThresholdShort', 0.55)) if current_price < current_ema200 else float(config.get('mlThresholdLong', 0.55))
                waiting_msg = DiagnosticLayer.get_pending_conditions(df_raw, config, score, ui_limit)

                base_threshold = ui_limit  
                if current_adx > 30:
                    adaptive_threshold = max(0.10, base_threshold - 0.05)
                elif current_adx < 20:
                    adaptive_threshold = min(0.95, base_threshold + 0.05)
                else:
                    adaptive_threshold = base_threshold
                
                adaptive_gate = score >= adaptive_threshold if config.get('mlMode') == 'on' else True

                recent_trades = [t for t in bot.get('trade_history', []) if t.get('type') == 'exit'][-5:]
                consecutive_losses = 0
                for t in reversed(recent_trades):
                    if float(t.get('pnl', 0)) < 0: consecutive_losses += 1
                    else: break

                if consecutive_losses >= 3:
                    adaptive_threshold = min(0.85, adaptive_threshold + (consecutive_losses - 2) * 0.03)
                    if config.get('mlMode') == 'on':
                        adaptive_gate = score >= adaptive_threshold
                        
                    if consecutive_losses == 3 and (datetime.now().timestamp() - last_log >= 15):
                        await emit_log(user_id, f"⚠️ COLD STREAK: {consecutive_losses} losses — raising bar to {int(adaptive_threshold*100)}%")

                atr_pct = (current_atr / current_price) * 100
                is_volatility_safe = atr_pct <= float(config.get('maxAtrPct', 3.0))

                max_dd_pct = float(config.get('maxDailyLoss', 5.0)) / 100.0
                is_circuit_breaker_tripped = bot['balance'] <= (start_capital * (1.0 - max_dd_pct))

                # Clean structural loop state tracking reset architecture
                if len(bot['positions']) == 0 and sig == 0:
                    signal_has_reset = True
                elif len(bot['positions']) > 0:
                    signal_has_reset = False

                market_gate_passed = signal_has_reset if len(bot['positions']) == 0 else True

                # 🌟 FIX SOLUTION: Explicitly compile adx_trending before passing to safety filter matrices
                try:
                    adx_trending = current_adx >= float(config.get('minAdx', 20.0))
                except Exception:
                    adx_trending = True

                volume_confirmed = True  # Permanently bypassed per volume settings configuration

                rule = config.get('comboConfig', {}).get('combinationRule', 'OR')
                if config.get('mlMode') == 'off' or rule == "OR":
                    is_trend_aligned = True
                else:
                    is_trend_aligned = (sig == 1 and current_price > current_ema200) or (sig == -1 and current_price < current_ema200)

                all_filters_pass = (
                    is_volatility_safe and 
                    not is_circuit_breaker_tripped and 
                    is_trend_aligned and
                    adx_trending and 
                    volume_confirmed and 
                    adaptive_gate and 
                    market_gate_passed and 
                    is_short_allowed
                )

                # ------------------------------------------------------------
                # TIER 5: TELEMETRY TRANSMISSION PACKAGING (WebSockets Sync)
                # ------------------------------------------------------------
                if "vetoed_signals" not in bot:
                    bot["vetoed_signals"] = []

                raw_sig = 0
                votes = sum([1 if signals_map.get(s['code'], 0) > 0.5 else -1 for s in strategies])
                if rule == "AND":
                    if votes >= len(strategies): raw_sig = 1
                    elif votes <= -len(strategies): raw_sig = -1
                else:
                    if votes > 0: raw_sig = 1
                    elif votes < 0: raw_sig = -1

                if raw_sig != 0 and score < ui_limit:
                    if not bot["vetoed_signals"] or (datetime.now(timezone.utc) - datetime.fromisoformat(bot["vetoed_signals"][-1]["time"])).total_seconds() > 300:
                        bot["vetoed_signals"].append({
                            "time": datetime.now(timezone.utc).isoformat(), "signal": "Long" if raw_sig == 1 else "Short",
                            "conf_score": round(score, 4), "limit": round(ui_limit, 4), "price": current_price
                        })
                        if len(bot["vetoed_signals"]) > 100: bot["vetoed_signals"].pop(0)
                
                if len(bot['positions']) < int(config.get('maxPyramiding', 1)):
                    hunting_summary = f"🏹 STALKING LEG {len(bot['positions'])+1}: {int(score*100)}% ({sentiment}) | {waiting_msg}"
                else:
                    hunting_summary = "✅ PYRAMID FULL: Managing Exits"

                targets_str = " | ".join([f"{s['code'].upper()}: {DiagnosticLayer.render_progress(int(signals_map.get(s['code'], 0) * 100), 100, reverse=True)}" for s in strategies])
                combined_status = f"{active_summary}{hunting_summary} | 🔍 {targets_str}"

                now_ts = datetime.now().timestamp()
                if (now_ts - last_log >= 15):
                    await emit_log(user_id, combined_status)
                    last_log = now_ts

                if (now_ts - last_ui_update >= 10):
                    exposure_pct = round((sum([p['entry'] * p['size'] for p in bot['positions']]) / current_equity) * 100, 1) if current_equity > 0 else 0
                    latest_candles = await process_data_packet(df_raw, strategies)
                    
                    if config.get('mlMode') == 'on':
                        if conf_score >= 0.68 or (conf_score > 0.55 and current_adx > 30):
                            deployed_gear = "Trend Armor (SuperTrend, PA Breakout, EMA Cloud, SMA Cross)"
                        elif conf_score <= 0.32 or (conf_score < 0.45 and current_adx > 30):
                            deployed_gear = "Capitulation Armor (SuperTrend, ATR Breakout, MACD Cross, Vol Profile)"
                        else:
                            deployed_gear = "Range Armor (Bollinger Bands, RSI Threshold, Stochastic Oscillator)"
                    else:
                        deployed_gear = f"Custom Suite ({', '.join([s['code'].upper() for s in strategies])})"

                    if len(bot['positions']) >= max_p:
                        regime_title = "Portfolio Full"
                        regime_desc = f"Capital allocation optimized. The bot has safely filled all available risk slots ({len(bot['positions'])}/{max_p}) to maintain institutional asset diversification. Trading pipelines are locked; the engine is now exclusively managing active trailing stops and profit targets."
                    elif sig != 0 and all_filters_pass:
                        regime_title = "Executing Entry"
                        regime_desc = f"All quantitative and structural check-gates cleared! Technical and neural systems have achieved absolute confluence. Dispatching open market {'BUY (Long)' if sig == 1 else 'SELL (Short)'} order route to the exchange order book at ${current_price:,.2f}."
                    elif sig != 0 and not all_filters_pass:
                        regime_title = "Entry Guarded"
                        reasons_snapshot = []
                        if not is_short_allowed:
                            reasons_snapshot.append(f"Capital Type Mismatch: A technical sell model triggered a SHORT entry request, but your profile configuration is strictly set to SPOT mode. To protect capital, the bot has blocked it. Spot portfolios only allow asset appreciation; holding for a compliant buy signal.")
                        elif not is_trend_aligned: 
                            reasons_snapshot.append(f"Macro Trend Protection: To maximize win-rate safety, the bot is waiting for live price (${current_price:,.2f}) to print a definitive structural breakout {'above' if sig == 1 else 'below'} the institutional 200 EMA baseline (${current_ema200:,.2f}) to ensure we are trading in harmony with long-term market direction.")
                        elif not adx_trending: 
                            reasons_snapshot.append(f"Chop-Loss Mitigation: The price is grinding sideways, exposing trades to bad fills. The engine is waiting for trend velocity (ADX) to cross back above our required momentum floor of {float(config.get('minAdx', 20.0)):.1f} to ensure a healthy, sustained price expansion is underway (Current ADX: {current_adx:.1f}).")
                        elif not volume_confirmed: 
                            reasons_snapshot.append(f"Institutional Volume Filter: The current price breakout lacks institutional backing. The engine is waiting for network transaction volume to cross above {float(config.get('minVolRatio', 0.8)):.2f}x of the 20-period moving average to prove the move is driven by smart money instead of a retail fake-out (Current Vol Ratio: {vol_ratio:.2f}x).")
                        elif not is_volatility_safe: 
                            reasons_snapshot.append(f"Extreme Volatility Defense: Market price swings are too erratic for a predictable entry. The bot is waiting for price action to stabilize and compress below our maximum security ceiling of {float(config.get('maxAtrPct', 3.0)):.1f}% of price to guarantee reliable stop-loss insurance coverage (Current ATR Volatility: {atr_pct:.2f}%).")
                        elif not adaptive_gate: 
                            reasons_snapshot.append(f"AI Council Veto: While individual chart indicators want to enter, our machine learning model smells a trap. The bot is withholding entry until our Ensemble Stacking Predictor's unified confidence score meets or breaches our strict risk safety threshold of {int(adaptive_threshold*100)}% (Current AI Confidence: {int(score*100)}%).")
                        elif not market_gate_passed: 
                            reasons_snapshot.append(f"Execution Timing Guard: Technical indicators are highly overextended. The bot is pausing execution to prevent 'chasing the market' late at the tip of a price leg. Waiting for active strategy modules to cleanly cycle back to a Neutral (0) baseline for a safer entry point.")
                        elif is_circuit_breaker_tripped: 
                            reasons_snapshot.append(f"Drawdown Protection Active: The bot has executed an automatic risk shutdown to insulate your capital. Net equity has breached your protective daily defensive drawdown firewall threshold of {float(config.get('maxDailyLoss', 5.0)):.1f}%. Systems are frozen until the daily interval resets to maintain strict capital preservation discipline.")
                        
                        regime_desc = reasons_snapshot[0] if reasons_snapshot else "Defensive safety filters engaged. Holding routing execution until downstream matrix parameters optimize."
                    else:
                        has_rsi_indicators = any(s['code'] == 'rsi_threshold' for s in strategies) or any(s['code'] == 'stoch' or s['code'] == 'bb_fade' for s in strategies)
                        has_trend_indicators = any(s['code'] == 'pa_breakout' for s in strategies) or any(s['code'] == 'supertrend' or s['code'] == 'atr_breakout' for s in strategies)
                        if has_rsi_indicators and has_trend_indicators and any(signals_map.get(s['code'], 0) >= 0.75 for s in strategies):
                            regime_title = "Indicators Disagree"
                            regime_desc = "System Standoff: Technical sub-modules are conflicting. Mean-reverting tools indicate the asset is overbought/oversold, while trend-following blocks show breakout momentum. Capital remains safe in cash until our quantitative layers achieve absolute agreement."
                        elif consecutive_losses >= 3:
                            regime_title = "Cold-Streak Shield Active"
                            regime_desc = f"Strategic Drawdown Shield: Following {consecutive_losses} consecutive small losses, the bot has automatically raised its security requirements. Mathematical entry parameters have been tightened by +{int((consecutive_losses - 2) * 3)}% to insulate your balance and fish exclusively for gold-medal setups."
                        else:
                            regime_title = "Quiet Market Structure"
                            regime_desc = f"Consolidation Phase: The market has flatlined sideways at ${current_price:,.2f} within a balanced consolidation range. Waiting for indicator triggers or an AI confidence spike past {int(adaptive_threshold*100)}% to capitalize on an explosive breakout."

                    closed_trades = [t for t in bot.get('trade_history', []) if t.get('type') == 'exit']
                    total_closed = len(closed_trades)
                    if total_closed > 0:
                        wins = len([t for t in closed_trades if float(t.get('pnl', 0)) > 0])
                        win_rate = round((wins / total_closed) * 100, 1)
                        gross_profits = sum([float(t.get('pnl', 0)) for t in closed_trades if float(t.get('pnl', 0)) > 0])
                        gross_losses = abs(sum([float(t.get('pnl', 0)) for t in closed_trades if float(t.get('pnl', 0)) < 0]))
                        profit_factor = round(gross_profits / gross_losses, 2) if gross_losses > 0 else round(gross_profits, 2) if gross_profits > 0 else 1.0
                    else:
                        win_rate, profit_factor = 0.0, 1.0

                    bot["currentBalance"] = round(current_equity, 2)
                    bot["unrealizedPnl"] = round(upnl, 2)
                    bot["exposure"] = exposure_pct
                    bot["currentConfidence"] = int(score * 100)
                    bot["signalsMap"] = signals_map
                    bot["candles"] = latest_candles
                    bot["aiRegimeTitle"] = regime_title
                    bot["aiRegimeDesc"] = regime_desc
                    bot["aiDeployedGear"] = deployed_gear
                    bot["winRate"] = win_rate
                    bot["profitFactor"] = profit_factor

                    await emit_status(user_id, {
                        "status": "running", "currentBalance": bot["currentBalance"], "exposure": bot["exposure"],
                        "activePositions": bot['positions'], "unrealizedPnl": bot["unrealizedPnl"], "tradeMarkers": bot['trade_history'], 
                        "candles": bot["candles"], "currentConfidence": bot["currentConfidence"], "signalsMap": bot["signalsMap"],
                        "initialCapital": start_capital, "aiRegimeTitle": bot["aiRegimeTitle"], "aiRegimeDesc": bot["aiRegimeDesc"],
                        "aiDeployedGear": bot["aiDeployedGear"], "winRate": bot["winRate"], "profitFactor": bot["profitFactor"]
                    })
                    bot["equityCurve"].append({"time": datetime.now().isoformat(), "balance": round(current_equity, 2), "confidence": int(score * 100)})
                    if len(bot["equityCurve"]) > 300: bot["equityCurve"].pop(0)
                    last_ui_update = now_ts

                # ------------------------------------------------------------
                # TIER 6: REJECTION CONSOLE REGISTRY FIELD LOGGER
                # ------------------------------------------------------------
                if sig != 0 and not all_filters_pass:
                    reasons = []
                    if not is_short_allowed: reasons.append("Spot mode cannot short sell")
                    if not is_volatility_safe: reasons.append(f"ATR Volatility too high ({atr_pct:.1f}%)")
                    if is_circuit_breaker_tripped: reasons.append("Daily Circuit Breaker Tripped")
                    if not is_trend_aligned: reasons.append("Trend misaligned with 200 EMA")
                    if not adx_trending: reasons.append(f"ADX Momentum too low ({current_adx:.0f})")
                    if not volume_confirmed: reasons.append(f"Volume weak ({vol_ratio:.2f}x)")
                    if not adaptive_gate: reasons.append(f"AI Veto (Confidence {int(score*100)}% < Required {int(adaptive_threshold*100)}%)")
                    if not market_gate_passed: reasons.append("Signal Guard active (Waiting for strategy to reset to Neutral)")
                    if not reasons: reasons.append("Execution lock (Pre-flight safety check failed to clear)")
                    await emit_log(user_id, f"⛔ ENTRY BLOCKED: {', '.join(reasons)}")

                # ------------------------------------------------------------
                # TIER 7: LIVE TRANSACTION DISPATCH (ENTRY OPERATIONS)
                # ------------------------------------------------------------
                if len(bot['positions']) < max_p and all_filters_pass:
                    if (sig == 1 and climb_satisfied) or (sig == -1 and climb_satisfied):
                        trade_type = "long" if sig == 1 else "short"
                        raw_risk = config.get('riskPercentage') or config.get('risk_percentage') or 1.0
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
                                    if formatted_size <= 0: raise Exception("Order size too small")
                                    
                                    order = await user_exchange.create_market_order(symbol, side, formatted_size, params=order_params)
                                    actual_entry_price = order.get('average') or order.get('price') or current_price
                                    safe_size = order.get('filled') or formatted_size
                            except Exception as ex_err:
                                await emit_log(user_id, f"❌ {target_exchange.upper()} ORDER FAILED: {str(ex_err)}")
                                await asyncio.sleep(5); continue
                        else:
                            bot['balance'] -= (safe_size * actual_entry_price) * fee_rate

                        tp_price_calc = actual_entry_price + (current_atr * atr_tp_mult) if sig == 1 else actual_entry_price - (current_atr * atr_tp_mult)
                        sl_price_calc = actual_entry_price - (current_atr * atr_sl_mult) if sig == 1 else actual_entry_price + (current_atr * atr_sl_mult)
                        partial_tp = actual_entry_price + (current_atr * 1.5) if sig == 1 else actual_entry_price - (current_atr * 1.5)

                        bot['positions'].append({
                            "symbol": symbol, "type": trade_type, "entry": actual_entry_price, "size": safe_size,
                            "time": current_time.isoformat(), "entry_conf": score, "tp": tp_price_calc, "sl": sl_price_calc, "tsl": sl_price_calc,
                            "partial_tp": partial_tp, "partial_taken": False, "atr_at_entry": round(current_atr, 4),
                            "adx_at_entry": round(current_adx, 1), "vol_ratio_at_entry": round(vol_ratio, 2)
                        })
                        bot['last_trade_time'] = current_time.isoformat()
                        rr = round(atr_tp_mult / atr_sl_mult, 1)
                        await emit_log(user_id, f"🚀 ENTERED {trade_type.upper()} @ ${actual_entry_price:,.2f} | TP ${tp_price_calc:,.2f} | SL ${sl_price_calc:,.2f} | R:R {rr}")
                        DatabaseHandler.save_state(user_id, bot)

                # ------------------------------------------------------------
                # TIER 8: MANAGEMENT SLOTS & TRAILING FIRES (EXIT OPERATIONS)
                # ------------------------------------------------------------
                for pos in bot['positions'][:]:
                    closed = False
                    exit_reason = ""
                    pos_atr = pos.get('atr_at_entry', current_atr)
                    trail_dist = pos_atr * atr_sl_mult

                    if not pos.get('partial_taken', True):
                        if pos['type'] == 'long' and current_price >= pos.get('partial_tp', float('inf')):
                            half_size = pos['size'] / 2.0
                            partial_pnl = (current_price - pos['entry']) * half_size
                            if is_live_trading:
                                try:
                                    exchange_class = getattr(ccxt, target_exchange)
                                    async with exchange_class({
                                        'apiKey': api_keys.get('krakenKey', api_keys.get('apiKey')),
                                        'secret': api_keys.get('krakenSecret', api_keys.get('secret')),
                                        'enableRateLimit': True
                                    }) as user_exchange:
                                        await user_exchange.load_markets()
                                        fmt_size = float(user_exchange.amount_to_precision(symbol, half_size))
                                        order = await user_exchange.create_market_order(symbol, "sell", fmt_size)
                                        partial_pnl = ((order.get('average') or current_price) - pos['entry']) * half_size
                                except Exception as e:
                                    await emit_log(user_id, f"⚠️ Partial exit failed: {e}")
                            else:
                                net_partial = partial_pnl - (half_size * current_price * fee_rate)
                                bot['balance'] += net_partial
                                partial_pnl = net_partial

                            pos['size'] = half_size
                            pos['partial_taken'] = True
                            pos['tsl'] = max(pos['tsl'], pos['entry'])
                            await emit_log(user_id, f"💎 PARTIAL EXIT: Took 50% at ${current_price:,.2f} | Realized ${partial_pnl:.2f}")
                            bot['trade_history'].append({
                                "type": "partial_exit", "side": pos['type'], "price": current_price, "pnl": round(partial_pnl, 2),
                                "time": int(time.time() * 1000)
                            })
                            DatabaseHandler.save_state(user_id, bot)
                            continue

                        elif pos['type'] == 'short' and current_price <= pos.get('partial_tp', 0):
                            half_size = pos['size'] / 2.0
                            partial_pnl = (pos['entry'] - current_price) * half_size
                            if not is_live_trading:
                                net_partial = partial_pnl - (half_size * current_price * fee_rate)
                                bot['balance'] += net_partial
                                partial_pnl = net_partial

                            pos['size'] = half_size
                            pos['partial_taken'] = True
                            pos['tsl'] = min(pos['tsl'], pos['entry'])
                            await emit_log(user_id, f"💎 PARTIAL EXIT: Took 50% SHORT at ${current_price:,.2f} | Realized ${partial_pnl:.2f}")
                            continue

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
                                    balance_data = await user_exchange.fetch_balance()
                                    real_fiat = balance_data.get('USD', {}).get('free') or balance_data.get('USDC', {}).get('free')
                                    if real_fiat: bot['balance'] = float(real_fiat)
                            except Exception as ex_err:
                                await emit_log(user_id, f"❌ EXIT FAILED: {str(ex_err)}"); continue

                        gross_pnl = (actual_close_price - pos['entry']) * pos['size'] if pos['type'] == 'long' else (pos['entry'] - actual_close_price) * pos['size']
                        net_pnl = gross_pnl - ((pos['size'] * actual_close_price) * fee_rate)
                        if not is_live_trading: bot['balance'] += net_pnl

                        bot['positions'].remove(pos)
                        bot['trade_history'].append({
                            "type": "exit", "side": pos['type'], "price": actual_close_price, "pnl": round(net_pnl, 2),
                            "reason": exit_reason, "time": datetime.now(timezone.utc).isoformat()
                        })
                        await emit_log(user_id, f"💰 {exit_reason}: CLOSED {pos['type'].upper()} @ ${actual_close_price:,.2f} | Net PnL: ${round(net_pnl, 2)}")
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
        
        ACTIVE_BOTS[user_id]["trade_history"] = []
        ACTIVE_BOTS[user_id]["logs"] = []
        ACTIVE_BOTS[user_id]["positions"] = []
        ACTIVE_BOTS[user_id]["equityCurve"] = [{"time": datetime.now().isoformat(), "balance": ui_capital, "confidence": 50}]
        
        await emit_log(user_id, f"♻️ SESSION INITIALIZED: Fresh slate at ${ui_capital}")
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
        config = request.dict()
        final_thresh = config.get('mlThresholdLong') or config.get('ml_confidence_threshold') or 0.8
        config['mlThresholdLong'] = final_thresh
        config['mlThresholdShort'] = config.get('mlThresholdShort') or final_thresh
        config['risk_percentage'] = config.get('riskPercentage', 1.0)
        config['strategies'] = [{"code": request.code, "params": request.params}]

        from app.backtest2 import Backtester
        tester = Backtester(config)
        result = await tester.run()
        return json.loads(json.dumps(result, default=str))
    except Exception as e:
        logger.error(f"❌ Simplified Atomic Run Error: {e}")
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})

@app.post('/api/backtest/combo')
async def run_combo_backtest(req: ComboRequest):
    async def event_generator():
        try:
            config = req.dict()
            config['mlThresholdLong'] = config.get('mlThresholdLong', 0.8)
            config['mlThresholdShort'] = config.get('mlThresholdShort', 0.8)
            config['risk_percentage'] = config.get('risk_percentage', 1.0)

            yield f"{json.dumps({'status': 'progress', 'percentage': 10, 'message': 'Assembling AI Council...'}, default=str)}\n"
            logger.info(f"⚖️ COMBO RUN START: {len(config.get('strategies', []))} Strategies")
            
            from app.backtest2 import Backtester
            tester = Backtester(config)
            yield f"{json.dumps({'status': 'progress', 'percentage': 30, 'message': 'Fetching Market History...'}, default=str)}\n"

            result = await tester.run()
            yield f"{json.dumps({'status': 'progress', 'percentage': 90, 'message': 'Finalizing Analytics...'}, default=str)}\n"

            final_payload = {"status": "success", "result": result}
            yield f"{json.dumps(final_payload, default=str)}\n"
        except Exception as e:
            logger.error(f"❌ Combo Stream Error: {e}")
            yield f"{json.dumps({'status': 'error', 'message': str(e)}, default=str)}\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


# ============================================================
# 🚰 SYNC SNAP RECOVERY HTTP ROUTE
# ============================================================
@app.get("/api/bot/status")
async def get_status(userId: str):
    bot = ACTIVE_BOTS.get(userId.strip())
    if not bot:
        bot = DatabaseHandler.load_state(userId.strip())
    
    if bot:
        return {
            "status": bot.get("status", "stopped"), 
            "currentBalance": bot.get("currentBalance", bot.get("balance", 0)),
            "balance": bot.get("balance", 0),
            "unrealizedPnl": bot.get("unrealizedPnl", 0),
            "exposure": bot.get("exposure", 0),
            "currentConfidence": bot.get("currentConfidence", 50),
            "signalsMap": bot.get("signalsMap", {}),
            "equityCurve": bot.get("equityCurve", []),
            "logs": bot.get("logs", []),
            "activePositions": bot.get("positions", []),
            "positions": bot.get("positions", []),
            "startedAt": bot.get("startedAt"),
            "config": bot.get("config"),
            "candles": bot.get("candles", []),
            "trade_history": bot.get('trade_history', []),
            "tradeMarkers": bot.get('trade_history', []),
            "aiRegimeTitle": bot.get("aiRegimeTitle", "Mean-Reverting Consolidation"),
            "aiRegimeDesc": bot.get("aiRegimeDesc", "Sideways Range"),
            "aiDeployedGear": bot.get("aiDeployedGear", "Syncing Core Strategy Modules Matrix...")
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
