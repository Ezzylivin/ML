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
from fastapi.exceptions import RequestValidationError
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
    cache_key = f"{symbol}_{timeframe}"
    if cache_key not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[cache_key] = StackingPredictor(symbol=symbol, timeframe=timeframe)
        logger.info(f"🧠 Cached predictor for {cache_key}")
    return _PREDICTOR_CACHE[cache_key]


# ==========================================
# 🗄️ DATABASE HANDLER
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
                    "config":        json.loads(row['config']),
                    "balance":       row['balance'],
                    "positions":     json.loads(row['positions']),
                    "trade_history": json.loads(row['trade_history']),
                    "equityCurve":   json.loads(row['equity_curve']),
                    "logs":          json.loads(row['logs']),
                    "status":        "stopped"
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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

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
    end_ts   = int(pd.to_datetime(end_str).timestamp()   * 1000)
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
                new_batch = await exchange.fetch_ohlcv(fetch_symbol, timeframe,
                                                       since=current_since, limit=300)
                if not new_batch:
                    break
                all_new_candles.extend(new_batch)
                current_since = new_batch[-1][0] + 1
            await exchange.close()
        except Exception as e:
            print(f"⚠️ Coinbase Fetch Error: {e}")
            await exchange.close()
        if all_new_candles:
            new_df = pd.DataFrame(all_new_candles,
                                  columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
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
        except:
            return "[----------] 0%"

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
                    atr   = ta.atr(df['high'], df['low'], df['close']).iloc[-1]
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
            # FIX: use helper so hybridMode is honoured when comboConfig is absent/stale
            rule = _get_combo_rule(config)
            if not pending:
                return f"🔍 TARGETS ({rule}): Scanning Setup..."
            return f"🔍 TARGETS ({rule}): " + " | ".join(pending[:2])
        except Exception:
            return "🔍 Scanning Market Conditions..."


# ============================================================
# 🔮 PREDICTIVE REGIME OPTIMIZER
# ============================================================
class PredictiveRegimeOptimizer:
    REGIME_DEBOUNCE_TICKS = 5

    @staticmethod
    def get_regime_key(ai_score: float, current_adx: float) -> str:
        if ai_score >= 0.68 or (ai_score > 0.55 and current_adx > 30):
            return "trend"
        elif ai_score <= 0.32 or (ai_score < 0.45 and current_adx > 30):
            return "bear"
        else:
            return "range"

    @staticmethod
    def get_strategies_for_regime(regime: str) -> List[Dict[str, Any]]:
        if regime == "trend":
            return [
                {"code": "supertrend",    "params": {"st_atr": 10, "st_factor": 3.0}},
                {"code": "pa_breakout",   "params": {"lookback": 20, "buffer": 0.01}},
                {"code": "ema_cloud",     "params": {"fast_ema": 9, "slow_ema": 21}},
                {"code": "sma_crossover", "params": {"fast_sma": 20, "slow_sma": 100}}
            ]
        elif regime == "bear":
            return [
                {"code": "supertrend",     "params": {"st_atr": 10, "st_factor": 2.5}},
                {"code": "atr_breakout",   "params": {"atr_length": 14, "multiplier": 1.5}},
                {"code": "macd_crossover", "params": {"fast": 12, "slow": 26, "signal": 9}},
                {"code": "vol_profile",    "params": {"vol_ma": 20, "threshold": 1.2}}
            ]
        else:
            return [
                {"code": "bb_fade",       "params": {"bb_period": 20, "bb_std": 2.0}},
                {"code": "rsi_threshold", "params": {"rsi_length": 14, "oversold": 30, "overbought": 70}},
                {"code": "stoch",         "params": {"k_period": 14, "d_period": 3}}
            ]

    @staticmethod
    def dynamically_tune_strategies(bot_config: Dict[str, Any], ai_score: float,
                                     current_adx: float,
                                     locked_regime: Optional[str] = None) -> List[Dict[str, Any]]:
        # FIX: use helper so hybridMode is honoured when comboConfig is absent/stale
        if _get_combo_rule(bot_config) == "AND":
            return bot_config.get("strategies", [])
        regime = locked_regime or PredictiveRegimeOptimizer.get_regime_key(ai_score, current_adx)
        return PredictiveRegimeOptimizer.get_strategies_for_regime(regime)


# ==========================================
# 🧠 NEURAL MACHINE LEARNING PREDICTOR
# ==========================================
class NeuralPredictor:
    @staticmethod
    def get_prediction(model_id: str, df: pd.DataFrame, symbol: str = "BTC-USD") -> float:
        """
        Output contract: returns P(next candle move is net-positive / long).
        Expected range [0.0, 1.0] — 1.0 = strong bull, 0.0 = strong bear, 0.5 = neutral.
        """
        try:
            council = get_cached_predictor(symbol=symbol, timeframe="1h")
            if hasattr(council, 'get_prediction_score'):
                prediction = council.get_prediction_score(df)
            else:
                prediction = council.predict_direction(df)
            return float(prediction)
        except Exception as e:
            logger.error(f"🧠 Council Predictor Error: {e}")
            return 0.5


# ============================================================
# 🔧 COMBO RULE HELPER
# ============================================================
def _get_combo_rule(config: Dict[str, Any]) -> str:
    """
    Read the combination rule with a two-key fallback chain so a stale or
    missing comboConfig never silently forces AND mode on a user who chose OR.

    Priority:
      1. config["comboConfig"]["combinationRule"]  — set explicitly by handleConfirmStart
      2. config["hybridMode"]                      — top-level spread from formConfig
      3. "OR"                                      — safe default
    """
    from_combo = config.get("comboConfig", {}).get("combinationRule")
    if from_combo:
        return from_combo
    from_hybrid = config.get("hybridMode")
    if from_hybrid:
        return from_hybrid
    return "OR"


# ==========================================
# PACKET FORMATTING LAYER
# ==========================================
async def process_data_packet(df: pd.DataFrame, strategies: list) -> list:
    df, feats = apply_mega_features(df)
    keys = ['bb_lower', 'bb_upper', 'ema_fast', 'ema_slow', 'sma_fast', 'sma_slow',
            'supertrend', 'rsi', 'macd', 'macd_signal', 'stoch_k', 'stoch_d',
            'atr_upper', 'atr_lower', 'pa_high', 'pa_low', 'vol_ma']
    candles_to_send = []
    for _, row in df.tail(100).iterrows():
        ts = int(row['time']) if 'time' in row else int(row.name.timestamp())
        c_obj = {"time": ts, "open": row['open'], "high": row['high'],
                 "low": row['low'], "close": row['close']}
        for k in keys:
            if k in row and not pd.isna(row[k]):
                c_obj[k] = round(float(row[k]), 2)
        candles_to_send.append(c_obj)
    return candles_to_send


# ============================================================
# 🔧 STRATEGY BRAIN
# ============================================================
class StrategyBrain:
    @staticmethod
    def calculate_signals(df_raw: pd.DataFrame, df_ai: pd.DataFrame,
                          config: Dict[str, Any], l_thresh: float, s_thresh: float,
                          symbol: str = "BTC-USD",
                          precomputed_conf: Optional[float] = None,
                          live_price: Optional[float] = None):
        active_thoughts, votes = [], 0
        signals_map   = {}
        strategies    = config.get('strategies', [])

        current_price = live_price if live_price is not None else float(df_raw['close'].iloc[-1])

        ema20  = ta.ema(df_raw['close'], length=20).iloc[-1]
        ema50  = ta.ema(df_raw['close'], length=50).iloc[-1]
        ema200 = ta.ema(df_raw['close'], length=200).iloc[-1]
        bb     = ta.bbands(df_raw['close'], length=20, std=2.0)
        lower, mid, upper = bb.iloc[-1, 0], bb.iloc[-1, 1], bb.iloc[-1, 2]
        pr = int((current_price - lower) / (upper - lower) * 100)
        weighted_votes = 0.0

        for strat in strategies:
            code = strat.get('code')
            p    = strat.get('params', {})
            raw_vote   = 0
            confidence = 0.5
            try:
                if code == "rsi_threshold":
                    digits = int(p.get('rsi_length', 14))
                    rsi    = ta.rsi(df_raw['close'], length=digits).iloc[-1]
                    dist   = min(abs(rsi - 30), abs(rsi - 70))
                    confidence = max(0.1, min(1.0, 1.0 - (dist / 40)))
                    if rsi < p.get('oversold', 30):    raw_vote = 1;  active_thoughts.append("RSI Low")
                    elif rsi > p.get('overbought', 70): raw_vote = -1; active_thoughts.append("RSI High")
                elif code == "sma_crossover":
                    f   = ta.sma(df_raw['close'], length=int(p.get('fast_sma', 50))).iloc[-1]
                    s   = ta.sma(df_raw['close'], length=int(p.get('slow_sma', 200))).iloc[-1]
                    gap = abs(f - s) / s
                    confidence = 1.0 if f > s else max(0.1, min(0.95, 1.0 - (gap * 50)))
                    raw_vote   = 1 if f > s else -1
                elif code == "macd_crossover":
                    macd      = ta.macd(df_raw['close'], fast=int(p.get('fast', 12))).iloc[-1]
                    norm_hist = abs(macd[1]) / (current_price * 0.0005)
                    confidence = max(0.1, min(1.0, norm_hist))
                    raw_vote   = 1 if macd[0] > macd[2] else -1
                elif code == "supertrend":
                    st_data = ta.supertrend(df_raw['high'], df_raw['low'], df_raw['close']).iloc[-1]
                    dist    = abs(current_price - st_data[0]) / current_price
                    confidence = 1.0 if st_data[1] == 1 else max(0.1, min(0.95, 1.0 - (dist * 20)))
                    raw_vote   = 1 if st_data[1] == 1 else -1
                elif code == "bb_fade":
                    confidence = max(0.1, min(1.0, pr / 100.0))
                    if current_price < lower:   raw_vote = 1
                    elif current_price > upper: raw_vote = -1
                elif code == "atr_breakout":
                    atr    = ta.atr(df_raw['high'], df_raw['low'], df_raw['close']).iloc[-1]
                    target = ema20 + (atr * float(p.get('multiplier', 1.5)))
                    confidence = max(0.1, min(1.0, current_price / target))
                    raw_vote   = 1 if current_price > target else -1
                elif code == "pa_breakout":
                    lb      = int(p.get('lookback', 20))
                    high_lb = df_raw['high'].tail(lb).max()
                    confidence = max(0.1, min(1.0, current_price / high_lb))
                    raw_vote   = 1 if current_price >= high_lb else -1
                elif code == "vol_profile":
                    v_ma  = ta.sma(df_raw['volume'], length=int(p.get('vol_ma', 20))).iloc[-1]
                    ratio = df_raw['volume'].iloc[-1] / (v_ma * float(p.get('threshold', 1.5)))
                    confidence = max(0.1, min(1.0, ratio))
                    raw_vote   = (1 if current_price > mid else -1) if ratio >= 1.0 else 0
                elif code == "stoch":
                    k    = ta.stoch(df_raw['high'], df_raw['low'], df_raw['close']).iloc[-1][0]
                    dist = min(abs(k - 20), abs(k - 80))
                    confidence = max(0.1, min(1.0, 1.0 - (dist / 40)))
                    if k < 20:   raw_vote = 1
                    elif k > 80: raw_vote = -1
                elif code == "ema_cloud":
                    f_ema = ta.ema(df_raw['close'], length=int(p.get('fast_ema', 9))).iloc[-1]
                    s_ema = ta.ema(df_raw['close'], length=int(p.get('slow_ema', 21))).iloc[-1]
                    gap   = abs(f_ema - s_ema) / s_ema
                    confidence = 1.0 if f_ema > s_ema else max(0.1, min(0.95, 1.0 - (gap * 100)))
                    raw_vote   = 1 if f_ema > s_ema else -1
                else:
                    confidence = 0.5
                signals_map[code]  = confidence
                votes             += raw_vote
                weighted_votes    += raw_vote * confidence
            except Exception:
                signals_map[code] = 0.0

        is_short = current_price < ema200
        ui_limit = (float(config.get('mlThresholdShort', 0.55)) if is_short
                    else float(config.get('mlThresholdLong', 0.55)))

        conf = precomputed_conf if precomputed_conf is not None else NeuralPredictor.get_prediction(
            config.get('mlModel', 'stacking'), df_ai, symbol=symbol)

        if config.get('mlMode') == 'off':
            gate_passed = True
        else:
            gate_passed = (conf >= ui_limit) if not is_short else ((1.0 - conf) >= ui_limit)

        # FIX: use helper so hybridMode is honoured when comboConfig is absent/stale
        rule      = _get_combo_rule(config)
        final_sig = 0
        if gate_passed:
            if rule == "AND":
                if votes >= len(strategies) and weighted_votes > 0:    final_sig = 1
                elif votes <= -len(strategies) and weighted_votes < 0: final_sig = -1
            else:
                min_weighted = float(config.get('minWeightedSignal', 0.3))
                if votes > 0  and weighted_votes >= min_weighted:  final_sig = 1
                elif votes < 0 and weighted_votes <= -min_weighted: final_sig = -1

        return final_sig, active_thoughts, {}, conf, signals_map


# ============================================================================
# 🚀 CORE ENGINE HEARTBEAT
# ============================================================================
async def live_neural_heartbeat(user_id: str):
    signal_has_reset = True
    last_log         = 0
    last_ui_update   = 0

    if user_id in ACTIVE_BOTS:
        if "equityCurve" not in ACTIVE_BOTS[user_id]:
            ACTIVE_BOTS[user_id]["equityCurve"] = [
                {"time": datetime.now().isoformat(),
                 "balance": ACTIVE_BOTS[user_id]["balance"], "confidence": 50}]
        if "logs" not in ACTIVE_BOTS[user_id]:
            ACTIVE_BOTS[user_id]["logs"] = []

    try:
        while user_id in ACTIVE_BOTS and ACTIVE_BOTS[user_id]["status"] == "running":
            bot           = ACTIVE_BOTS[user_id]
            config        = bot.get('config', {})
            params        = config.get('params', {})
            strategies    = config.get('strategies', [])
            symbol        = config['symbol'].replace('-', '/')
            ticker_symbol = config['symbol']

            max_p         = min(5, int(config.get('maxPyramiding', 5)))
            start_capital = float(config.get('capitalAllocation', config.get('initialBalance', 200.0)))
            api_keys      = bot.get('api_keys') or config.get('api_keys', {})
            trading_mode  = config.get('trading_mode', 'paper').lower()
            use_margin    = config.get('enable_shorting', False)
            leverage_val  = float(config.get('leverage', 1.0))

            target_exchange = "kraken" if use_margin else "coinbase"
            has_valid_keys  = bool(api_keys.get('krakenKey') if use_margin else api_keys.get('apiKey'))
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

                # ── TIER 1: MARKET FEED ──────────────────────────────────────
                ohlcv_raw = await fetch_live_candles_ccxt(
                    config['symbol'], config.get('timeframe', '1h'), 500,
                    exchange_id=target_exchange)
                if not ohlcv_raw:
                    await emit_log(user_id, f"⚠️ {target_exchange.upper()} Feed Unstable - Retrying...")
                    await asyncio.sleep(5); continue

                exchange_class = getattr(ccxt, target_exchange)
                async with exchange_class({'enableRateLimit': True}) as ex:
                    ticker        = await ex.fetch_ticker(ticker_symbol.replace('-', '/'))
                    current_price = float(ticker['last'])

                ohlcv_raw[-1]['close'] = current_price
                df_raw            = pd.DataFrame(ohlcv_raw)
                df_closed_history = df_raw.iloc[:-1].copy()
                df_ai, _          = await asyncio.to_thread(apply_mega_features, df_closed_history)

                current_ema200 = float(ta.ema(df_raw['close'], length=200).iloc[-1])
                current_atr    = float(ta.atr(df_raw['high'], df_raw['low'], df_raw['close'], length=14).iloc[-1])
                atr_tp_mult    = float(params.get('atr_tp_mult', config.get('atrTpMultiplier', 3.0)))
                atr_sl_mult    = float(params.get('atr_sl_mult', config.get('atrSlMultiplier', 1.5)))

                try:
                    current_adx = float(ta.adx(df_raw['high'], df_raw['low'], df_raw['close'], length=14).iloc[-1, 0])
                except Exception:
                    current_adx = 25.0

                try:
                    vol_ma    = float(ta.sma(df_raw['volume'], length=20).iloc[-1])
                    vol_ratio = df_raw['volume'].iloc[-2] / vol_ma if vol_ma > 0 else 1.0
                except Exception:
                    vol_ratio = 1.0

                # ── TIER 2: AI COUNCIL ────────────────────────────────────────
                conf_score = NeuralPredictor.get_prediction(
                    config.get('mlModel', 'stacking'), df_ai, symbol=ticker_symbol)

                desired_regime = PredictiveRegimeOptimizer.get_regime_key(conf_score, current_adx)
                if 'active_regime' not in bot:
                    bot['active_regime']        = desired_regime
                    bot['regime_pending_ticks'] = 0
                elif desired_regime != bot['active_regime']:
                    bot['regime_pending_ticks'] = bot.get('regime_pending_ticks', 0) + 1
                    if bot['regime_pending_ticks'] >= PredictiveRegimeOptimizer.REGIME_DEBOUNCE_TICKS:
                        prev_regime             = bot['active_regime']
                        bot['active_regime']    = desired_regime
                        bot['regime_pending_ticks'] = 0
                        await emit_log(user_id, f"🔄 REGIME SHIFT: {prev_regime.upper()} → {desired_regime.upper()}")
                else:
                    bot['regime_pending_ticks'] = 0

                active_strategies = (
                    PredictiveRegimeOptimizer.dynamically_tune_strategies(
                        config, conf_score, current_adx,
                        locked_regime=bot['active_regime'])
                    if config.get('mlMode') == 'on'
                    else config.get('strategies', strategies)
                )
                temp_config = {**config, 'strategies': active_strategies}

                sig, thoughts, nums, score, signals_map = StrategyBrain.calculate_signals(
                    df_closed_history, df_ai, temp_config, 0.5, 0.5,
                    symbol=ticker_symbol,
                    precomputed_conf=conf_score,
                    live_price=current_price)

                # ── TIER 3: PROP-MAPPING ──────────────────────────────────────
                sentiment = "STRONG BUY" if score > 0.85 else "BUY" if score > 0.70 else "NEUTRAL"
                if score < 0.20:   sentiment = "STRONG SELL"
                elif score < 0.35: sentiment = "SELL"

                last_pos = bot['positions'][-1] if bot['positions'] else None
                if last_pos:
                    pos_is_profitable = (
                        (last_pos['type'] == 'long'  and current_price > last_pos['entry']) or
                        (last_pos['type'] == 'short' and current_price < last_pos['entry'])
                    )
                    climb_satisfied = score >= last_pos.get('entry_conf', 0) + 0.10 and pos_is_profitable
                else:
                    climb_satisfied = True

                is_short_allowed = (sig != -1) or config.get('enable_shorting', False)

                upnl = sum([
                    (current_price - p['entry']) * p['size'] if p['type'] == 'long'
                    else (p['entry'] - current_price) * p['size']
                    for p in bot['positions']
                ])
                current_equity = bot['balance'] + upnl

                active_summary = ""
                if bot['positions']:
                    p        = bot['positions'][-1]
                    stop_pct = round((abs(current_price - p['tsl']) / current_price) * 100, 2)
                    buffer_bar = DiagnosticLayer.render_progress(stop_pct, 2.0, reverse=True)
                    active_summary = f"⚡ ACTIVE: {len(bot['positions'])} POS (${upnl:,.2f}) | TSL {buffer_bar} {stop_pct}% | "

                # ── TIER 4: RISK GATEWAY ──────────────────────────────────────
                ui_limit = (float(config.get('mlThresholdShort', 0.55))
                            if current_price < current_ema200
                            else float(config.get('mlThresholdLong', 0.55)))
                waiting_msg = DiagnosticLayer.get_pending_conditions(df_raw, temp_config, score, ui_limit)

                base_threshold = ui_limit
                if current_adx > 30:   adaptive_threshold = max(0.45, base_threshold - 0.05)
                elif current_adx < 20: adaptive_threshold = min(0.80, base_threshold + 0.05)
                else:                  adaptive_threshold = base_threshold

                recent_trades      = [t for t in bot.get('trade_history', []) if t.get('type') == 'exit'][-5:]
                consecutive_losses = 0
                for t in reversed(recent_trades):
                    if float(t.get('pnl', 0)) < 0: consecutive_losses += 1
                    else: break

                if consecutive_losses >= 3:
                    adaptive_threshold = min(0.80, adaptive_threshold + (consecutive_losses - 2) * 0.03)
                    if consecutive_losses == 3 and (datetime.now().timestamp() - last_log) >= 15:
                        await emit_log(user_id, f"⚠️ COLD STREAK: {consecutive_losses} losses — raising bar to {int(adaptive_threshold*100)}%")

                adaptive_gate = score >= adaptive_threshold if config.get('mlMode') == 'on' else True

                atr_pct    = (current_atr / current_price) * 100
                min_atr_pct = float(config.get('minAtrPct', 0.3))
                max_atr_pct = float(config.get('maxAtrPct', 3.0))
                is_volatility_safe = min_atr_pct <= atr_pct <= max_atr_pct

                is_circuit_breaker_tripped = (
                    bot['balance'] <= start_capital * (1.0 - float(config.get('maxDailyLoss', 5.0)) / 100.0))

                if len(bot['positions']) == 0 and sig == 0: signal_has_reset = True
                elif len(bot['positions']) > 0:             signal_has_reset = False
                market_gate_passed = signal_has_reset if len(bot['positions']) == 0 else True

                try:    adx_trending = current_adx >= float(config.get('minAdx', 20.0))
                except: adx_trending = True

                volume_confirmed = vol_ratio >= float(config.get('minVolRatio', 0.8))
                is_trend_aligned = (
                    (sig == 1 and current_price > current_ema200) or
                    (sig == -1 and current_price < current_ema200) or
                    (sig == 0)
                )

                # ── DYNAMIC TIME GATE ────────────────────────────────────────
                # Standard cooldown is 10 minutes.  The gate can be bypassed
                # early once a 2-minute hard floor has elapsed AND the AI
                # confidence has both improved since the last entry AND is
                # trending upward over recent ticks — meaning the market is
                # genuinely improving, not just bouncing off a local noise peak.
                #
                # Hard floor  → prevents re-hammering on fast reversals
                # Normal gate → 10-minute default inter-trade spacing
                # Surge path  → 2-min floor + conf ≥8 pts above last entry
                #                + recent 3-tick trend rising ≥5 pts
                MIN_COOLDOWN_SECS  = 2 * 60
                STANDARD_GATE_SECS = float(config.get('minMinutesBetweenTrades', 10)) * 60

                time_since_last = 0.0
                if bot.get('last_trade_time'):
                    try:
                        time_since_last = (
                            current_time - datetime.fromisoformat(bot['last_trade_time'])
                        ).total_seconds()
                    except Exception:
                        time_since_last = STANDARD_GATE_SECS  # treat as expired

                past_floor  = time_since_last >= MIN_COOLDOWN_SECS
                past_normal = time_since_last >= STANDARD_GATE_SECS

                # How much has confidence improved since the last trade entry?
                conf_at_last_entry = float(bot.get('last_trade_conf', 0.0))
                conf_improvement   = score - conf_at_last_entry  # positive = risen

                # Is confidence trending up over the last 6 equity-curve points?
                recent_curve  = bot.get('equityCurve', [])[-6:]
                recent_confs  = [p.get('confidence', 50) / 100.0 for p in recent_curve]
                if len(recent_confs) >= 4:
                    half          = len(recent_confs) // 2
                    earlier_avg   = sum(recent_confs[:half])  / half
                    later_avg     = sum(recent_confs[half:])  / half
                    conf_trending = (later_avg - earlier_avg) >= 0.05
                else:
                    conf_trending = False  # not enough history yet

                # Gate passes when:
                #   A) Normal: standard gap elapsed, OR
                #   B) Surge:  2-min floor cleared + conf rose ≥8 pts + trend up
                time_gate_passed = (
                    not bot.get('last_trade_time') or
                    past_normal or
                    (past_floor and conf_improvement >= 0.08 and conf_trending)
                )
                min_gap_seconds = STANDARD_GATE_SECS  # kept for rejection logger

                all_filters_pass = (
                    is_volatility_safe and not is_circuit_breaker_tripped and
                    is_trend_aligned and adx_trending and volume_confirmed and
                    adaptive_gate and market_gate_passed and is_short_allowed and time_gate_passed
                )

                # ── TIER 5: TELEMETRY + VETO TRACKER ─────────────────────────
                if "vetoed_signals" not in bot:
                    bot["vetoed_signals"] = []

                raw_sig = 0
                votes_t = sum([1 if signals_map.get(s['code'], 0) > 0.5 else -1 for s in active_strategies])
                # FIX: use helper so hybridMode is honoured when comboConfig is absent/stale
                rule    = _get_combo_rule(config)
                if rule == "AND":
                    if votes_t >= len(active_strategies):    raw_sig = 1
                    elif votes_t <= -len(active_strategies): raw_sig = -1
                else:
                    if votes_t > 0:  raw_sig = 1
                    elif votes_t < 0: raw_sig = -1

                if raw_sig != 0 and score < ui_limit:
                    last_veto_ts = (datetime.fromisoformat(bot["vetoed_signals"][-1]["time"]).timestamp()
                                    if bot["vetoed_signals"] else 0)
                    if (current_time.timestamp() - last_veto_ts) > 300:
                        bot["vetoed_signals"].append({
                            "time": current_time.isoformat(), "signal": "Long" if raw_sig == 1 else "Short",
                            "conf_score": round(score, 4), "limit": round(ui_limit, 4), "price": current_price
                        })
                        if len(bot["vetoed_signals"]) > 100: bot["vetoed_signals"].pop(0)

                if len(bot['positions']) < int(config.get('maxPyramiding', 1)):
                    hunting_summary = f"🏹 STALKING LEG {len(bot['positions'])+1}: {int(score*100)}% ({sentiment}) | {waiting_msg}"
                else:
                    hunting_summary = "✅ PYRAMID FULL: Managing Exits"
                targets_str = " | ".join([
                    f"{s['code'].upper()}: {DiagnosticLayer.render_progress(int(signals_map.get(s['code'], 0)*100), 100, reverse=True)}"
                    for s in active_strategies
                ])
                combined_status = f"{active_summary}{hunting_summary} | 🔍 {targets_str}"
                now_ts = datetime.now().timestamp()
                if (now_ts - last_log) >= 15:
                    await emit_log(user_id, combined_status)
                    last_log = now_ts

                # ── TIER 5b: UI BROADCAST ─────────────────────────────────────
                if (now_ts - last_ui_update) >= 10:
                    exposure_pct = (round((sum([p['entry'] * p['size'] for p in bot['positions']])
                                           / current_equity) * 100, 1)
                                    if current_equity > 0 else 0)
                    latest_candles = await process_data_packet(df_raw, active_strategies)

                    if config.get('mlMode') == 'on':
                        active_regime = bot.get('active_regime', 'range')
                        if active_regime == "trend":
                            deployed_gear = "Trend Armor (SuperTrend, PA Breakout, EMA Cloud, SMA Cross)"
                        elif active_regime == "bear":
                            deployed_gear = "Capitulation Armor (SuperTrend, ATR Breakout, MACD Cross, Vol Profile)"
                        else:
                            deployed_gear = "Range Armor (Bollinger Bands, RSI Threshold, Stochastic)"
                    else:
                        deployed_gear = f"Custom Suite ({', '.join([s['code'].upper() for s in active_strategies])})"

                    if len(bot['positions']) >= max_p:
                        regime_title = "Portfolio Full"
                        regime_desc  = f"All slots filled ({len(bot['positions'])}/{max_p}). Managing exits."
                    elif sig != 0 and all_filters_pass:
                        regime_title = "Executing Entry"
                        regime_desc  = f"All checkpoints cleared. Dispatching route at ${current_price:,.2f}."
                    elif sig != 0 and not all_filters_pass:
                        regime_title = "Entry Guarded"
                        regime_desc  = "Defensive logic engaged. Holding until signals optimise."
                    else:
                        regime_title = "Quiet Market Structure"
                        regime_desc  = f"Consolidation at ${current_price:,.2f}. Standing guard."

                    closed_trades = [t for t in bot.get('trade_history', []) if t.get('type') == 'exit']
                    if closed_trades:
                        wins          = len([t for t in closed_trades if float(t.get('pnl', 0)) > 0])
                        win_rate      = round((wins / len(closed_trades)) * 100, 1)
                        gross_profit  = sum(float(t.get('pnl', 0)) for t in closed_trades if float(t.get('pnl', 0)) > 0)
                        gross_loss    = abs(sum(float(t.get('pnl', 0)) for t in closed_trades if float(t.get('pnl', 0)) < 0))
                        profit_factor = (round(gross_profit / gross_loss, 2) if gross_loss > 0
                                         else (round(gross_profit, 2) if gross_profit > 0 else 1.0))
                    else:
                        win_rate, profit_factor = 0.0, 1.0

                    # FIX: compute daily_profit for the MetricCard
                    daily_profit = round(current_equity - start_capital, 2)

                    bot.update({
                        "currentBalance":    round(current_equity, 2),
                        "unrealizedPnl":     round(upnl, 2),
                        "exposure":          exposure_pct,
                        "currentConfidence": int(score * 100),
                        "signalsMap":        signals_map,
                        "candles":           latest_candles,
                        "aiRegimeTitle":     regime_title,
                        "aiRegimeDesc":      regime_desc,
                        "aiDeployedGear":    deployed_gear,
                        "winRate":           win_rate,
                        "profitFactor":      profit_factor,
                        "dailyProfit":       daily_profit,
                    })

                    # FIX: append equity curve point BEFORE emit so the payload
                    # contains the up-to-date curve (previously appended after emit).
                    bot["equityCurve"].append({
                        "time":       datetime.now().isoformat(),
                        "balance":    round(current_equity, 2),
                        "confidence": int(score * 100)
                    })
                    if len(bot["equityCurve"]) > 300:
                        bot["equityCurve"].pop(0)

                    # FIX: diagnostic log so you can confirm emit_status is actually reached.
                    # Remove or set to DEBUG once charts are confirmed working.
                    logger.info(
                        f"📡 emit_status → {user_id} | "
                        f"balance=${bot['currentBalance']:.2f} | "
                        f"conf={bot['currentConfidence']}% | "
                        f"signals={list(signals_map.keys())} | "
                        f"equity_pts={len(bot['equityCurve'])}"
                    )

                    # FIX: emit_status payload now includes equityCurve, dailyProfit,
                    # startedAt, and tradeHistory — previously missing keys that caused
                    # the equity/confidence area charts to stay blank until the frontend
                    # accumulated 2+ events from its own local curve builder.
                    await emit_status(user_id, {
                        "status":            "running",
                        "currentBalance":    bot["currentBalance"],
                        "currentConfidence": bot["currentConfidence"],
                        "signalsMap":        bot["signalsMap"],
                        # FIX: equityCurve was never emitted via socket — charts built
                        # locally were blank until 2+ events arrived (min ~20 s).
                        "equityCurve":       bot["equityCurve"],
                        # FIX: dailyProfit was absent — MetricCard showed 0 on every reconnect.
                        "dailyProfit":       bot["dailyProfit"],
                        # FIX: startedAt was absent — uptime timer reset on reconnect.
                        "startedAt":         bot.get("startedAt"),
                        "exposure":          bot["exposure"],
                        "activePositions":   bot['positions'],
                        "unrealizedPnl":     bot["unrealizedPnl"],
                        # FIX: send under both keys so the frontend fallback chain always hits.
                        "tradeHistory":      bot['trade_history'],
                        "tradeMarkers":      bot['trade_history'],
                        "candles":           bot["candles"],
                        "initialCapital":    start_capital,
                        "aiRegimeTitle":     bot["aiRegimeTitle"],
                        "aiRegimeDesc":      bot["aiRegimeDesc"],
                        "aiDeployedGear":    bot["aiDeployedGear"],
                        "winRate":           bot["winRate"],
                        "profitFactor":      bot["profitFactor"],
                    })

                    logger.info(f"✅ emit_status dispatched for {user_id}")
                    last_ui_update = now_ts

                # ── TIER 6: REJECTION LOGGER ──────────────────────────────────
                if sig != 0 and not all_filters_pass:
                    reasons = []
                    if not time_gate_passed:
                        elapsed_min    = time_since_last / 60
                        normal_min     = STANDARD_GATE_SECS / 60
                        floor_min      = MIN_COOLDOWN_SECS  / 60
                        surge_eligible = past_floor and conf_improvement >= 0.08
                        if not past_floor:
                            reasons.append(
                                f"hard floor ({elapsed_min:.0f}/{floor_min:.0f} min)"
                            )
                        elif not surge_eligible:
                            reasons.append(
                                f"time gate ({elapsed_min:.0f}/{normal_min:.0f} min, "
                                f"conf Δ{int(conf_improvement*100):+}% — need +8% surge to bypass)"
                            )
                        else:
                            # Floor cleared and conf rose, but trend not yet up
                            reasons.append(
                                f"time gate ({elapsed_min:.0f}/{normal_min:.0f} min, "
                                f"conf Δ{int(conf_improvement*100):+}% but trend still flat)"
                            )
                    if not is_short_allowed: reasons.append("Spot mode cannot short sell")
                    if not is_volatility_safe:
                        if atr_pct < min_atr_pct: reasons.append(f"ATR {atr_pct:.2f}% too flat (min {min_atr_pct}%)")
                        else:                     reasons.append(f"ATR {atr_pct:.1f}% too high (max {max_atr_pct}%)")
                    if is_circuit_breaker_tripped: reasons.append("Daily drawdown limit hit")
                    if not is_trend_aligned:       reasons.append("Trend misaligned with 200 EMA")
                    if not adx_trending:           reasons.append(f"ADX {current_adx:.0f} below min {float(config.get('minAdx', 20.0)):.0f}")
                    if not volume_confirmed:       reasons.append(f"Volume {vol_ratio:.2f}x below min {float(config.get('minVolRatio', 0.8)):.1f}x")
                    if not adaptive_gate:          reasons.append(f"AI {int(score*100)}% below threshold {int(adaptive_threshold*100)}%")
                    if not market_gate_passed:     reasons.append("Waiting for signal reset")
                    if not reasons: reasons.append("Pre-flight check failed")
                    await emit_log(user_id, f"⛔ ENTRY BLOCKED: {', '.join(reasons)}")

                # ── TIER 7: ENTRY DISPATCH ────────────────────────────────────
                if len(bot['positions']) < max_p and all_filters_pass:
                    if (sig == 1 and climb_satisfied) or (sig == -1 and climb_satisfied):
                        trade_type     = "long" if sig == 1 else "short"
                        raw_risk       = config.get('riskPercentage') or config.get('risk_percentage') or 1.0
                        size_in_fiat   = current_equity * (float(raw_risk) / 100.0 / max_p)
                        size_in_crypto = size_in_fiat / current_price
                        safe_size      = float(f"{size_in_crypto:.6f}")
                        actual_entry_price = current_price

                        if is_live_trading:
                            try:
                                exchange_class = getattr(ccxt, target_exchange)
                                async with exchange_class({
                                    'apiKey': api_keys.get('krakenKey', api_keys.get('apiKey')),
                                    'secret': api_keys.get('krakenSecret', api_keys.get('secret')),
                                    'enableRateLimit': True
                                }) as user_exchange:
                                    side         = "buy" if sig == 1 else "sell"
                                    order_params = {'leverage': leverage_val} if use_margin else {}
                                    await emit_log(user_id, f"🔗 ROUTING {side.upper()} TO {target_exchange.upper()}...")
                                    await user_exchange.load_markets()
                                    formatted_size = float(user_exchange.amount_to_precision(symbol, size_in_crypto))
                                    if formatted_size <= 0: raise Exception("Order size too small")
                                    order = await user_exchange.create_market_order(symbol, side, formatted_size, params=order_params)
                                    actual_entry_price = order.get('average') or order.get('price') or current_price
                                    safe_size          = order.get('filled') or formatted_size
                            except Exception as ex_err:
                                await emit_log(user_id, f"❌ {target_exchange.upper()} ORDER FAILED: {str(ex_err)}")
                                await asyncio.sleep(5); continue
                        else:
                            bot['balance'] -= (safe_size * actual_entry_price) * fee_rate

                        tp_price_calc = (actual_entry_price + current_atr * atr_tp_mult if sig == 1
                                         else actual_entry_price - current_atr * atr_tp_mult)
                        sl_price_calc = (actual_entry_price - current_atr * atr_sl_mult if sig == 1
                                         else actual_entry_price + current_atr * atr_sl_mult)
                        partial_tp    = (actual_entry_price + current_atr * 1.5 if sig == 1
                                         else actual_entry_price - current_atr * 1.5)

                        bot['positions'].append({
                            "symbol":             symbol,
                            "type":               trade_type,
                            "entry":              actual_entry_price,
                            "size":               safe_size,
                            "time":               current_time.isoformat(),
                            "entry_conf":         score,
                            "tp":                 tp_price_calc,
                            "sl":                 sl_price_calc,
                            "tsl":                sl_price_calc,
                            "partial_tp":         partial_tp,
                            "partial_taken":      False,
                            "atr_at_entry":       round(current_atr, 4),
                            "adx_at_entry":       round(current_adx, 1),
                            "vol_ratio_at_entry": round(vol_ratio, 2),
                        })
                        bot['last_trade_time'] = current_time.isoformat()
                        # Store the confidence score at entry so the dynamic time gate
                        # can measure improvement on the next trade opportunity.
                        bot['last_trade_conf'] = score
                        rr = round(atr_tp_mult / atr_sl_mult, 1)
                        await emit_log(user_id,
                                       f"🚀 ENTERED {trade_type.upper()} @ ${actual_entry_price:,.2f} | "
                                       f"TP ${tp_price_calc:,.2f} | SL ${sl_price_calc:,.2f} | "
                                       f"R:R {rr} | ADX {current_adx:.0f}")
                        DatabaseHandler.save_state(user_id, bot)

                # ── TIER 8: EXIT MONITORING ───────────────────────────────────
                for pos in bot['positions'][:]:
                    closed      = False
                    exit_reason = ""
                    pos_atr     = pos.get('atr_at_entry', current_atr)
                    trail_dist  = pos_atr * atr_sl_mult

                    if config.get('enablePartialExit', False) and not pos.get('partial_taken', False):

                        if pos['type'] == 'long' and current_price >= pos.get('partial_tp', float('inf')):
                            half_size   = pos['size'] / 2.0
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
                                    await emit_log(user_id, f"⚠️ Partial long exit failed: {e}")
                            else:
                                net_partial    = partial_pnl - (half_size * current_price * fee_rate)
                                bot['balance'] += net_partial
                                partial_pnl    = net_partial
                            pos['size']          = half_size
                            pos['partial_taken'] = True
                            pos['tsl']           = max(pos['tsl'], pos['entry'])
                            await emit_log(user_id, f"💎 PARTIAL LONG EXIT: 50% at ${current_price:,.2f} | Locked ${partial_pnl:.2f} | Stop → Breakeven")
                            bot['trade_history'].append({"type": "partial_exit", "side": pos['type'], "price": current_price, "pnl": round(partial_pnl, 2), "time": int(time.time() * 1000)})
                            DatabaseHandler.save_state(user_id, bot)

                        elif pos['type'] == 'short' and current_price <= pos.get('partial_tp', float('-inf')):
                            half_size   = pos['size'] / 2.0
                            partial_pnl = (pos['entry'] - current_price) * half_size
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
                                        order = await user_exchange.create_market_order(symbol, "buy", fmt_size)
                                        partial_pnl = (pos['entry'] - (order.get('average') or current_price)) * half_size
                                except Exception as e:
                                    await emit_log(user_id, f"⚠️ Partial short exit failed: {e}")
                            else:
                                net_partial    = partial_pnl - (half_size * current_price * fee_rate)
                                bot['balance'] += net_partial
                                partial_pnl    = net_partial
                            pos['size']          = half_size
                            pos['partial_taken'] = True
                            pos['tsl']           = min(pos['tsl'], pos['entry'])
                            await emit_log(user_id, f"💎 PARTIAL SHORT EXIT: 50% at ${current_price:,.2f} | Locked ${partial_pnl:.2f} | Stop → Breakeven")
                            bot['trade_history'].append({"type": "partial_exit", "side": pos['type'], "price": current_price, "pnl": round(partial_pnl, 2), "time": int(time.time() * 1000)})
                            DatabaseHandler.save_state(user_id, bot)

                    # Trailing stop ratchet
                    if pos['type'] == 'long':
                        new_tsl = current_price - trail_dist
                        if new_tsl > pos['tsl']: pos['tsl'] = new_tsl
                        if current_price >= pos['tp']:    closed = True; exit_reason = "Take Profit"
                        elif current_price <= pos['tsl']: closed = True; exit_reason = "Trailing Stop"
                    else:
                        new_tsl = current_price + trail_dist
                        if new_tsl < pos['tsl']: pos['tsl'] = new_tsl
                        if current_price <= pos['tp']:    closed = True; exit_reason = "Take Profit"
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
                                    close_side   = "sell" if pos['type'] == 'long' else "buy"
                                    order_params = {'leverage': leverage_val} if use_margin else {}
                                    await user_exchange.load_markets()
                                    fmt_size = float(user_exchange.amount_to_precision(symbol, pos['size']))
                                    order = await user_exchange.create_market_order(symbol, close_side, fmt_size, params=order_params)
                                    actual_close_price = order.get('average') or order.get('price') or current_price
                                    balance_data = await user_exchange.fetch_balance()
                                    real_fiat = balance_data.get('USD', {}).get('free') or balance_data.get('USDC', {}).get('free')
                                    if real_fiat: bot['balance'] = float(real_fiat)
                            except Exception as ex_err:
                                await emit_log(user_id, f"❌ EXIT FAILED: {str(ex_err)}")
                                continue

                        gross_pnl = ((actual_close_price - pos['entry']) * pos['size'] if pos['type'] == 'long'
                                     else (pos['entry'] - actual_close_price) * pos['size'])
                        net_pnl   = gross_pnl - ((pos['size'] * actual_close_price) * fee_rate)
                        if not is_live_trading: bot['balance'] += net_pnl

                        bot['positions'].remove(pos)
                        bot['trade_history'].append({
                            "type":   "exit",
                            "side":   pos['type'],
                            "entry":  pos['entry'],
                            "price":  actual_close_price,
                            "pnl":    round(net_pnl, 2),
                            "reason": exit_reason,
                            "time":   int(time.time() * 1000)
                        })
                        await emit_log(user_id,
                                       f"💰 {exit_reason}: CLOSED {pos['type'].upper()} @ "
                                       f"${actual_close_price:,.2f} | Net PnL: ${round(net_pnl, 2):+}")
                        DatabaseHandler.save_state(user_id, bot)

                # Periodic DB save
                if (datetime.now().timestamp() - last_log) >= 60:
                    DatabaseHandler.save_state(user_id, bot)

                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"❌ WS Stream Error: {e}", exc_info=True)
                await asyncio.sleep(5)
    finally:
        logger.info(f"🔌 Heartbeat loop terminated for {user_id}")


# =============================================================
# ENDPOINTS
# =============================================================
async def fetch_live_candles_ccxt(symbol: str, timeframe: str, limit: int,
                                   exchange_id: str = "coinbase"):
    exchange_class = getattr(ccxt, exchange_id)
    async with exchange_class({'enableRateLimit': True}) as ex:
        try:
            fetch_symbol = symbol.replace('-', '/')
            ohlcv = await ex.fetch_ohlcv(fetch_symbol, timeframe, limit=limit)
            return [{"time": int(c[0]/1000), "open": c[1], "high": c[2], "low": c[3],
                     "close": c[4], "volume": c[5] if len(c) > 5 else 0} for c in ohlcv]
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
    user_id    = data.userId.strip()
    raw_cap    = (data.config.get("capitalAllocation") or data.config.get("capital_allocation")
                  or data.config.get("initialBalance"))
    ui_capital = float(raw_cap) if raw_cap else 200.0

    if user_id in TASK_REGISTRY:
        try: TASK_REGISTRY[user_id].cancel(); logger.info(f"♻️ Registry: Cleaned old loop for {user_id}")
        except Exception as e: logger.error(f"⚠️ Registry Cleanup Error: {e}")

    initial_ohlcv = await fetch_live_candles_ccxt(
        data.config['symbol'], data.config.get('timeframe', '1h'), 350)
    processed_candles = []
    if initial_ohlcv:
        df_init = pd.DataFrame(initial_ohlcv)
        processed_candles = await process_data_packet(df_init, data.config.get('strategies', []))

    saved_state = DatabaseHandler.load_state(user_id)
    if saved_state:
        ACTIVE_BOTS[user_id] = saved_state
        ACTIVE_BOTS[user_id].update({
            "status": "running", "config": data.config, "balance": ui_capital,
            "startedAt": datetime.now(timezone.utc).isoformat(),
            "trade_history": [], "logs": [], "positions": [],
            "equityCurve": [{"time": datetime.now().isoformat(), "balance": ui_capital, "confidence": 50}],
        })
        await emit_log(user_id, f"♻️ SESSION INITIALIZED: Fresh slate at ${ui_capital}")
    else:
        ACTIVE_BOTS[user_id] = {
            "status": "running", "config": data.config, "balance": ui_capital,
            "positions": [], "trade_history": [], "logs": [],
            "equityCurve": [{"time": datetime.now().isoformat(), "balance": ui_capital, "confidence": 50}],
            "startedAt": datetime.now(timezone.utc).isoformat(),
        }
        await emit_log(user_id, f"🚀 Engine Started. Portfolio: ${ui_capital}")

    DatabaseHandler.save_state(user_id, ACTIVE_BOTS[user_id])
    await emit_status(user_id, {
        "status":     "running",
        "currentBalance": ACTIVE_BOTS[user_id]["balance"],
        "candles":    processed_candles,
        "startedAt":  ACTIVE_BOTS[user_id]["startedAt"],
        "equityCurve": ACTIVE_BOTS[user_id]["equityCurve"],
        "initialCapital": ui_capital,
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
        ACTIVE_BOTS[user_id].update({
            "status": "stopped", "positions": [], "trade_history": [], "equityCurve": [], "logs": []})
        await emit_status(user_id, {
            "status": "stopped", "currentBalance": ACTIVE_BOTS[user_id]['balance'],
            "activePositions": [], "tradeMarkers": [], "equityCurve": [], "startedAt": None})
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
        config['mlThresholdLong']  = final_thresh
        config['mlThresholdShort'] = config.get('mlThresholdShort') or final_thresh
        config['risk_percentage']  = config.get('riskPercentage', 1.0)
        config['strategies']       = [{"code": request.code, "params": request.params}]
        from app.backtest2 import Backtester
        tester = Backtester(config)
        result = await tester.run()
        return json.loads(json.dumps(result, default=str))
    except Exception as e:
        logger.error(f"❌ Backtest Run Error: {e}")
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})


@app.post('/api/backtest/combo')
async def run_combo_backtest(req: ComboRequest):
    async def event_generator():
        try:
            config = req.dict()
            config['mlThresholdLong']  = config.get('mlThresholdLong', 0.8)
            config['mlThresholdShort'] = config.get('mlThresholdShort', 0.8)
            config['risk_percentage']  = config.get('risk_percentage', 1.0)
            yield f"{json.dumps({'status': 'progress', 'percentage': 10, 'message': 'Assembling AI Council...'}, default=str)}\n"
            logger.info(f"⚖️ COMBO RUN START: {len(config.get('strategies', []))} Strategies")
            from app.backtest2 import Backtester
            tester = Backtester(config)
            yield f"{json.dumps({'status': 'progress', 'percentage': 30, 'message': 'Fetching Market History...'}, default=str)}\n"
            result = await tester.run()
            yield f"{json.dumps({'status': 'progress', 'percentage': 90, 'message': 'Finalizing Analytics...'}, default=str)}\n"
            yield f"{json.dumps({'status': 'success', 'result': result}, default=str)}\n"
        except Exception as e:
            logger.error(f"❌ Combo Stream Error: {e}")
            yield f"{json.dumps({'status': 'error', 'message': str(e)}, default=str)}\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Connection": "keep-alive"})


@app.get("/api/bot/status")
async def get_status(userId: str):
    bot = ACTIVE_BOTS.get(userId.strip())
    if not bot: bot = DatabaseHandler.load_state(userId.strip())
    if bot:
        return {
            "status":            bot.get("status", "stopped"),
            "currentBalance":    bot.get("currentBalance", bot.get("balance", 0)),
            "balance":           bot.get("balance", 0),
            "unrealizedPnl":     bot.get("unrealizedPnl", 0),
            "exposure":          bot.get("exposure", 0),
            "currentConfidence": bot.get("currentConfidence", 50),
            "signalsMap":        bot.get("signalsMap", {}),
            "equityCurve":       bot.get("equityCurve", []),
            "logs":              bot.get("logs", []),
            "activePositions":   bot.get("positions", []),
            "positions":         bot.get("positions", []),
            "startedAt":         bot.get("startedAt"),
            "config":            bot.get("config"),
            "candles":           bot.get("candles", []),
            "trade_history":     bot.get('trade_history', []),
            "tradeHistory":      bot.get('trade_history', []),
            "tradeMarkers":      bot.get('trade_history', []),
            "dailyProfit":       bot.get("dailyProfit", 0),
            "initialCapital":    (bot.get("config", {}).get("capitalAllocation") or
                                  bot.get("config", {}).get("initialBalance") or
                                  bot.get("balance", 0)),
            "aiRegimeTitle":     bot.get("aiRegimeTitle",  "Mean-Reverting Consolidation"),
            "aiRegimeDesc":      bot.get("aiRegimeDesc",   "Sideways Range"),
            "aiDeployedGear":    bot.get("aiDeployedGear", "Syncing Core Strategy Modules..."),
            "winRate":           bot.get("winRate",        0.0),
            "profitFactor":      bot.get("profitFactor",   1.0),
        }
    return {"status": "inactive", "balance": 0}


@app.post("/api/bot/reset")
async def reset_bot(data: BotStopRequest):
    if data.userId in ACTIVE_BOTS:
        ACTIVE_BOTS[data.userId].update({
            "status": "stopped", "positions": [], "trade_history": [], "equityCurve": [], "logs": []})
        DatabaseHandler.save_state(data.userId, ACTIVE_BOTS[data.userId])
        return {"status": "reset"}
    return {"status": "not_found"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
