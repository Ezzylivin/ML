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
import aiohttp

# 🟢 SOCKET HELPERS (Must be async/await)
from app.services.socket_emitter import emit_log, emit_status
from app.backtest2 import Backtester 
from app.config2 import MODEL_DIR

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
    start_date: str
    endDate: str
    initial_capital: float = 1000.0
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
    global GLOBAL_SESSION, global_exchange
    GLOBAL_SESSION = aiohttp.ClientSession()
    global_exchange = ccxtpro.coinbase({'enableRateLimit': True, 'session': GLOBAL_SESSION})
    yield

    await GLOBAL_SESSION.close()
    await global_exchange.close()
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

def calculate_strategy_signal(df, code, params):
    close, high, low, vol = df['close'], df['high'], df['low'], df['volume']
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
            bb = ta.bbands(close, length=int(params.get('bb_period', 20)), std=float(params.get('bb_std', 2.0)))
            if bb is not None:
                signal[close < bb.iloc[:, 0]] = 1
                signal[close > bb.iloc[:, 2]] = -1
        elif code == "atr_breakout":
            atr = ta.atr(high, low, close, length=int(params.get('atr_length', 14)))
            sma = ta.sma(close, length=20)
            signal[close > (sma + atr * float(params.get('multiplier', 1.5)))] = 1
            signal[close < (sma - atr * float(params.get('multiplier', 1.5)))] = -1
        elif code == "macd_crossover":
            macd = ta.macd(close, fast=int(params.get('fast', 12)), slow=int(params.get('slow', 26)), signal=int(params.get('signal', 9)))
            if macd is not None:
                signal[macd.iloc[:, 0] > macd.iloc[:, 2]] = 1
                signal[macd.iloc[:, 0] < macd.iloc[:, 2]] = -1
        elif code == "stoch":
            stoch = ta.stoch(high, low, close, k=int(params.get('k_period', 14)))
            if stoch is not None:
                k, d = stoch.iloc[:, 0], stoch.iloc[:, 1]
                signal[(k > d) & (k < 30)] = 1
                signal[(k < d) & (k > 70)] = -1
        elif code == "supertrend":
            st = ta.supertrend(high, low, close, length=int(params.get('st_atr', 10)), multiplier=float(params.get('st_factor', 3.0)))
            if st is not None:
                signal[st.iloc[:, 1] == 1] = 1
                signal[st.iloc[:, 1] == -1] = -1
        elif code == "ema_cloud":
            fast_ema = ta.ema(close, length=int(params.get('fast_ema', 9)))
            slow_ema = ta.ema(close, length=int(params.get('slow_ema', 21)))
            signal[fast_ema > slow_ema] = 1
            signal[fast_ema < slow_ema] = -1
        elif code == "pa_breakout":
            lb = int(params.get('lookback', 20))
            highest = high.rolling(lb).max().shift(1)
            lowest = low.rolling(lb).min().shift(1)
            signal[close > highest] = 1
            signal[close < lowest] = -1
        elif code == "vol_profile":
            vol_ma = ta.sma(vol, length=int(params.get('vol_ma', 20)))
            vol_spike = vol > (vol_ma * float(params.get('threshold', 1.5)))
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
    _model_cache = {}
    @staticmethod
    def get_prediction(model_id: str, df: pd.DataFrame) -> float:
        try:
            recent = df.tail(20)
            if len(recent) < 10: return 0.5
            momentum = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
            if model_id not in NeuralPredictor._model_cache:
                model_path = f"./models/{model_id}_model.pkl"
                if os.path.exists(model_path):
                    logger.info(f"🧠 Loading Neural Model into Cache: {model_id}")
                    NeuralPredictor._model_cache[model_id] = joblib.load(model_path)
                else:
                    NeuralPredictor._model_cache[model_id] = None
            cached_model = NeuralPredictor._model_cache.get(model_id)
            if cached_model:
                prediction = cached_model.predict_proba([[momentum]])[0][1]
                return float(prediction)
            base = 1.0 / (1.0 + np.exp(-momentum * 100))
            return float(min(0.99, max(0.01, base)))
        except Exception as e:
            logger.error(f"🧠 Neural Predictor Error: {e}")
            return 0.5

async def execute_backtest_logic(data: BacktestRequest):
    try:
        df = None
        csv_path = f"data/{data.symbol.replace('/', '-')}_{data.timeframe}.csv"
        if os.path.exists(csv_path):
            logger.info(f"📂 Loading historical data from {csv_path}")
            df = pd.read_csv(csv_path)
            df.columns = [c.lower() for c in df.columns]
            rename_map = {'timestamp': 'time', 'date': 'time', 'volume': 'vol'}
            df.rename(columns=rename_map, inplace=True)
            df['time'] = pd.to_datetime(df['time'])
            start_dt = pd.to_datetime(data.start_date).replace(tzinfo=None)
            end_dt = pd.to_datetime(data.end_date).replace(tzinfo=None)
            if df['time'].dt.tz is not None:
                df['time'] = df['time'].dt.tz_localize(None)
            df = df[(df['time'] >= start_dt) & (df['time'] <= end_dt)]
            df.reset_index(drop=True, inplace=True)

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
        sim_config = {
            "strategies": [{"code": data.trend_strategy, "params": data.params}],
            "comboConfig": {"combinationRule": "OR"},
            "mlModel": data.mlModel or "stacking" 
        }

        for i in range(50, len(df)):
            window = df.iloc[:i+1].copy()
            sig, _, _, _ = StrategyBrain.calculate_signals(window, sim_config, 0.5, 0.5)
            row = df.iloc[i]
            price = row['close']
            ts = row['time'].isoformat()

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
    df.columns = [c.lower() for c in df.columns]
    if 'vol' in df.columns: df.rename(columns={'vol': 'volume'}, inplace=True)
    
    for strat in strategies:
        code = strat.get('code')
        p = strat.get('params', {})
        try:
            if code in ['bb_fade', 'bollinger_bands']:
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
            elif code == 'macd_crossover':
                macd = ta.macd(df['close'], fast=int(p.get('fast', 12)), slow=int(p.get('slow', 26)), signal=int(p.get('signal', 9)))
                df['macd'], df['macd_signal'] = macd.iloc[:, 0], macd.iloc[:, 2]
            elif code == 'stoch':
                stoch = ta.stoch(df['high'], df['low'], df['close'], k=int(p.get('k_period', 14)))
                df['stoch_k'], df['stoch_d'] = stoch.iloc[:, 0], stoch.iloc[:, 1]
            elif code == 'atr_breakout':
                atr = ta.atr(df['high'], df['low'], df['close'], length=int(p.get('atr_length', 14)))
                ema20 = ta.ema(df['close'], 20)
                df['atr_upper'] = ema20 + (atr * float(p.get('multiplier', 1.5)))
                df['atr_lower'] = ema20 - (atr * float(p.get('multiplier', 1.5)))
            elif code == 'pa_breakout':
                lb = int(p.get('lookback', 20))
                df['pa_high'] = df['high'].rolling(lb).max()
                df['pa_low'] = df['low'].rolling(lb).min()
            elif code == 'vol_profile':
                if 'volume' in df.columns:
                    df['vol_ma'] = ta.sma(df['volume'], length=int(p.get('vol_ma', 20)))
        except Exception: continue

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
                if code == "rsi_threshold":
                    rsi = ta.rsi(df['close'], length=int(p.get('rsi_length', 14))).iloc[-1]
                    if rsi < p.get('oversold', 30): votes += 1; active_thoughts.append(f"RSI Low ({int(rsi)})")
                    elif rsi > p.get('overbought', 70): votes -= 1; active_thoughts.append(f"RSI High ({int(rsi)})")
                elif code == "stoch":
                    stoch_df = ta.stoch(df['high'], df['low'], df['close'], k=int(p.get('k_period', 14)))
                    if stoch_df is not None and len(stoch_df) > 0:
                        k_val = stoch_df.iloc[-1, 0]
                        if k_val < 20: votes += 1; active_thoughts.append(f"Stoch Low ({int(k_val)})")
                        elif k_val > 80: votes -= 1; active_thoughts.append(f"Stoch High ({int(k_val)})")
                elif code == "macd_crossover":
                    macd_df = ta.macd(df['close'], fast=int(p.get('fast', 12)), slow=int(p.get('slow', 26)), signal=int(p.get('signal', 9)))
                    if macd_df is not None:
                        if macd_df.iloc[-1, 0] > macd_df.iloc[-1, 2]: votes += 1; active_thoughts.append("MACD Bullish")
                        else: votes -= 1
                elif code == "supertrend":
                    st_df = ta.supertrend(df['high'], df['low'], df['close'], length=int(p.get('st_atr', 10)), multiplier=float(p.get('st_factor', 3.0)))
                    if st_df.iloc[-1, 1] == 1: votes += 1; active_thoughts.append("SuperTrend Long")
                    else: votes -= 1
                elif code == "ema_cloud":
                    ema_fast = ta.ema(df['close'], length=int(p.get('fast_ema', 9))).iloc[-1]
                    ema_slow = ta.ema(df['close'], length=int(p.get('slow_ema', 21))).iloc[-1]
                    if current_price > ema_slow: votes += 1; active_thoughts.append("Above EMA Cloud")
                    else: votes -= 1
                elif code == "sma_crossover":
                    sma_fast = ta.sma(df['close'], length=int(p.get('fast_sma', 50))).iloc[-1]
                    sma_slow = ta.sma(df['close'], length=int(p.get('slow_sma', 200))).iloc[-1]
                    if sma_fast > sma_slow: votes += 1; active_thoughts.append("SMA Golden Cross")
                elif code == "bb_fade":
                    if current_price < lower: votes += 1; active_thoughts.append("Price < BB Floor")
                    elif current_price > upper: votes -= 1; active_thoughts.append("Price > BB Ceiling")
                elif code == "atr_breakout":
                    atr = ta.atr(df['high'], df['low'], df['close'], length=int(p.get('atr_length', 14))).iloc[-1]
                    if current_price > (ema20 + atr * float(p.get('multiplier', 1.5))): votes += 1; active_thoughts.append("ATR Breakout")
                elif code == "pa_breakout":
                    lookback = int(p.get('lookback', 20))
                    if current_price >= df['high'].tail(lookback).max(): votes += 1; active_thoughts.append("PA High Break")
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
        rule = config.get('comboConfig', {}).get('combinationRule', 'OR')
        final_sig = 0
        if gate_passed:
            if rule == "AND":
                if votes >= len(strategies) and not is_short: final_sig = 1
                elif votes <= -len(strategies) and is_short: final_sig = -1
            else: # "OR"
                if votes > 0 and not is_short: final_sig = 1
                elif votes < 0 and is_short: final_sig = -1
        return final_sig, active_thoughts, numeric_details, conf

# ==========================================
# 🚀 3. THE HEARTBEAT (Dynamic Calculation Loop)
# ==========================================
global_exchange = ccxtpro.coinbase({'enableRateLimit': True, 'session': GLOBAL_SESSION})

async def live_neural_heartbeat(user_id: str):
    last_log = 0
    last_ui_update = 0
    
    # 🟢 1. INITIAL SETUP
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
            tsl_pct = float(config['params'].get('trailing_stop', 0.01))
            symbol = config['symbol'].replace('-', '/')
            
            try:
                # 🟢 2. DATA FETCHING
                ohlcv_raw = await fetch_live_candles_ccxt(config['symbol'], config.get('timeframe', '1h'), 350)
                if not ohlcv_raw:
                    logger.warning(f"⚠️ Market feed unstable for {config['symbol']}")
                    await emit_log(user_id, "⚠️ Market Data Feed Unstable - Retrying...")
                    await asyncio.sleep(10); continue

                # 🟢 3. ZERO-DRIFT PATCH & TICKER
                raw_candle_ts = ohlcv_raw[-1]['time']
                ticker = await global_exchange.watch_ticker(symbol)
                current_price = float(ticker['last'])
                
                # Force alignment to 'Now' (Integer Seconds)
                clean_ts = int(datetime.now(timezone.utc).timestamp())
                ohlcv_raw[-1]['close'] = current_price
                ohlcv_raw[-1]['time'] = clean_ts 
                
                local_now_ms = datetime.now(timezone.utc).timestamp() * 1000
                real_lag = round((local_now_ms - ticker['timestamp']) / 1000, 2)
                logger.info(f"⚡ WS PULSE: ${current_price} | Lag: {real_lag}s")
                
                # PNL Calculation
                upnl = sum([(current_price - p['entry']) * p['size'] if p['type'] == 'long' else (p['entry'] - current_price) * p['size'] for p in bot['positions']])
                current_equity = bot['balance'] + upnl

                # 🟢 4. INDICATOR CALCULATION
                df = pd.DataFrame(ohlcv_raw)
                if 'vol' in df.columns: df.rename(columns={'vol': 'volume'}, inplace=True)
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

                # 🟢 5. SIGNAL PROCESSING
                sig, thoughts, nums, score = StrategyBrain.calculate_signals(df, config, 0.5, 0.5)                
                ui_limit = float(config.get('mlThresholdLong', 0.5)) if current_price > ta.ema(df['close'], 200).iloc[-1] else float(config.get('mlThresholdShort', 0.5))
                waiting_msg = DiagnosticLayer.get_pending_conditions(df, config, score, ui_limit)
                status_header = "🔍 SCANNING"

                # Restoring clean_curve for the UI
                clean_curve = [p for p in bot.get("equityCurve", []) if p and isinstance(p, dict) and 'time' in p]

                # 🛡️ DYNAMIC THINKING OVERWRITE (Exit Logic for Active Trade)
                is_in_trade = len(bot['positions']) > 0
                tsl_pct = float(config['params'].get('trailing_stop', 0.01))
                if is_in_trade:
                    p = bot['positions'][0]
                    side = p['type'].upper()
                    target, entry, tsl = p['tp'], p['entry'], p.get('tsl', 0)

                    entry_time = datetime.fromisoformat(p['time'].replace('Z', '+00:00'))
                    duration = datetime.now(timezone.utc) - entry_time
                    days, seconds = duration.days, duration.seconds
                    hours = seconds // 3600
                    minutes = (seconds % 3600) // 60
                    secs = seconds % 60
                    
                    time_str = f"{duration.seconds // 60}m {duration.seconds % 60}s"
                    status_header = f"⚡ ACTIVE: {side} (Held: {time_str})"
                    
                    
                    if hours > 0: time_str = f"{hours}h " + time_str
                    if days > 0: time_str = f"{days}d " + time_str
                    
                    total_distance = abs(target - entry)
                    current_progress = abs(current_price - entry)
                    tp_pct = min(100, round((current_progress / total_distance) * 100, 1)) if total_distance > 0 else 0
                    tp_bar = "┃" + "█" * int(tp_pct / 10) + "░" * (10 - int(tp_pct / 10)) + "┃"

                    stop_dist = abs(current_price - tsl_val)
                    stop_pct = round((stop_dist / current_price) * 100, 2)
                    
                    # Visual buffer bar (closer to 0% = Danger)
                    buffer_filled = max(0, min(10, int(stop_pct * 5))) # Scale: 2% distance = full bar
                    buffer_bar = "┃" + "█" * buffer_filled + "░" * (10 - buffer_filled) + "┃"
                    
                    nums['market']['mindset'] = f"🛡️ SAFETY: TSL Buffer {buffer_bar} {stop_pct}% to STOP"

                    nums['market']['intent'] = (
                        f"🛡️ EXIT LOGIC: {side} @ ${round(entry, 2)} | "
                        f"TSL: ${round(tsl, 2)} | "
                        f"Target: ${round(target, 2)} | "
                        f"Progress: {tp_bar} {tp_pct}%"
                    )

                combined_status = f"{nums['market']['logic']} | {nums['market']['intent']} | {nums['market']['mindset']} | {waiting_msg}"
                await emit_log(user_id, combined_status)


                now_ts = datetime.now().timestamp()
                if (now_ts - last_ui_update >= 10):
                    combined_status = f"{status_header} | {nums['market']['logic']} | {nums['market']['intent']} | {nums['market']['mindset']} | {waiting_msg}"
                    await emit_log(user_id, combined_status)
                    
                    clean_curve = [pt for pt in bot.get("equityCurve", []) if pt and isinstance(pt, dict) and 'time' in pt]
                    markers = [{"time": int(c['time']), "position": "belowBar", "color": "#10b981", "text": thoughts[0] if thoughts else ""} for c in ohlcv_raw[-1:] if thoughts]
                    
                    current_pos_val = sum([pos['entry'] * pos['size'] for pos in bot['positions']])
                    exposure_pct = round((current_pos_val / bot['balance']) * 100, 1) if bot['balance'] > 0 else 0

                    await emit_status(user_id, {
                        "status": "running", "currentBalance": round(current_equity, 2), 
                        "exposure": exposure_pct, "activePositions": bot['positions'],
                        "unrealizedPnl": round(upnl, 2), "tradeMarkers": bot['trade_history'] + markers,
                        "equityCurve": clean_curve, "startedAt": bot.get("startedAt"), 
                        "currentConfidence": int(score * 100), "candles": await process_data_packet(df, strategies)
                    })
                    last_ui_update = now_ts

                # 🟢 5. TRADE EXECUTION: Kelly Criterion Position Sizing

                # 🟢 6. TRADE EXECUTION: Open Trades
                if not is_in_trade and len(bot['positions']) < int(config.get('maxPyramiding', 1)):
                    risk_val = float(config.get('riskPercentage', 1)) / 100
                    size = (bot['balance'] * risk_val) / current_price
                    

                    if sig == 1: # LONG
                        sl_price = current_price * (1 - float(config['params'].get('stop_loss', 0.05)))
                        pos = {
                            "symbol": config.get('symbol', 'BTC-USD'),
                            "type": "long", "entry": current_price, "size": size, 
                            "time": datetime.now(timezone.utc).isoformat(),
                            "sl": sl_price, "tp": current_price * (1 + float(config['params'].get('take_profit', 0.10))),
                            "tsl": sl_price, "tsl_active": True if tsl_pct > 0 else False
                        }
                        bot['positions'].append(pos)
                        bot['trade_history'].append({"type": "buy", "price": current_price, "time": pos['time']})
                        await emit_log(user_id, f"🚀 LONG EXECUTED @ ${current_price}")
                        DatabaseHandler.save_state(user_id, bot)
                    elif sig == -1 and config.get('enable_shorting', True): # SHORT
                        sl_price = current_price * (1 + float(config['params'].get('stop_loss', 0.05)))
                        pos = {
                            "symbol": config.get('symbol', 'BTC-USD'),
                            "type": "short", "entry": current_price, "size": size, 
                            "time": datetime.now(timezone.utc).isoformat(),
                            "sl": sl_price, "tp": current_price * (1 - float(config['params'].get('take_profit', 0.10))),
                            "tsl": sl_price, "tsl_active": True if tsl_pct > 0 else False
                        }
                        bot['positions'].append(pos)
                        bot['trade_history'].append({"type": "short", "price": current_price, "time": pos['time']})
                        await emit_log(user_id, f"🔻 SHORT EXECUTED @ ${current_price}")
                        DatabaseHandler.save_state(user_id, bot)

                # 🟢 7. TRADE EXECUTION: Check Exits
                active_pos = bot['positions'][:]
                for pos in active_pos:
                    pnl, closed = 0, False
                    if pos['type'] == 'long':
                        if tsl_pct > 0:
                            new_tsl = current_price * (1 - tsl_pct)
                            if new_tsl > pos['tsl']: pos['tsl'] = new_tsl
                        if current_price >= pos['tp']: 
                            pnl = (current_price - pos['entry']) * pos['size']
                            closed = True; await emit_log(user_id, f"💰 TP HIT: +${round(pnl, 2)}")
                        elif current_price <= pos['tsl']: 
                            pnl = (current_price - pos['entry']) * pos['size']
                            closed = True; await emit_log(user_id, f"🛑 TSL/SL HIT: ${round(pnl, 2)}")
                    elif pos['type'] == 'short':
                        if tsl_pct > 0:
                            new_tsl = current_price * (1 + tsl_pct)
                            if new_tsl < pos['tsl']: pos['tsl'] = new_tsl
                        if current_price <= pos['tp']: 
                            pnl = (pos['entry'] - current_price) * pos['size']
                            closed = True; await emit_log(user_id, f"💰 TP HIT: +${round(pnl, 2)}")
                        elif current_price >= pos['tsl']: 
                            pnl = (pos['entry'] - current_price) * pos['size']
                            closed = True; await emit_log(user_id, f"🛑 TSL/SL HIT: ${round(pnl, 2)}")
                    
                    if closed:
                        bot['balance'] += pnl
                        bot['positions'].remove(pos)
                        bot['trade_history'].append({"type": "exit", "price": current_price, "pnl": round(pnl, 2), "time": datetime.now(timezone.utc).isoformat()})
                        DatabaseHandler.save_state(user_id, bot)

                # 🟢 8. PACKAGING DATA FOR UI
                keys = ['bb_lower', 'bb_upper', 'ema_fast', 'ema_slow', 'sma_fast', 'sma_slow', 'supertrend', 'pa_high', 'pa_low', 'atr_upper', 'atr_lower', 'rsi', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'vol_ma']
                candles_to_send = []
                for _, row in df.tail(100).iterrows():
                    c_obj = {"time": int(row['time']), "open": row['open'], "high": row['high'], "low": row['low'], "close": row['close']}
                    for k in keys:
                        if k in row and not pd.isna(row[k]): c_obj[k] = float(row[k])
                    candles_to_send.append(c_obj)

                markers = [{"time": int(c['time']), "position": "belowBar", "color": "#10b981", "text": thoughts[0] if thoughts else ""} for c in ohlcv_raw[-1:] if thoughts]
                
                # Correct Exposure % Calculation
                current_pos_val = sum([p['entry'] * p['size'] for p in bot['positions']])
                exposure_pct = round((current_pos_val / bot['balance']) * 100, 1) if bot['balance'] > 0 else 0

                # 🚀 CONSOLIDATED EMIT
                await emit_status(user_id, {
                    "status": "running", 
                    "currentBalance": round(current_equity, 2), 
                    "unrealizedPnl": round(upnl, 2),
                    "activePositions": bot['positions'], 
                    "tradeMarkers": bot['trade_history'] + markers,
                    "equityCurve": clean_curve, 
                    "startedAt": bot.get("startedAt"), 
                    "candles": candles_to_send,
                    "exposure": exposure_pct,
                    "currentConfidence": int(score * 100)
                })
                
                await asyncio.sleep(0.1) # Yield for network

                # Housekeeping
                if datetime.now().timestamp() - last_log >= 60:
                    bot["equityCurve"].append({"time": datetime.now().isoformat(), "balance": round(current_equity, 2), "confidence": int(score * 100)})
                    if len(bot["equityCurve"]) > 100: bot["equityCurve"].pop(0)
                    DatabaseHandler.save_state(user_id, bot)
                    last_log = datetime.now().timestamp()

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"❌ WS Stream Error: {e}"); await asyncio.sleep(5)
    finally:
        logger.info(f"🔌 Heartbeat loop terminated for {user_id}")

# =============================================================
# ENDPOINTS
# =============================================================
async def fetch_live_candles_ccxt(symbol: str, timeframe: str, limit: int):
    async with ccxt.coinbase() as ex:
        try:
            ohlcv = await ex.fetch_ohlcv(symbol.replace('-', '/'), timeframe, limit=limit)
            return [{"time": int(c[0]/1000), "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5] if len(c) > 5 else 0} for c in ohlcv]
        except: return []

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
        config = request.dict()
        p_dict = config.get('params', {})
        def fix_pct(val, default):
            if val is None: return default
            return val / 100 if val > 1.0 else val

        tp = fix_pct(p_dict.get('take_profit'), 0.06)
        sl = fix_pct(p_dict.get('stop_loss'), 0.03)
        ts = fix_pct(p_dict.get('trailing_stop'), 0.02)
        
        config['params']['take_profit'] = tp
        config['params']['stop_loss'] = sl
        config['params']['trailing_stop'] = ts
        
        logger.info(f"🛡️ Atomic Risk Corrected: TP={tp}, SL={sl}, TS={ts}")
        bot = Backtester(config)
        result = bot.run()
        
        if 'candleData' not in result or not result['candleData']:
            df = load_data_robust(config['symbol'], config['timeframe'])
            if df is not None:
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
    try:
        logger.info(f"🔥 COMBO Request: {len(req.strategies)} strategies on {req.symbol}")
        df = load_data_robust(req.symbol, req.timeframe)
        if df is None: raise HTTPException(status_code=404, detail=f"Data file not found for {req.symbol}")
        try:
            df = df.loc[req.startDate:req.endDate]
            if df.empty: raise Exception("No data in date range")
        except: pass

        signals_list = []
        for strat in req.strategies:
            sig_series = calculate_strategy_signal(df, strat.code, strat.params)
            signals_list.append(sig_series)
        
        if not signals_list: return {"status": "error", "message": "No strategies executed"}

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
            
            if position == 1: # LONG
                if high > best_price: best_price = high
                stop_price = entry_price * (1 - sl_pct)
                take_price = entry_price * (1 + tp_pct)
                trail_price = best_price * (1 - ts_pct) if ts_pct > 0 else 0
                exit_reason = None
                exit_p = price
                
                if low <= stop_price: exit_reason = "SL"; exit_p = stop_price
                elif ts_pct > 0 and low <= trail_price: exit_reason = "Trail"; exit_p = trail_price
                elif high >= take_price: exit_reason = "TP"; exit_p = take_price
                
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
                
                if high >= stop_price: exit_reason = "SL"; exit_p = stop_price
                elif ts_pct > 0 and high >= trail_price: exit_reason = "Trail"; exit_p = trail_price
                elif low <= take_price: exit_reason = "TP"; exit_p = take_price
                    
                if exit_reason:
                    pnl_pct = (entry_price - exit_p) / entry_price
                    balance *= (1 + pnl_pct - comm)
                    trades_log.append({"type": "close_short", "price": exit_p, "time": date, "balance": balance, "reason": exit_reason})
                    position = 0
            
            if position == 0:
                if signal == 1:
                    position = 1; entry_price = price; best_price = price; balance *= (1 - comm)
                    trades_log.append({"type": "buy", "price": price, "time": date})
                elif signal == -1:
                    position = -1; entry_price = price; best_price = price; balance *= (1 - comm)
                    trades_log.append({"type": "sell", "price": price, "time": date})
            
            elif position == 1 and signal == -1:
                pnl_pct = (price - entry_price) / entry_price
                balance *= (1 + pnl_pct - comm)
                trades_log.append({"type": "flip_to_short", "price": price, "time": date, "balance": balance, "reason": "Signal Flip"})
                position = -1; entry_price = price; best_price = price; balance *= (1 - comm)
                
            elif position == -1 and signal == 1:
                pnl_pct = (entry_price - price) / entry_price
                balance *= (1 + pnl_pct - comm)
                trades_log.append({"type": "flip_to_long", "price": price, "time": date, "balance": balance, "reason": "Signal Flip"})
                position = 1; entry_price = price; best_price = price; balance *= (1 - comm)

            equity_curve.append({"time": date, "balance": balance})

        roi = ((balance - req.initialBalance) / req.initialBalance) * 100
        chart_df = df.reset_index()[['timestamp', 'open', 'high', 'low', 'close']].copy()
        chart_df['timestamp'] = chart_df['timestamp'].astype(str)
        candle_data = chart_df.to_dict('records')

        return {
            "status": "completed",
            "metrics": {"final_balance": balance, "roi": roi, "total_trades": len(trades_log)},
            "equity_curve": equity_curve,
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
