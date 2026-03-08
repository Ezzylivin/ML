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
    mlThresholdLong: Optional[float] = 0.50
    mlThresholdShort: Optional[float] = 0.50
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


async def ensure_full_data(symbol, timeframe, start_str, end_str):
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
    return df.loc[target_start:target_end]


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
            start_dt = pd.to_datetime(data.startDate).replace(tzinfo=None)
            end_dt = pd.to_datetime(data.endDate).replace(tzinfo=None)
            if df['time'].dt.tz is not None:
                df['time'] = df['time'].dt.tz_localize(None)
            df = df[(df['time'] >= start_dt) & (df['time'] <= end_dt)]
            df.reset_index(drop=True, inplace=True)

        if df is None or df.empty:
            logger.info("⚠️ CSV not found or empty. Falling back to CCXT.")
            async with ccxt.coinbase() as exchange:
                since = exchange.parse8601(data.startDate)
                ohlcv = await exchange.fetch_ohlcv(data.symbol.replace('-', '/'), data.timeframe, since=since, limit=1000)
                df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
                df['time'] = pd.to_datetime(df['time'], unit='ms')

        if df.empty:
            return {"status": "error", "message": "No data found for backtest range"}

        balance, position, trades, curve = data.initialBalance, None, [], []
        sim_config = {
            "strategies": [{"code": data.trend_strategy, "params": data.params}],
            "comboConfig": {"combinationRule": "OR"},
            "mlModel": data.mlModel or "stacking" 
        }

        for i in range(50, len(df)):
            window = df.iloc[:i+1].copy()
            sig, _, _, _,_ = StrategyBrain.calculate_signals(window, sim_config, 0.5, 0.5)
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
            params = config.get('params', {})
            strategies = config.get('strategies', [])
            upnl = 0
            current_equity = bot['balance']
            markers = []
            candles_to_send = []
            tsl_pct = float(config['params'].get('trailing_stop', 0.01))
            symbol = config['symbol'].replace('-', '/')
            max_p = int(config.get('maxPyramiding', 1))

            ui_tp = float(params.get('take_profit', 0.10))
            ui_sl = float(params.get('stop_loss', 0.05))
            ui_tsl = float(params.get('trailing_stop', 0.01))
            
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
                sig, thoughts, nums, score, signals_map = StrategyBrain.calculate_signals(df, config, 0.5, 0.5)                
                ui_limit = float(config.get('mlThresholdLong', 0.5)) if current_price > ta.ema(df['close'], 200).iloc[-1] else float(config.get('mlThresholdShort', 0.5))
                waiting_msg = DiagnosticLayer.get_pending_conditions(df, config, score, ui_limit)
                status_header = "🔍 SCANNING"

                # Restoring clean_curve for the UI
                clean_curve = [p for p in bot.get("equityCurve", []) if p and isinstance(p, dict) and 'time' in p]
                total_raw_pnl = sum([(current_price - p['entry']) * p['size'] if p['type'] == 'long' else (p['entry'] - current_price) * p['size'] for p in bot['positions']])

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

                    stop_dist = abs(current_price - tsl)
                    stop_pct = round((stop_dist / current_price) * 100, 2)
                    
                    # Visual buffer bar (closer to 0% = Danger)
                    buffer_filled = max(0, min(10, int(stop_pct * 5))) # Scale: 2% distance = full bar
                    buffer_bar = "┃" + "█" * buffer_filled + "░" * (10 - buffer_filled) + "┃"




                    total_raw_pnl = sum([(current_price - p['entry']) * p['size'] if p['type'] == 'long' else (p['entry'] - current_price) * p['size'] for p in bot['positions']])
                    
                    
                    exit_details = []
                    status_header = f"⚡ ACTIVE: {len(bot['positions'])} POS"
                    for i, p in enumerate(bot['positions']):
                        side_icon = "🟢" if p['type'] == 'long' else "🔴"
                        # Distance to Trailing Stop
                        dist_to_stop = abs(current_price - p['tsl'])
                        stop_pct = (dist_to_stop / current_price) * 100
                        buffer_bar = DiagnosticLayer.render_progress(stop_pct, 2.0, reverse=True) # 2% is 'Full Safety'
                        
                        exit_details.append(f"Leg{i+1} {side_icon}: TSL {buffer_bar}")

                    
                    waiting_msg = f"🛡️ EXIT MODE | Total: ${total_raw_pnl:,.2f} | " + " | ".join(exit_details)
                    status_header = f"⚡ ACTIVE: {len(bot['positions'])} POS"
                    nums['market']['mindset'] = f"🛡️ SAFETY: TSL Buffer {buffer_bar} {stop_pct}% to STOP"

                    nums['market']['intent'] = (
                        f"🛡️ EXIT LOGIC: {side} @ ${round(entry, 2)} | "
                        f"TSL: ${round(tsl, 2)} | "
                        f"Target: ${round(target, 2)} | "
                        f"Progress: {tp_bar} {tp_pct}%"
                    )
                    combined_status = f"{status_header} | {nums['market']['intent']} | {nums['market']['mindset']} | {waiting_msg}"
                    trade_happended = False

                else:
                   
                     status_header = "🔍 SCANNING"
                     waiting_msg = DiagnosticLayer.get_pending_conditions(df, config, score, ui_limit)
                     hybrid_mode = (
                         config.get('hybridMode') or 
                         config.get('comboConfig', {}).get('combinationRule') or 
                         "AND"
                     ).upper()
                     
                     

                active_results = []     
                for strat in strategies:
                    max_name_len = max([len(s['code'].replace('_', ' ')) for s in strategies]) if strategies else 10
                    code = strat['code']
                    val = int(signals_map.get(code, 0) * 100)
                    display_name = code.replace('_', ' ').upper().ljust(max_name_len)
                    filled = max(0, min(10, val // 10))
                    bar = "┃" + "█" * filled + "░" * (10 - filled) + "┃"
                    active_results.append(f"{display_name}: {bar} {val}%")
                
                targets_str = " | ".join(active_results)

                # 🟢 NEW: UNIFIED DUAL-MODE LOGIC
                # This prevents the Scanning string from overwriting the Exit string
                if len(bot['positions']) > 0:
                    # We are in a trade: Use the exit logic we calculated earlier in the loop
                    exit_section = nums['market']['intent'] 
                    combined_status = f"{status_header} | {exit_section} | 🔍 TARGETS ({hybrid_mode}): {targets_str}"
                else:
                    # We are scanning: Just show logic gates and targets
                    intent_section = nums['market']['intent']
                    combined_status = f"{status_header} | {intent_section} | {nums['market']['logic']} | 🔍 TARGETS ({hybrid_mode}): {targets_str}"

                # 🟢 EMIT LOG (Every 15s)
                now_ts = datetime.now().timestamp()
                if (now_ts - last_log >= 15):
                    await emit_log(user_id, combined_status)
                    last_log = now_ts         


                
                    
                clean_curve = [pt for pt in bot.get("equityCurve", []) if pt and isinstance(pt, dict) and 'time' in pt]
                markers = [{"time": int(c['time']), "position": "belowBar", "color": "#10b981", "text": thoughts[0] if thoughts else ""} for c in ohlcv_raw[-1:] if thoughts]
                    
                current_pos_val = sum([pos['entry'] * pos['size'] for pos in bot['positions']])
                exposure_pct = round((current_pos_val / bot['balance']) * 100, 1) if bot['balance'] > 0 else 0

                now_ts = datetime.now().timestamp()
                if (now_ts - last_ui_update >= 10):
                    await emit_log(user_id, combined_status)
                    exposure_pct = round((sum([p['entry'] * p['size'] for p in bot['positions']]) / bot['balance']) * 100, 1) if bot['balance'] > 0 else 0
                    await emit_status(user_id, {
                        "status": "running", "currentBalance": round(current_equity, 2), "exposure": exposure_pct,
                        "activePositions": bot['positions'], "unrealizedPnl": round(upnl, 2),
                        "tradeMarkers": bot['trade_history'], "candles": await process_data_packet(df, config.get('strategies', [])),
                        "currentConfidence": int(score * 100), "signalsMap": signals_map
                })
                last_ui_update = now_ts
                    
                 
                # 🟢 6. TRADE EXECUTION: Confidence Climb & 60m Stability Logic
                max_p = min(5, int(config.get('maxPyramiding', 5)))
                
                # ⏳ --- STEP A: STABILITY TIME GATE ---
                current_time = datetime.now(timezone.utc)
                last_trade_str = bot.get('last_trade_time')
                time_gate_passed = True
                minutes_remaining = 0

                if last_trade_str:
                    last_trade_dt = datetime.fromisoformat(last_trade_str)
                    minutes_since_last = (current_time - last_trade_dt).total_seconds() / 60
                    
                    if minutes_since_last < 60:
                        time_gate_passed = False
                        minutes_remaining = int(60 - minutes_since_last)

                # 💰 --- STEP B: RISK & SIZE CALCULATIONS ---
                ui_total_risk = float(config.get('riskPercentage', 30.0))
                leg_risk_pct = max(5.0, ui_total_risk / max_p)
                risk_decimal = leg_risk_pct / 100
                size = (bot['balance'] * risk_decimal) / current_price
                
                # 🧠 --- STEP C: CONFIDENCE CLIMB CHECK (+10% Rule) ---
                last_pos = bot['positions'][-1] if bot['positions'] else None
                last_conf = last_pos.get('entry_conf', 0) if last_pos else 0
                
                # Requirement: Score must be 0.10 (10%) higher than the last leg
                climb_satisfied = (score >= last_conf + 0.10) if last_pos else True

                # 🚀 --- STEP D: EXECUTION DECISION ---
                if len(bot['positions']) < max_p:
                    # Directional checks
                    can_long = (sig == 1) and (not last_pos or (last_pos['type'] == 'long' and climb_satisfied) or (last_pos['type'] == 'short'))
                    can_short = (sig == -1) and (not last_pos or (last_pos['type'] == 'short' and climb_satisfied) or (last_pos['type'] == 'long'))

                    if (can_long or can_short):
                        # 🛡️ THE STABILITY LOCK: Check if we are still in the 60m window
                        if not time_gate_passed:
                            # Log every 15s to let you know we are stalking the next leg
                            if (now_ts - last_log >= 15):
                                await emit_log(user_id, f"⏳ STABILITY GATE: Signal is valid ({int(score*100)}%), but waiting {minutes_remaining}m to confirm trend stability.")
                        else:
                            # 🟢 EXECUTE THE TRADE
                            trade_type = "long" if can_long else "short"
                            bot['positions'].append({
                                "symbol": symbol, 
                                "type": trade_type, 
                                "entry": current_price, 
                                "size": size, 
                                "time": current_time.isoformat(), 
                                "tp": current_price * (1 + ui_tp) if can_long else current_price * (1 - ui_tp),
                                "sl": current_price * (1 - ui_sl) if can_long else current_price * (1 + ui_sl), 
                                "tsl": current_price * (1 - ui_sl) if can_long else current_price * (1 + ui_sl),
                                "entry_conf": score 
                            })

                            # 🟢 UPDATE TIMESTAMPS FOR NEXT LEG
                            bot['last_trade_time'] = current_time.isoformat()
                            
                            icon = "🚀" if can_long else "🔻"
                            await emit_log(user_id, f"{icon} {trade_type.upper()} LEG {len(bot['positions'])} | Conf: {int(score*100)}% | Stability Window Reset (60m)")
                            
                            # Save state immediately after trade
                            DatabaseHandler.save_state(user_id, bot)
                

                # 🟢 6. TRADE EXECUTION: Exit Monitoring (Individually per Leg)
                # We iterate over a slice [:] to allow safe removal from the list
                for pos in bot['positions'][:]:
                    closed, exit_reason, pnl = False, "", 0
                    tsl_pct = float(config.get('params', {}).get('trailing_stop', 0.01))
                    
                    # A. LONG LEG MONITORING
                    if pos['type'] == 'long':
                        # 📈 Trailing Stop Movement
                        if tsl_pct > 0:
                            # Calculate where the TSL should be based on CURRENT price
                            potential_tsl = current_price * (1 - tsl_pct)
                            # Only move it UP, never down
                            if potential_tsl > pos['tsl']:
                                pos['tsl'] = potential_tsl

                        # 🛑 Check Exit Conditions
                        if current_price >= pos['tp']:
                            closed, exit_reason = True, "💰 TP HIT"
                            pnl = (current_price - pos['entry']) * pos['size']
                        elif current_price <= pos['tsl']:
                            closed, exit_reason = True, "🛑 TSL HIT"
                            pnl = (current_price - pos['entry']) * pos['size']

                    # B. SHORT LEG MONITORING
                    elif pos['type'] == 'short':
                        # 📉 Trailing Stop Movement
                        if tsl_pct > 0:
                            # Calculate where the TSL should be based on CURRENT price
                            potential_tsl = current_price * (1 + tsl_pct)
                            # Only move it DOWN, never up
                            if potential_tsl < pos['tsl']:
                                pos['tsl'] = potential_tsl

                        # 🛑 Check Exit Conditions
                        if current_price <= pos['tp']:
                            closed, exit_reason = True, "💰 TP HIT"
                            pnl = (pos['entry'] - current_price) * pos['size']
                        elif current_price >= pos['tsl']:
                            closed, exit_reason = True, "🛑 TSL HIT"
                            pnl = (pos['entry'] - current_price) * pos['size']

                    # C. EXECUTE CLOSURE
                    if closed:
                        bot['balance'] += pnl
                        bot['positions'].remove(pos)
                        
                        # Record in history
                        exit_time = datetime.now(timezone.utc).isoformat()
                        bot['trade_history'].append({
                            "type": "exit", 
                            "side": pos['type'],
                            "price": current_price, 
                            "pnl": round(pnl, 2), 
                            "time": exit_time,
                            "reason": exit_reason
                        })
                        
                        await emit_log(user_id, f"{exit_reason}: {pos['type'].upper()} closed at ${current_price} | PnL: ${round(pnl, 2)}")
                        trade_happened = True
                        DatabaseHandler.save_state(user_id, bot)

                # 🟢 Final Breath for the Loop
                await asyncio.sleep(0.1)

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
                # 🟢 1. UI HEARTBEAT (Execute every loop)
                # This appends to the curve immediately so the UI charts draw lines instantly.
                bot["equityCurve"].append({
                    "time": datetime.now().isoformat(), 
                    "balance": round(current_equity, 2), 
                    "confidence": int(score * 100)
                })
                
                # Keep the memory buffer lean (last 100-300 points)
                if len(bot["equityCurve"]) > 300: 
                    bot["equityCurve"].pop(0)
                
                # 🟢 2. DATABASE SYNC (Execute every 60s)
                # We only write to the disk/database periodically to save performance.
                if datetime.now().timestamp() - last_log >= 60:
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

        # 1. Standardize percentages (matches your engine logic)
        p_dict = config.get('params', {})
        def fix_pct(val, default):
            if val is None: return default
            return val / 100 if val > 1.0 else val

        config['params']['take_profit'] = fix_pct(p_dict.get('take_profit'), 0.13)
        config['params']['stop_loss'] = fix_pct(p_dict.get('stop_loss'), 0.086)
        config['params']['trailing_stop'] = fix_pct(p_dict.get('trailing_stop'), 0.086)
        
        logger.info(f"🛡️ Atomic Request: {config.get('code')} on {config['symbol']}")

        # 2. Call the Unified Engine (backtest2.py)
        # This automatically handles ML Vetoes, Intra-candle exits, and Ledger AI scores
        from app.backtest2 import Backtester
        bot = Backtester(config)
        result = await bot.run()

        # 3. Return EVERYTHING 
        # Don't manually build 'final_response'—just return the result dictionary
        # This ensures 'vetoed_signals' and 'ai_score' are included.
        return result

    except Exception as e:
        logger.error(f"❌ Atomic API Error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "failed", "error": str(e)}
        )
        
@app.post('/api/backtest/combo')
async def run_combo_backtest(req: ComboRequest):
    try:
        # 1. Convert the incoming request to a dictionary
        config = req.dict()
        
        # 2. Initialize your specialized Engine
        from app.backtest2 import Backtester
        tester = Backtester(config)
        
        # 3. Run the engine (This uses the AI, Vetoes, and TSL logic)
        result = await tester.run()
        
        # 4. Return the complete result (including vetoed_signals)
        return result

    except Exception as e:
        logger.error(f"Combo API Error: {e}")
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})



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
