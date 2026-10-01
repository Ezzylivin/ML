import os
import json
import pandas as pd
import numpy as np
import joblib
import logging
import traceback
import math
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas_ta as ta
import warnings
from copy import deepcopy
from datetime import datetime, timezone, timedelta

# --- 1. CONFIGURATION ---
os.environ['PYTHONHASHSEED'] = '12345'
np.random.seed(12345)
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/root/Project/ML"
MODEL_DIR = "/root/Project/ML/app/models"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "optimizer_results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_TAKER_FEE = 0.006
SLIPPAGE_PCT = 0.001

app = FastAPI(title="Trading ML Server API v7.0")

# --- 2. MODELS ---
class BacktestConfig(BaseModel):
    symbol: str
    timeframe: str
    startDate: str
    endDate: str
    mlMode: str = 'off'
    mlModel: Optional[str] = None
    mlThreshold: float = 0.65
    code: Optional[str] = None
    strategies: Optional[List[Dict[str, Any]]] = None
    initialBalance: float = 1000.0
    fee: float = DEFAULT_TAKER_FEE
    params: Dict[str, Any] = Field(default_factory=dict)

class BotConfig(BaseModel):
    symbol: str
    timeframe: str
    strategyId: Optional[str] = None
    comboConfig: Optional[Dict[str, Any]] = None
    capitalAllocation: float = 1000.0
    tradingMode: str = 'paper'
    params: Optional[Dict[str, Any]] = None

class CertifyConfig(BaseModel):
    symbol: str
    timeframe: str
    best_params: Dict[str, Any]

# --- 3. HELPERS ---
def safe_int(val, default=0):
    try: return int(float(val))
    except: return default

def safe_float(val, default=0.0):
    try: return float(val)
    except: return default

def normalize_params(config: BacktestConfig) -> Dict:
    merged = deepcopy(config.params)
    if config.strategies:
        for strat in config.strategies:
            for k, v in strat.get('params', {}).items():
                merged[k] = v
    return merged

def find_col(df, key_fragment):
    if key_fragment in df.columns:
        return key_fragment
    for col in df.columns:
        if col.lower() == key_fragment.lower():
            return col
    for col in sorted(df.columns, key=len, reverse=True):
        if col.upper().startswith(key_fragment.upper()):
            return col
    return None

def load_efficient_data(symbol, timeframe, start_date, end_date):
    safe_symbol = symbol.replace('/', '-')
    data_path = os.path.join(DATA_DIR, f"{safe_symbol}-{timeframe}.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.index = df.index.tz_localize("UTC")

    try:
        buffer = pd.to_datetime(start_date, utc=True) - pd.Timedelta(days=365)
        df = df.loc[buffer:pd.to_datetime(end_date, utc=True)]
    except:
        pass

    df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}, inplace=True)

    return df

def convert_numpy_types(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj

# --- Features ---
def engineer_features_for_backtest(df, params):
    df = df.copy()
    df[['open','high','low','close']] = df[['open','high','low','close']].fillna(method='ffill')

    try:
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.bbands(append=True)
        df.ta.stoch(append=True)
        df.ta.cci(append=True)
        df.ta.psar(append=True)
        df.ta.obv(append=True)
        df.ta.ichimoku(append=True)
        df.ta.ema(length=20, append=True)
        df.ta.macd(append=True)
    except:
        pass

    return df

# --- TA Signals ---
def generate_ta_signals(df, strategy_code, params):
    df = df.copy()
    df["ta_signal"] = 0

    try:
        if strategy_code == "psar_signal":
            psarl = find_col(df, "PSARl")
            psars = find_col(df, "PSARs")
            if psarl and psars:
                df.loc[df[psarl] > 0, "ta_signal"] = 1
                df.loc[df[psars] > 0, "ta_signal"] = -1
    except:
        pass

    return df

# --- Backtest ---
def run_backtest(df, signal_col, initial_balance, fee):
    balance = float(initial_balance)
    position = 0
    trades = []
    equity_curve = []
    entry_price = 0
    entry_time = None
    entry_balance = 0

    fee_pct = float(fee)
    slippage_pct = float(SLIPPAGE_PCT)

    equity_curve.append({"timestamp": df.index[0].isoformat(), "balance": balance})

    for i in range(1, len(df)):
        price = float(df["close"].iloc[i])
        sig = df[signal_col].iloc[i]
        t = df.index[i]

        if position == 0:
            if sig == 1:
                position = 1
                entry_price = price * (1 + slippage_pct)
                entry_time = t
                entry_balance = balance
            elif sig == -1:
                position = -1
                entry_price = price * (1 - slippage_pct)
                entry_time = t
                entry_balance = balance

        elif position == 1 and sig == -1:
            exit_price = price * (1 - slippage_pct)
            pnl = (exit_price - entry_price) / entry_price
            profit = entry_balance * pnl - (entry_balance * fee_pct)
            balance += profit
            trades.append({"entryTime": entry_time.isoformat(), "exitTime": t.isoformat(), "profit": profit})
            position = 0

        elif position == -1 and sig == 1:
            exit_price = price * (1 + slippage_pct)
            pnl = (entry_price - exit_price) / entry_price
            profit = entry_balance * pnl - (entry_balance * fee_pct)
            balance += profit
            trades.append({"entryTime": entry_time.isoformat(), "exitTime": t.isoformat(), "profit": profit})
            position = 0

        equity_curve.append({"timestamp": t.isoformat(), "balance": balance})

    wins = [t for t in trades if t["profit"] > 0]
    balances = [e["balance"] for e in equity_curve]

    max_dd = 0
    peak = balances[0]
    for b in balances:
        peak = max(peak, b)
        dd = (peak - b) / peak * 100
        max_dd = max(max_dd, dd)

    total_return = ((balance - initial_balance) / initial_balance * 100)

    return {
        "metrics": {
            "totalReturn": total_return,
            "finalBalance": balance,
            "totalTrades": len(trades),
            "winningTrades": len(wins),
            "losingTrades": len(trades) - len(wins),
            "winRate": (len(wins) / len(trades) * 100) if trades else 0,
            "maxDrawdown": max_dd
        },
        "equityCurve": equity_curve,
        "tradeBreakdown": trades,
        "candleData": df.reset_index().to_dict("records")
    }

# --- Endpoints ---
@app.post("/api/ml/run-backtest-on")
@app.post("/api/ml/run-combo-backtest")
async def handle_run_combo_backtest(config: BacktestConfig):
    try:
        params = normalize_params(config)
        df = load_efficient_data(config.symbol, config.timeframe, config.startDate, config.endDate)

        df = engineer_features_for_backtest(df, params)

        if config.code:
            df = generate_ta_signals(df, config.code, params)
            df["comb"] = df["ta_signal"]

        elif config.strategies:
            cols = []
            for i, s in enumerate(config.strategies):
                df = generate_ta_signals(df, s["code"], params)
                df[f"s_{i}"] = df["ta_signal"]
                cols.append(f"s_{i}")

            if len(cols) > 1:
                df["comb"] = df[cols].apply(lambda r: 1 if (r==1).any() else -1 if (r==-1).any() else 0, axis=1)
            elif cols:
                df["comb"] = df[cols[0]]
            else:
                df["comb"] = 0
        else:
            df["comb"] = 0

        res = run_backtest(df, "comb", config.initialBalance, config.fee)
        return JSONResponse(content=convert_numpy_types(res))

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))


@app.get("/api/ml/available-models")
def list_models():
    return [{"id": f.split(".")[0], "name": f.split(".")[0]}
            for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")]


@app.get("/api/bot/winners")
def get_winners():
    if not os.path.exists(RESULTS_DIR):
        return []

    winners = []

    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".json") or "winner" not in fname:
            continue

        try:
            with open(os.path.join(RESULTS_DIR, fname), "r") as f:
                data = json.load(f)

            raw_strat = data.get("strategies", "Unknown")

            if isinstance(raw_strat, list):
                strat_name = ",".join([s.get("code", "Unknown") for s in raw_strat])
            else:
                strat_name = str(raw_strat)

            metrics = data.get("metrics", {})
            calmar = metrics.get("calmar", metrics.get("StitchedCalmarRatio", 0))

            winners.append({
                "id": fname,
                "name": f"{strat_name} (Calmar: {calmar:.2f})",
                "config": data,   # fixed
                "timestamp": data.get("timestamp", "")
            })

        except Exception as e:
            logger.error(f"Error reading {fname}: {e}")

    winners.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return winners


@app.post("/api/bot/start")
async def start_live_bot(config: BotConfig):
    return {"status": "started", "message": "Bot loop initialized"}

@app.post("/api/bot/stop")
async def stop_live_bot():
    return {"status": "stopped", "message": "Bot loop terminated"}

@app.get("/api/bot/status")
async def get_live_status():
    return {
        "status": "running",
        "activeBots": ["BTC/USDT", "ETH/USDT"],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
