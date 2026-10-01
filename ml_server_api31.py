# File: ml_server_api19.py (Fixed Variable Definition)
import os
import json
import pandas as pd
import numpy as np
import joblib
import logging
import typing
import math
import uuid
import traceback
import time
import threading
import requests
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas_ta as ta
import warnings
from copy import deepcopy
from decimal import Decimal, getcontext, ROUND_HALF_UP

# --- 1. CONFIGURATION & CONSTANTS ---
getcontext().prec = 28
os.environ['PYTHONHASHSEED'] = '12345'
np.random.seed(12345)
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/root/Project/ML"
MODEL_DIR = "/root/Project/ML/app/models"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OPTIMIZER_CACHE_DIR = os.path.join(DATA_DIR, "optimizer_cache")
RESULTS_DIR = os.path.join(DATA_DIR, "optimizer_results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OPTIMIZER_CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RISK_FREE_RATE = 0.02
MAX_LEVERAGE = 10
SLIPPAGE_PCT = 0.001        
DEFAULT_TAKER_FEE = 0.006   
PRICE_PRECISION = 2
SIZE_PRECISION = 8
EPSILON = 1e-9

app = FastAPI(
    title="Trading ML Server API v3.8 (Fixed Risk Var)",
    description="Backtesting, Certification, and Live Coinbase Paper Trading",
    version="3.8.0" 
)

# ==============================================================================
# 2. PYDANTIC MODELS
# ==============================================================================

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
    riskManagementMode: str = 'standard'
    riskPercentage: float = 1.0
    growthCapitalTarget: Optional[float] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    optimizer_mode: bool = False

class CertifyConfig(BaseModel):
    symbol: str
    timeframe: str
    best_params: Dict[str, Any] 
    base_config: Dict[str, Any]

class BotStartConfig(BaseModel):
    symbol: str = 'BTC-USD'
    timeframe: str = '1h'
    capitalAllocation: float = 1000
    mlMode: str = 'off'
    mlModel: Optional[str] = None
    mlThreshold: float = 0.65
    isCombo: bool = False
    strategies: Optional[List[Dict[str, Any]]] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    comboConfig: Optional[Dict[str, Any]] = None 

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

def safe_int(val, default=0):
    try:
        if val is None: return default
        if isinstance(val, (float, np.float64, np.float32)) and (np.isnan(val) or np.isinf(val)): return default
        return int(float(val))
    except: return default

def safe_float(val, default=0.0):
    try:
        if val is None: return default
        if isinstance(val, (float, np.float64, np.float32)) and (np.isnan(val) or np.isinf(val)): return default
        return float(val)
    except: return default

def sanitize_params(params: Dict) -> Dict:
    clean = {}
    for k, v in params.items():
        if any(x in k.lower() for x in ['period', 'length', 'leaves', 'depth', 'estimators']):
            clean[k] = safe_int(v)
        elif any(x in k.lower() for x in ['mult', 'pct', 'std', 'threshold', 'alpha', 'lambda', 'rate']):
            clean[k] = safe_float(v)
        else:
            clean[k] = v
    return clean

def set_nested_value(d, keys, value):
    keys = keys.split('.')
    for key in keys[:-1]: d = d.setdefault(key, {})
    d[keys[-1]] = value

def find_col(df, key, exclude=None):
    if key in df.columns: return key
    for col in df.columns:
        if col.lower() == key.lower(): return col
        
    key_upper = key.upper()
    candidates = []
    for col in df.columns:
        col_upper = col.upper()
        if exclude and exclude.upper() in col_upper: continue
        parts = col_upper.split('_')
        if key_upper in parts: 
             candidates.append(col)
        elif col_upper.startswith(key_upper + '_'):
             candidates.append(col)

    if not candidates: return None
    candidates.sort(key=len)
    return candidates[0]

def load_efficient_data(symbol, timeframe, start_date, end_date):
    safe_symbol = symbol.replace('/', '-')
    data_filename = f"{safe_symbol}-{timeframe}.csv"
    data_path = os.path.join(DATA_DIR, data_filename)
    if not os.path.exists(data_path): raise FileNotFoundError(f"Raw data file not found: {data_path}.")
    try: buffer_start_dt = pd.to_datetime(start_date, utc=True) - pd.Timedelta(days=300) 
    except Exception: buffer_start_dt = pd.to_datetime('2017-01-01', utc=True)
    end_dt = pd.to_datetime(end_date, utc=True)
    df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
    else: df.index = df.index.tz_convert('UTC')
    df_sliced = df.loc[buffer_start_dt:(end_dt + pd.Timedelta(days=1))].copy()
    df_sliced.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True, errors='ignore')
    return df_sliced

def convert_numpy_types(obj):
    if isinstance(obj, Decimal): return float(obj)
    if isinstance(obj, (np.bool_, bool)): return bool(obj)
    if isinstance(obj, (np.floating, np.float64)): return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, np.ndarray): return convert_numpy_types(obj.tolist()) 
    if isinstance(obj, (pd.Timestamp, datetime)): return obj.isoformat()
    if isinstance(obj, pd.DataFrame): return convert_numpy_types(obj.to_dict('records'))
    if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()} 
    if isinstance(obj, list): return [convert_numpy_types(v) for v in obj] 
    return obj

def normalize_params(config_strategies: List[Dict], global_params: Dict) -> Dict:
    merged = deepcopy(global_params)
    if not config_strategies: return merged
    for strat in config_strategies:
        p = strat.get('params', {})
        code = strat.get('code', '')
        safe_p = sanitize_params(p)
        
        if 'bollinger' in code or 'bb' in code:
            if 'period' in safe_p: merged['bb_length'] = safe_p['period']
            if 'stdDev' in safe_p: merged['bb_std'] = safe_p['stdDev']
            if 'bb_length' in safe_p: merged['bb_length'] = safe_p['bb_length']
            if 'bb_std' in safe_p: merged['bb_std'] = safe_p['bb_std']
        if 'rsi' in code:
            if 'period' in safe_p: merged['rsi_length'] = safe_p['period']
            if 'rsi_length' in safe_p: merged['rsi_length'] = safe_p['rsi_length']
        if 'sma' in code or 'ema' in code:
            if 'shortPeriod' in safe_p: merged['sma_fast_period'] = safe_p['shortPeriod']
            if 'longPeriod' in safe_p: merged['sma_slow_period'] = safe_p['longPeriod']
            if 'sma_fast_period' in safe_p: merged['sma_fast_period'] = safe_p['sma_fast_period']
            if 'sma_slow_period' in safe_p: merged['sma_slow_period'] = safe_p['sma_slow_period']
        if 'stoch' in code:
            if 'kPeriod' in safe_p: merged['k_period'] = safe_p['kPeriod']
            if 'dPeriod' in safe_p: merged['d_period'] = safe_p['dPeriod']
            if 'k_period' in safe_p: merged['k_period'] = safe_p['k_period']
            if 'd_period' in safe_p: merged['d_period'] = safe_p['d_period']
        if 'cci' in code:
            if 'period' in safe_p: merged['cci_length'] = safe_p['period']
            if 'cci_length' in safe_p: merged['cci_length'] = safe_p['cci_length']
    return sanitize_params(merged)

# --- FEATURE ENGINEERING ---
def engineer_features_for_backtest(df: pd.DataFrame, trend_filter_period: typing.Optional[int], params: dict = None) -> pd.DataFrame:
    df = df.copy()
    if df['close'].isnull().any(): df['close'] = df['close'].fillna(method='ffill')
    df.replace([np.inf, -np.inf], np.nan, inplace=True) 
    if params is None: params = {} 

    tf_period = safe_int(trend_filter_period, 0)
    if tf_period == 0: tf_period = safe_int(params.get('trendFilterPeriod'), 0)

    try:
        df.ta.rsi(length=14, append=True, col_names=('RSI_14',))
        df.ta.atr(length=14, append=True, col_names=('ATR_14',))
        df.ta.adx(length=14, append=True) 
        df.ta.sma(length=50, append=True, col_names=('SMA_50',))
        
        rsi_len = safe_int(params.get('rsi_length'), 14)
        if rsi_len != 14: df.ta.rsi(length=rsi_len, append=True, col_names=(f'RSI_{rsi_len}',))
        
        bb_len = safe_int(params.get('bb_length'), 20)
        bb_std = safe_float(params.get('bb_std'), 2.0)
        df.ta.bbands(length=20, std=2.0, append=True, col_names=('BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0'))
        if bb_len != 20 or bb_std != 2.0:
             df.ta.bbands(length=bb_len, std=bb_std, append=True, col_names=(f'BBL_{bb_len}_{bb_std}', f'BBM_{bb_len}_{bb_std}', f'BBU_{bb_len}_{bb_std}', f'BBB_{bb_len}_{bb_std}', f'BBP_{bb_len}_{bb_std}'))

        df.ta.macd(fast=12, slow=26, signal=9, append=True, col_names=('MACD_12_26_9', 'MACDH_12_26_9', 'MACDS_12_26_9'))
        df.ta.stoch(k=14, d=3, smooth_k=3, append=True, col_names=('STOCHk_14_3_3', 'STOCHd_14_3_3', 'STOCHs_14_3_3'))
        df.ta.cci(length=20, append=True, col_names=('CCI_20_0.015',))
        df.ta.psar(append=True) 
        df.ta.obv(append=True)
        if 'OBV' in df.columns: df['OBV_SMA_20'] = df['OBV'].rolling(window=20).mean()

        if tf_period > 0:
            sma_trend_col = f'SMA_{tf_period}'
            if sma_trend_col not in df.columns: df.ta.sma(length=tf_period, append=True, col_names=(sma_trend_col,))
    except Exception as e:
        logger.error(f"Feature Error: {e}")

    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    return df

# --- SIGNAL GENERATION ---
def generate_ta_signals(df: pd.DataFrame, strategy_code: str, params: dict = None) -> pd.DataFrame:
    df = df.copy()
    df['ta_signal'] = 0
    if params is None: params = {} 

    try:
        rsi_len = safe_int(params.get('rsi_length'), 14)
        bb_len = safe_int(params.get('bb_length'), 20)
        bb_std = safe_float(params.get('bb_std'), 2.0)
        
        rsi_col = find_col(df, f'RSI_{rsi_len}')
        bbl_col = find_col(df, f'BBL_{bb_len}_{bb_std}')
        bbu_col = find_col(df, f'BBU_{bb_len}_{bb_std}')
        psar_cols = [c for c in df.columns if c.startswith('PSAR')]
        
        df['close_prev'] = df['close'].shift(1)
        if rsi_col: df['rsi_prev'] = df[rsi_col].shift(1)

        if strategy_code == "psar_signal":
            found_psar = False
            for col in psar_cols:
                 if 'l' in col: df.loc[df[col].notna(), 'ta_signal'] = 1; found_psar = True
                 if 's' in col: df.loc[df[col].notna(), 'ta_signal'] = -1; found_psar = True
            if not found_psar and len(psar_cols) > 0:
                psar_col = psar_cols[0]
                df.loc[df['close'] > df[psar_col], 'ta_signal'] = 1
                df.loc[df['close'] < df[psar_col], 'ta_signal'] = -1

        elif strategy_code == "bollinger_bands" and bbl_col and bbu_col:
            df.loc[(df['close_prev'] < df[bbl_col].shift(1)) & (df['close'] >= df[bbl_col]), 'ta_signal'] = 1
            df.loc[(df['close_prev'] > df[bbu_col].shift(1)) & (df['close'] <= df[bbu_col]), 'ta_signal'] = -1
        
        elif strategy_code == "rsi_divergence" and rsi_col:
             oversold = safe_float(params.get('oversold_level'), 30)
             overbought = safe_float(params.get('overbought_level'), 70)
             df.loc[(df['rsi_prev'] < oversold) & (df[rsi_col] >= oversold), 'ta_signal'] = 1
             df.loc[(df['rsi_prev'] > overbought) & (df[rsi_col] <= overbought), 'ta_signal'] = -1

        elif strategy_code == "sma_crossover":
            f = safe_int(params.get('sma_fast_period'), 10)
            s = safe_int(params.get('sma_slow_period'), 50)
            sma_f = find_col(df, f'SMA_{f}')
            sma_s = find_col(df, f'SMA_{s}')
            if sma_f and sma_s:
                df.loc[(df[sma_f] > df[sma_s]) & (df[sma_f].shift(1) <= df[sma_s].shift(1)), 'ta_signal'] = 1
                df.loc[(df[sma_f] < df[sma_s]) & (df[sma_f].shift(1) >= df[sma_s].shift(1)), 'ta_signal'] = -1

        elif strategy_code == "stochastic_crossover":
             k_col = find_col(df, "STOCHk")
             d_col = find_col(df, "STOCHd")
             if k_col and d_col:
                 df.loc[(df[k_col] > df[d_col]) & (df[k_col].shift(1) <= df[d_col].shift(1)) & (df[k_col] < 20), 'ta_signal'] = 1
                 df.loc[(df[k_col] < df[d_col]) & (df[k_col].shift(1) >= df[d_col].shift(1)) & (df[k_col] > 80), 'ta_signal'] = -1
    except Exception: pass
    return df

# ==============================================================================
# 4. SAFE BACKTEST ENGINE (Fixed Risk Logic)
# ==============================================================================

def run_backtest(
    df: pd.DataFrame,
    signal_column: str,
    initial_balance: float,
    fee: float,
    stop_loss_pct: Optional[float],
    take_profit_pct: Optional[float],
    risk_mode: str,
    risk_percent: float,
    growth_target: Optional[float],
    min_atr_pct: Optional[float],
    trend_filter_period: Optional[int],
    ml_mode: str,
    min_adx_level: Optional[float],
    trailing_stop_atr_mult: Optional[float],
    trailing_stop_pct: Optional[float],
    max_leverage: float = MAX_LEVERAGE,
    **kwargs
) -> Dict[str, Any]:
    
    if signal_column not in df.columns: raise ValueError(f"Signal column '{signal_column}' not found.")
    
    atr_col = find_col(df, 'ATR_14')
    adx_col = find_col(df, 'ADX_14')
    tf_period = safe_int(trend_filter_period, 0)
    trend_col = find_col(df, f'SMA_{tf_period}') if tf_period > 0 else None

    balance = float(initial_balance)
    fee_dec = float(fee)
    slippage_pct_dec = float(SLIPPAGE_PCT)
    position = 0
    position_size = 0.0
    entry_price = 0.0
    entry_time = None
    equity_curve = []
    trades = []
    stop_loss_price = 0.0
    
    # 🚀 FIX: Initialize current_risk_percent correctly from arguments
    current_risk_percent = float(risk_percent) if risk_percent is not None else 1.0
    
    c_open = df['open'].to_numpy()
    c_high = df['high'].to_numpy()
    c_low = df['low'].to_numpy()
    c_close = df['close'].to_numpy()
    c_sig = df[signal_column].to_numpy()
    
    c_atr = df[atr_col].to_numpy(dtype=float) if atr_col else np.zeros(len(df))
    c_adx = df[adx_col].to_numpy(dtype=float) if adx_col else np.zeros(len(df))
    c_trend = df[trend_col].to_numpy(dtype=float) if trend_col else None

    min_atr = safe_float(min_atr_pct, 0)
    min_adx = safe_float(min_adx_level, 0)
    tsl_mult = safe_float(trailing_stop_atr_mult, 0)

    for i in range(1, len(df) - 1):
        if balance <= 0: break 

        curr_high = float(c_high[i])
        curr_low = float(c_low[i])
        curr_close = float(c_close[i])
        curr_atr = float(c_atr[i])
        curr_adx = float(c_adx[i])
        next_open = float(c_open[i+1])
        
        if position != 0:
            exit_price = 0.0
            pnl_reason = ""
            
            if tsl_mult > 0:
                if position == 1:
                    new_stop = curr_close - (curr_atr * tsl_mult)
                    stop_loss_price = max(stop_loss_price, new_stop)
                elif position == -1:
                    new_stop = curr_close + (curr_atr * tsl_mult)
                    stop_loss_price = min(stop_loss_price, new_stop) if stop_loss_price > 0 else new_stop

            if position == 1:
                if stop_loss_price > 0 and curr_low <= stop_loss_price:
                    exit_price = stop_loss_price
                    pnl_reason = "Stop Loss"
                elif c_sig[i] == -1:
                     exit_price = next_open * (1 - slippage_pct_dec)
                     pnl_reason = "Signal"
            elif position == -1:
                if stop_loss_price > 0 and curr_high >= stop_loss_price:
                    exit_price = stop_loss_price
                    pnl_reason = "Stop Loss"
                elif c_sig[i] == 1:
                    exit_price = next_open * (1 + slippage_pct_dec)
                    pnl_reason = "Signal"

            if exit_price > 0:
                if position == 1: exit_price = exit_price * (1 - slippage_pct_dec)
                else: exit_price = exit_price * (1 + slippage_pct_dec)

                if position == 1: gross_pnl = (exit_price - entry_price) * position_size
                else: gross_pnl = (entry_price - exit_price) * position_size
                
                fees = (entry_price * position_size * fee_dec) + (exit_price * position_size * fee_dec)
                net_pnl = gross_pnl - fees
                balance += net_pnl
                
                trades.append({
                    "action": "sell" if position == 1 else "cover",
                    "price": float(exit_price),
                    "time": df.index[i],
                    "size": float(position_size),
                    "profit_usd": float(net_pnl),
                    "reason": pnl_reason,
                    "entryPrice": float(entry_price), 
                    "position": "long" if position == 1 else "short",
                    "result": "win" if net_pnl > 0 else "loss"
                })
                position = 0

        if position == 0 and balance > 0:
            signal = c_sig[i]
            if min_atr > 0 and (curr_atr / (curr_close + 1e-9)) * 100 < min_atr: signal = 0
            if min_adx > 0 and curr_adx < min_adx: signal = 0
            if c_trend is not None:
                if (signal == 1 and curr_close < c_trend[i]) or (signal == -1 and curr_close > c_trend[i]): signal = 0

            if signal != 0:
                dist = 0.0
                if tsl_mult > 0: dist = curr_atr * tsl_mult
                else: dist = next_open * 0.02 
                if dist <= 0: dist = next_open * 0.01
                
                risk_amt = balance * (current_risk_percent / 100.0)
                raw_size = risk_amt / dist
                max_size = (balance * max_leverage) / next_open
                final_size = min(raw_size, max_size)
                
                if final_size * next_open > 10:
                    entry_time = df.index[i+1]
                    position = signal
                    position_size = float(final_size)
                    
                    if position == 1:
                        entry_price = next_open * (1 + slippage_pct_dec)
                        stop_loss_price = entry_price - dist
                        trades.append({"action": "buy", "price": float(entry_price), "time": entry_time, "size": float(position_size), "position": "long"})
                    else:
                        entry_price = next_open * (1 - slippage_pct_dec)
                        stop_loss_price = entry_price + dist
                        trades.append({"action": "sell_short", "price": float(entry_price), "time": entry_time, "size": float(position_size), "position": "short"})

        equity_curve.append({"timestamp": df.index[i], "balance": float(balance)})

    closed_trades = [t for t in trades if t.get('result')]
    wins = [t for t in closed_trades if t.get('result') == 'win']
    
    total_ret = ((float(balance) - initial_balance) / initial_balance) * 100
    eq_s = pd.Series([e['balance'] for e in equity_curve])
    cummax = eq_s.cummax()
    drawdown = (eq_s - cummax) / cummax
    max_dd = abs(drawdown.min()) * 100 if not drawdown.empty else 0.0
    
    calmar = 0
    if max_dd > 0:
         days = len(df) / 24 
         if "1d" in kwargs.get("timeframe", ""): days = len(df)
         years = days / 365.0
         if years > 0:
             ann_ret = ((balance / initial_balance) ** (1 / years)) - 1
             calmar = (ann_ret * 100) / max_dd

    return {
        "metrics": {
            "totalReturn": total_ret, 
            "maxDrawdown": max_dd,
            "totalTrades": len(closed_trades), 
            "finalBalance": float(balance),
            "winRate": (len(wins)/len(closed_trades)*100) if closed_trades else 0,
            "calmarRatio": calmar
        },
        "equityCurve": equity_curve,
        "tradeBreakdown": trades,
        "candleData": df.reset_index().rename(columns={'datetime':'timestamp'})[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')
    }

# ==============================================================================
# 5. LIVE PAPER TRADING BOT
# ==============================================================================

class PaperTradingBot:
    def __init__(self, config: dict):
        self.config = config
        self.symbol = config.get('symbol', 'BTC-USD')
        self.timeframe = config.get('timeframe', '1h')
        self.balance = float(config.get('capitalAllocation', 1000))
        self.initial_balance = self.balance
        self.position = 0 
        self.entry_price = 0.0
        self.position_size = 0.0
        self.stop_loss_price = 0.0
        self.is_running = False
        self.thread = None
        self.logs = []
        self.trades = []
        self.candles_buffer = []
        self.params = config.get('params', {})
        self.strategies = config.get('strategies', [])
        self.ml_mode = config.get('mlMode', 'off')
        self.ml_model = None
        
        if self.ml_mode in ['on', 'predictions'] and config.get('mlModel'):
             try:
                model_path = os.path.join(MODEL_DIR, f"{config.get('mlModel')}.joblib")
                if os.path.exists(model_path): self.ml_model = joblib.load(model_path)
             except: pass
             
        self.log(f"Bot Initialized. ${self.balance:.2f} on {self.symbol}.")

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.insert(0, {"timestamp": timestamp, "message": message})
        if len(self.logs) > 50: self.logs.pop()
        logger.info(f"[BOT] {message}")

    def start(self):
        if self.is_running: return
        self.is_running = True
        self.log("Bot Started.")
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        self.log("Bot Stopped.")
        if self.thread: self.thread.join(timeout=3)

    def get_status(self):
        return {
            "status": "running" if self.is_running else "stopped",
            "currentBalance": self.balance,
            "position": "LONG" if self.position == 1 else ("SHORT" if self.position == -1 else "FLAT"),
            "performanceMetrics": {
                "totalProfit": self.balance - self.initial_balance,
                "totalTrades": len(self.trades),
                "currentBalance": self.balance
            },
            "logs": self.logs,
            "candles": self.candles_buffer,
            "trades": self.trades
        }

    def _fetch_candles(self):
        granularity = 3600 
        if '4h' in self.timeframe: granularity = 14400
        if '1d' in self.timeframe: granularity = 86400
        url = f"https://api.exchange.coinbase.com/products/{self.symbol}/candles?granularity={granularity}"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                data = res.json()
                df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('datetime', inplace=True)
                df.sort_index(inplace=True)
                return df
        except: return pd.DataFrame()
        return pd.DataFrame()

    def _calculate_signal(self, df):
        df = engineer_features_for_backtest(df, self.params.get('trendFilterPeriod'), self.params)
        
        # Placeholder ML logic
        ml_signal = 0
        if self.ml_mode == 'on': return ml_signal, df.iloc[-1]

        ta_signal = 0
        hybrid_mode = self.params.get('hybridMode', 'AND')
        sigs = []
        for strat in self.strategies or []:
            code = strat.get('code')
            if code:
                temp_df = generate_ta_signals(df, code, self.params)
                sigs.append(temp_df['ta_signal'].iloc[-1])
        
        if sigs:
            if hybrid_mode == 'REGIME' and len(sigs) >= 2:
                adx = df['ADX_14'].iloc[-1]
                thresh = safe_float(self.params.get('regime_threshold'), 25)
                ta_signal = sigs[0] if adx > thresh else sigs[1]
            elif hybrid_mode == 'AND':
                ta_signal = 1 if all(s==1 for s in sigs) else (-1 if all(s==-1 for s in sigs) else 0)
            else:
                ta_signal = 1 if any(s==1 for s in sigs) else (-1 if any(s==-1 for s in sigs) else 0)
        
        return ta_signal, df.iloc[-1]

    def _run_loop(self):
        while self.is_running:
            try:
                df = self._fetch_candles()
                if not df.empty:
                    export_df = df.copy()
                    export_df['time'] = export_df.index.astype(np.int64) // 10**6 
                    self.candles_buffer = export_df.tail(100)[['time', 'open', 'high', 'low', 'close']].to_dict('records')

                    signal, last_candle = self._calculate_signal(df)
                    current_price = float(last_candle['close'])
                    curr_atr = float(last_candle.get('ATR_14', 0))
                    
                    if self.position != 0:
                        tsl_mult = safe_float(self.params.get('tslAtrMult'), 0)
                        if tsl_mult > 0:
                            if self.position == 1:
                                new_stop = current_price - (curr_atr * tsl_mult)
                                self.stop_loss_price = max(self.stop_loss_price, new_stop)
                            else:
                                new_stop = current_price + (curr_atr * tsl_mult)
                                self.stop_loss_price = min(self.stop_loss_price, new_stop) if self.stop_loss_price > 0 else new_stop
                        
                        if self.position == 1 and current_price <= self.stop_loss_price:
                            self._close(self.stop_loss_price, "Stop Loss")
                        elif self.position == -1 and current_price >= self.stop_loss_price:
                            self._close(self.stop_loss_price, "Stop Loss")
                        elif (self.position == 1 and signal == -1) or (self.position == -1 and signal == 1):
                            self._close(current_price, "Signal Flip")

                    if self.position == 0 and signal != 0:
                        self._execute(signal, current_price, curr_atr)

                time.sleep(10)
            except Exception as e:
                self.log(f"Error: {e}")
                time.sleep(10)

    def _execute(self, side, price, atr):
        risk_pct = 0.01
        tsl_mult = safe_float(self.params.get('tslAtrMult'), 0)
        dist = (atr * tsl_mult) if tsl_mult > 0 else (price * 0.02)
        if dist == 0: dist = price * 0.01
        
        risk_amt = self.balance * risk_pct
        raw_size = risk_amt / dist
        max_size = (self.balance * 2) / price
        size = min(raw_size, max_size)
        
        if size * price < 10: 
            self.log("Ignored: Size too small")
            return

        self.entry_price = price
        self.position = side
        self.position_size = size
        if side == 1: self.stop_loss_price = price - dist
        else: self.stop_loss_price = price + dist
        
        self.log(f"OPEN {'LONG' if side==1 else 'SHORT'} @ ${price:.2f}")

    def _close(self, price, reason):
        gross = (price - self.entry_price) * self.position_size if self.position == 1 else (self.entry_price - price) * self.position_size
        fees = (self.entry_price + price) * self.position_size * DEFAULT_TAKER_FEE
        net = gross - fees
        self.balance += net
        self.trades.append({"pnl": net, "reason": reason})
        self.log(f"CLOSE ({reason}). PnL: ${net:.2f}")
        self.position = 0

active_bot = None

@app.post('/api/bot/start')
async def start_bot(config: BotStartConfig):
    global active_bot
    if active_bot and active_bot.is_running: return {"status": "already_running"}
    active_bot = PaperTradingBot(config.dict())
    active_bot.start()
    return {"status": "started"}

@app.post('/api/bot/stop')
async def stop_bot():
    global active_bot
    if active_bot: active_bot.stop()
    return {"status": "stopped"}

@app.get('/api/bot/status')
async def bot_status():
    global active_bot
    if active_bot: return active_bot.get_status()
    return {"status": "stopped"}

@app.get("/api/ml/available-models", response_model=List[Dict[str, Any]])
def list_models():
    try: return [{"id": f.split('.')[0], "name": f.split('.')[0]} for f in os.listdir(MODEL_DIR) if f.endswith('.joblib')]
    except: return []

@app.get("/api/ml/latest-winner")
async def get_latest_winner():
    try:
        files = [os.path.join(RESULTS_DIR, f) for f in os.listdir(RESULTS_DIR) if f.startswith("winner_")]
        if not files: return JSONResponse(content={})
        latest = max(files, key=os.path.getmtime)
        with open(latest, 'r') as f: return JSONResponse(content=json.load(f))
    except: return JSONResponse(content={})

@app.get("/api/ml/winners")
async def list_all_winners():
    try:
        winners = []
        if not os.path.exists(RESULTS_DIR): return []
        files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("winner_")]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(RESULTS_DIR, x)), reverse=True)
        for f in files:
            with open(os.path.join(RESULTS_DIR, f), 'r') as file:
                data = json.load(file)
            name = data.get('combo_strategies', 'Unknown').replace(',', ' + ')
            winners.append({"id": f, "name": name, "config": data})
        return winners
    except: return []

@app.post('/api/ml/run-combo-backtest')
async def handle_run_combo_backtest(config: BacktestConfig):
    try:
        params = config.params or {}
        strategies_config = config.strategies or []
        norm_params = normalize_params(strategies_config, params)
        df = load_efficient_data(config.symbol, config.timeframe, config.startDate, config.endDate)
        df = engineer_features_for_backtest(df, norm_params.get('trendFilterPeriod'), norm_params)
        
        hybrid = norm_params.get('hybridMode', 'AND')
        sigs = []
        for i, s in enumerate(config.strategies or []):
             df = generate_ta_signals(df, s['code'], norm_params)
             df[f's_{i}'] = df['ta_signal']
             sigs.append(f's_{i}')
        
        if len(sigs)>=2 and hybrid=='REGIME':
             df['comb'] = np.where(df['ADX_14']>safe_float(norm_params.get('regime_threshold'), 25), df[sigs[0]], df[sigs[1]])
        elif hybrid=='AND': df['comb'] = df[sigs].apply(lambda r: 1 if (r==1).all() else (-1 if (r==-1).all() else 0), axis=1)
        else: df['comb'] = df[sigs].apply(lambda r: 1 if (r==1).any() else (-1 if (r==-1).any() else 0), axis=1)
        
        res = run_backtest(
            df, 'comb', config.initialBalance, config.fee,
            None, None, 'standard', config.riskPercentage,
            None, norm_params.get('minAtrPct'), norm_params.get('trendFilterPeriod'), config.mlMode,
            norm_params.get('minAdxLevel'), norm_params.get('tslAtrMult'), None,
            timeframe=config.timeframe 
        )
        return JSONResponse(content={"combinedResult": convert_numpy_types(res)})
    except Exception as e: 
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))

@app.post('/api/ml/certify-strategy')
async def handle_certify_strategy(config: CertifyConfig):
    try:
        df = load_efficient_data(config.symbol, config.timeframe, '2017-01-01', '2099-12-31')
        start = (df.index[-1] - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
        end = df.index[-1].strftime("%Y-%m-%d")
        df_h = df.loc[start:end].copy()
        
        params = config.best_params.get('params', {})
        df_h = engineer_features_for_backtest(df_h, params.get('trendFilterPeriod'), params)
        
        strategies = []
        if 'combo_strategies' in config.best_params:
            codes = config.best_params['combo_strategies'].split(',')
            for c in codes: strategies.append({'code': c, 'params': params})
        
        if strategies:
            sigs = []
            for i, s in enumerate(strategies):
                 df_h = generate_ta_signals(df_h, s['code'], params)
                 df_h[f's_{i}'] = df_h['ta_signal']
                 sigs.append(f's_{i}')
            
            hybrid = config.best_params.get('hybridMode', 'REGIME')
            if len(sigs)>=2 and hybrid=='REGIME':
                 df_h['comb'] = np.where(df_h['ADX_14']>safe_float(params.get('regime_threshold'), 25), df_h[sigs[0]], df_h[sigs[1]])
            elif hybrid=='AND': 
                 df_h['comb'] = df_h[sigs].apply(lambda r: 1 if (r==1).all() else (-1 if (r==-1).all() else 0), axis=1)
            else: 
                 df_h['comb'] = df_h[sigs].apply(lambda r: 1 if (r==1).any() else (-1 if (r==-1).any() else 0), axis=1)
            
            res = run_backtest(df_h, 'comb', 1000, 0.001, None, None, 'standard', 1.0, None, None, None, 'off', None, params.get('tslAtrMult'), None)
            metrics = res['metrics']
        else:
            metrics = {}
            res = {}
        
        return JSONResponse(content=convert_numpy_types({
            "certification_passed": metrics.get('totalTrades', 0) > 5, 
            "holdout_metrics": metrics,
            "holdout_equity_curve": res.get('equityCurve', [])
        }))
    except Exception as e: 
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
