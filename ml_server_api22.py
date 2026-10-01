# File: ml_server_api19.py
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
# 🚀 UPGRADE: Standard Floats for Speed/Compatibility
os.environ['PYTHONHASHSEED'] = '12345'
np.random.seed(12345)
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/root/Project/ML/"
MODEL_DIR =  "/root/Project/ML/app/models/"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OPTIMIZER_CACHE_DIR = os.path.join(DATA_DIR, "optimizer_cache")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OPTIMIZER_CACHE_DIR, exist_ok=True)

# TRADING CONSTANTS (Coinbase Advanced)
RISK_FREE_RATE = 0.02
MAX_LEVERAGE = 10
SLIPPAGE_PCT = 0.001        # 0.1% estimated slippage
DEFAULT_TAKER_FEE = 0.006   # 0.6% (Coinbase Tier <$10k)
PRICE_PRECISION = 2         # BTC-USD is 2 decimals
SIZE_PRECISION = 8          # BTC is 8 decimals
EPSILON = 1e-9

app = FastAPI(
    title="Trading ML Server API v3.5 (Complete)",
    description="Backtesting, Certification, and Live Coinbase Paper Trading",
    version="3.5.0" 
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

# 🚀 FIX: Added Missing Bot Configuration Model
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
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)): return default
        return int(float(val))
    except: return default

def safe_float(val, default=0.0):
    try:
        if val is None: return default
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)): return default
        return float(val)
    except: return default

def set_nested_value(d, keys, value):
    keys = keys.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def find_col(df, key, exclude=None):
    key_upper = key.upper()
    exclude_upper = exclude.upper() if exclude else ''
    found_cols = []
    for col in df.columns:
        col_upper = col.upper()
        if exclude_upper and exclude_upper in col_upper: continue
        if col_upper == key_upper or col_upper.startswith(key_upper + '_') or col_upper.startswith(key_upper + '.'):
            found_cols.append(col)
    if not found_cols:
        for col in df.columns:
            col_upper = col.upper()
            if exclude_upper and exclude_upper in col_upper: continue
            if key_upper in col_upper: found_cols.append(col)
    if not found_cols: return None
    if len(found_cols) > 1: found_cols.sort(key=len)
    return found_cols[0]

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
        if 'bollinger' in code or 'bb' in code:
            if 'period' in p: merged['bb_length'] = safe_int(p['period'])
            if 'stdDev' in p: merged['bb_std'] = safe_float(p['stdDev'])
            if 'bb_length' in p: merged['bb_length'] = safe_int(p['bb_length'])
            if 'bb_std' in p: merged['bb_std'] = safe_float(p['bb_std'])
        if 'rsi' in code:
            if 'period' in p: merged['rsi_length'] = safe_int(p['period'])
            if 'rsi_length' in p: merged['rsi_length'] = safe_int(p['rsi_length'])
        if 'sma' in code or 'ema' in code:
            if 'shortPeriod' in p: merged['sma_fast_period'] = safe_int(p['shortPeriod'])
            if 'longPeriod' in p: merged['sma_slow_period'] = safe_int(p['longPeriod'])
            if 'sma_fast_period' in p: merged['sma_fast_period'] = safe_int(p['sma_fast_period'])
            if 'sma_slow_period' in p: merged['sma_slow_period'] = safe_int(p['sma_slow_period'])
        if 'stoch' in code:
            if 'kPeriod' in p: merged['k_period'] = safe_int(p['kPeriod'])
            if 'dPeriod' in p: merged['d_period'] = safe_int(p['dPeriod'])
            if 'k_period' in p: merged['k_period'] = safe_int(p['k_period'])
            if 'd_period' in p: merged['d_period'] = safe_int(p['d_period'])
        if 'cci' in code:
            if 'period' in p: merged['cci_length'] = safe_int(p['period'])
            if 'cci_length' in p: merged['cci_length'] = safe_int(p['cci_length'])
    return merged

# --- FEATURE ENGINEERING ---
def engineer_features_for_backtest(df: pd.DataFrame, trend_filter_period: typing.Optional[int], params: dict = None) -> pd.DataFrame:
    df = df.copy()
    if df['close'].isnull().any(): df['close'] = df['close'].fillna(method='ffill')
    df.replace([np.inf, -np.inf], np.nan, inplace=True) 
    if params is None: params = {} 

    try:
        df.ta.rsi(length=14, append=True, col_names=('RSI_14',))
        df.ta.macd(fast=12, slow=26, signal=9, append=True, col_names=('MACD_12_26_9', 'MACDH_12_26_9', 'MACDS_12_26_9'))
        df.ta.stoch(k=14, d=3, smooth_k=3, append=True, col_names=('STOCHk_14_3_3', 'STOCHd_14_3_3', 'STOCHs_14_3_3'))
        df.ta.cci(length=20, append=True, col_names=('CCI_20_0.015',))
        df.ta.bbands(length=20, std=2.0, append=True, col_names=('BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0'))
        df.ta.atr(length=14, append=True, col_names=('ATR_14',))
        df.ta.sma(length=10, append=True, col_names=('SMA_10',))
        df.ta.sma(length=50, append=True, col_names=('SMA_50',))
        df.ta.sma(length=200, append=True, col_names=('SMA_200',))
        df.ta.psar(append=True) 
        df.ta.adx(length=14, append=True) 
        df.ta.obv(append=True)
        if 'OBV' in df.columns: df['OBV_SMA_20'] = df['OBV'].rolling(window=20).mean()
        
        rsi_len = safe_int(params.get('rsi_length'), 14)
        if rsi_len != 14: df.ta.rsi(length=rsi_len, append=True, col_names=(f'RSI_{rsi_len}',))
        
        bb_len = safe_int(params.get('bb_length'), 20)
        bb_std = safe_float(params.get('bb_std'), 2.0)
        if bb_len != 20 or bb_std != 2.0:
             df.ta.bbands(length=bb_len, std=bb_std, append=True, col_names=(f'BBL_{bb_len}_{bb_std}', f'BBM_{bb_len}_{bb_std}', f'BBU_{bb_len}_{bb_std}', f'BBB_{bb_len}_{bb_std}', f'BBP_{bb_len}_{bb_std}'))

        tf_period = safe_int(trend_filter_period, 0)
        if tf_period > 0:
            sma_trend_col = f'SMA_{tf_period}'
            if sma_trend_col not in df.columns: df.ta.sma(length=tf_period, append=True, col_names=(sma_trend_col,))

    except Exception as e:
        pass 

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
                 if 'l' in col: 
                     df.loc[df[col].notna(), 'ta_signal'] = 1
                     found_psar = True
                 if 's' in col: 
                     df.loc[df[col].notna(), 'ta_signal'] = -1
                     found_psar = True
            if not found_psar and len(psar_cols) > 0:
                psar_col = psar_cols[0]
                df.loc[df['close'] > df[psar_col], 'ta_signal'] = 1
                df.loc[df['close'] < df[psar_col], 'ta_signal'] = -1

        elif strategy_code == "bollinger_bands" and bbl_col and bbu_col:
            df.loc[(df['close_prev'] >= df[bbl_col].shift(1)) & (df['close'] < df[bbl_col]), 'ta_signal'] = 1
            df.loc[(df['close_prev'] <= df[bbu_col].shift(1)) & (df['close'] > df[bbu_col]), 'ta_signal'] = -1
        
        elif strategy_code == "rsi_divergence" and rsi_col:
             oversold = safe_float(params.get('oversold_level'), 30)
             overbought = safe_float(params.get('overbought_level'), 70)
             df.loc[(df['rsi_prev'] >= oversold) & (df[rsi_col] < oversold), 'ta_signal'] = 1
             df.loc[(df['rsi_prev'] <= overbought) & (df[rsi_col] > overbought), 'ta_signal'] = -1

        elif strategy_code == "sma_crossover":
            sma_fast = find_col(df, 'SMA_10')
            sma_slow = find_col(df, 'SMA_50')
            if sma_fast and sma_slow:
                df.loc[(df[sma_fast] > df[sma_slow]) & (df[sma_fast].shift(1) <= df[sma_slow].shift(1)), 'ta_signal'] = 1
                df.loc[(df[sma_fast] < df[sma_slow]) & (df[sma_fast].shift(1) >= df[sma_slow].shift(1)), 'ta_signal'] = -1

    except Exception as e:
        pass
    return df

# ==============================================================================
# 4. SAFE BACKTEST ENGINE (Float-Based)
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
    trend_col = find_col(df, f'SMA_{trend_filter_period}') if trend_filter_period else None

    balance = float(initial_balance)
    equity = float(initial_balance)
    fee_dec = float(fee)
    slippage_pct_dec = float(SLIPPAGE_PCT)
    
    position = 0
    position_size = 0.0
    entry_price = 0.0
    entry_time = None
    equity_curve = []
    trades = []
    stop_loss_price = 0.0
    take_profit_price = 0.0
    current_risk_percent = float(risk_percent)
    
    c_open = df['open'].to_numpy()
    c_high = df['high'].to_numpy()
    c_low = df['low'].to_numpy()
    c_close = df['close'].to_numpy()
    c_sig = df[signal_column].to_numpy()
    c_atr = df[atr_col].to_numpy(dtype=float) if atr_col else np.zeros(len(df))
    c_adx = df[adx_col].to_numpy(dtype=float) if adx_col else np.zeros(len(df))
    c_trend = df[trend_col].to_numpy(dtype=float) if trend_col else None

    for i in range(1, len(df) - 1):
        if balance <= 0: break 

        curr_high, curr_low = float(c_high[i]), float(c_low[i])
        curr_close, curr_atr = float(c_close[i]), float(c_atr[i])
        curr_adx = float(c_adx[i])
        next_open = float(c_open[i+1])
        
        if position != 0:
            exit_price = 0.0
            pnl_reason = ""
            
            if position == 1:
                if trailing_stop_atr_mult:
                    new_stop = curr_close - (curr_atr * float(trailing_stop_atr_mult))
                    stop_loss_price = max(stop_loss_price, new_stop)
            elif position == -1:
                if trailing_stop_atr_mult:
                    new_stop = curr_close + (curr_atr * float(trailing_stop_atr_mult))
                    stop_loss_price = min(stop_loss_price, new_stop) if stop_loss_price else new_stop

            if position == 1:
                if curr_low <= stop_loss_price: exit_price, pnl_reason = stop_loss_price, "Stop Loss"
                elif take_profit_pct and curr_high >= take_profit_price: exit_price, pnl_reason = take_profit_price, "Take Profit"
                elif c_sig[i] == -1: exit_price, pnl_reason = next_open * (1 - SLIPPAGE_PCT), "Signal"
            elif position == -1:
                if curr_high >= stop_loss_price: exit_price, pnl_reason = stop_loss_price, "Stop Loss"
                elif take_profit_pct and curr_low <= take_profit_price: exit_price, pnl_reason = take_profit_price, "Take Profit"
                elif c_sig[i] == 1: exit_price, pnl_reason = next_open * (1 + SLIPPAGE_PCT), "Signal"

            if exit_price > 0:
                if position == 1: exit_price = exit_price * (1 - SLIPPAGE_PCT)
                else: exit_price = exit_price * (1 + SLIPPAGE_PCT)

                if position == 1: gross_pnl = (exit_price - entry_price) * position_size
                else: gross_pnl = (entry_price - exit_price) * position_size
                
                fees = (entry_price * position_size * fee) + (exit_price * position_size * fee)
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
                    "entryTime": entry_time, 
                    "position": "long" if position == 1 else "short",
                    "result": "win" if net_pnl > 0 else "loss"
                })
                position = 0

        if position == 0 and balance > 0:
            signal = c_sig[i]
            if min_atr_pct and min_atr_pct > 0:
                if (curr_atr / (curr_close + 1e-9)) * 100 < min_atr_pct: signal = 0
            if min_adx_level and min_adx_level > 0:
                if curr_adx < min_adx_level: signal = 0
            if trend_filter_period and c_trend is not None:
                if (signal == 1 and curr_close < c_trend[i]) or (signal == -1 and curr_close > c_trend[i]):
                    signal = 0

            if signal != 0 and next_open > 0:
                risk_amt = balance * (current_risk_percent / 100.0)
                dist = 0.0
                if trailing_stop_atr_mult: dist = curr_atr * float(trailing_stop_atr_mult)
                else: dist = next_open * 0.02 
                if dist <= 0: dist = next_open * 0.01
                
                raw_size = risk_amt / dist
                max_size = (balance * max_leverage) / next_open
                final_size = min(raw_size, max_size)
                final_size = float(int(final_size * 1e8)) / 1e8 

                if final_size > 0:
                    entry_time = df.index[i+1]
                    position = signal
                    position_size = float(final_size)
                    
                    if position == 1:
                        entry_price = next_open * (1 + SLIPPAGE_PCT)
                        stop_loss_price = entry_price - dist
                        if take_profit_pct: take_profit_price = entry_price * (1 + take_profit_pct/100)
                        trades.append({"action": "buy", "price": float(entry_price), "time": entry_time, "entryTime": entry_time, "size": float(position_size), "position": "long"})
                    else:
                        entry_price = next_open * (1 - SLIPPAGE_PCT)
                        stop_loss_price = entry_price + dist
                        if take_profit_pct: take_profit_price = entry_price * (1 - take_profit_pct/100)
                        trades.append({"action": "sell_short", "price": float(entry_price), "time": entry_time, "entryTime": entry_time, "size": float(position_size), "position": "short"})

        equity_curve.append({"timestamp": df.index[i], "balance": float(balance)})

    closed_trades = [t for t in trades if t.get('result')]
    wins = [t for t in closed_trades if t.get('result') == 'win']
    losses = [t for t in closed_trades if t.get('result') == 'loss']
    
    total_ret = ((float(balance) - initial_balance) / initial_balance) * 100
    win_rate = (len(wins) / len(closed_trades)) * 100 if closed_trades else 0
    prof_factor = (sum(t['profit_usd'] for t in wins) / abs(sum(t['profit_usd'] for t in losses))) if losses else 999
    
    eq_s = pd.Series([e['balance'] for e in equity_curve])
    peak = eq_s.cummax()
    dd = (eq_s - peak) / peak
    max_dd = abs(dd.min()) * 100 if not dd.empty else 0
    
    sharpe = 0
    calmar = 0
    if len(equity_curve) > 1:
         dr = eq_s.pct_change().fillna(0)
         if dr.std() > 0:
             ann_ret = ((balance / initial_balance) ** (365.25 / (len(df)/24))) - 1
             sharpe = (ann_ret - RISK_FREE_RATE) / (dr.std() * np.sqrt(365.25))
             calmar = (ann_ret * 100) / (max_dd + 1e-9)

    return {
        "metrics": {
            "totalReturn": total_ret, "profitFactor": prof_factor, "maxDrawdown": max_dd,
            "winRate": win_rate, "totalTrades": len(closed_trades), "finalBalance": float(balance),
            "winningTrades": len(wins), "losingTrades": len(losses),
            "averageWin": sum(t['profit_usd'] for t in wins)/len(wins) if wins else 0,
            "averageLoss": abs(sum(t['profit_usd'] for t in losses))/len(losses) if losses else 0,
            "sharpeRatio": sharpe, "sortinoRatio": 0, "calmarRatio": calmar
        },
        "equityCurve": equity_curve,
        "tradeBreakdown": trades,
        "candleData": df.reset_index().rename(columns={'datetime':'timestamp'})[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')
    }

# ==============================================================================
# 5. LIVE PAPER TRADING BOT ENGINE
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
        self.is_running = False
        self.thread = None
        self.logs = []
        self.trades = []
        self.params = config.get('params', {})
        self.strategies = config.get('strategies', [])
        
        self.ml_mode = config.get('mlMode', 'off')
        self.ml_model = None
        if self.ml_mode in ['on', 'predictions'] and config.get('mlModel'):
             try:
                model_path = os.path.join(MODEL_DIR, f"{config.get('mlModel')}.joblib")
                if os.path.exists(model_path):
                    self.ml_model = joblib.load(model_path)
                    self.log(f"Loaded ML Model: {config.get('mlModel')}")
             except Exception as e:
                self.log(f"Failed to load ML model: {e}")

        self.log(f"Bot Initialized. ${self.balance} on {self.symbol}.")

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.insert(0, {"timestamp": timestamp, "message": message})
        if len(self.logs) > 50: self.logs.pop()
        logger.info(f"[BOT] {message}")

    def start(self):
        if self.is_running: return
        self.is_running = True
        self.log("Bot Started. Polling Coinbase...")
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        self.log("Bot Stopped.")
        if self.thread: self.thread.join(timeout=3)

    def get_status(self):
        return {
            "status": "running" if self.is_running else "stopped",
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "currentBalance": self.balance,
            "position": "LONG" if self.position == 1 else ("SHORT" if self.position == -1 else "FLAT"),
            "performanceMetrics": {
                "totalProfit": self.balance - self.initial_balance,
                "totalTrades": len(self.trades),
                "winRate": 0, 
                "currentBalance": self.balance
            },
            "logs": self.logs
        }

    def _fetch_candles(self):
        granularity = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}.get(self.timeframe, 3600)
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
        except Exception:
            return pd.DataFrame()
        return pd.DataFrame()

    def _calculate_signal(self, df):
        df = engineer_features_for_backtest(df, self.params.get('trendFilterPeriod'), self.params)
        
        ml_signal = 0
        if self.ml_model and self.ml_mode in ['on', 'predictions']:
            try:
                latest = df.iloc[[-1]].fillna(0)
                if hasattr(self.ml_model, 'predict_proba'):
                    prob = self.ml_model.predict_proba(latest)[0]
                    if prob[0] > self.config.get('mlThreshold', 0.65): ml_signal = 1
                    elif prob[2] > self.config.get('mlThreshold', 0.65): ml_signal = -1
            except: pass

        if self.ml_mode == 'on': return ml_signal

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
                thresh = self.params.get('regime_threshold', 25)
                ta_signal = sigs[0] if adx > thresh else sigs[1]
            elif hybrid_mode == 'AND':
                ta_signal = 1 if all(s==1 for s in sigs) else (-1 if all(s==-1 for s in sigs) else 0)
            else:
                ta_signal = 1 if any(s==1 for s in sigs) else (-1 if any(s==-1 for s in sigs) else 0)

        if self.ml_mode == 'predictions':
            if ta_signal == 1 and ml_signal == 1: return 1
            if ta_signal == -1 and ml_signal == -1: return -1
            return 0
            
        return ta_signal

    def _run_loop(self):
        while self.is_running:
            try:
                df = self._fetch_candles()
                if not df.empty:
                    signal = self._calculate_signal(df)
                    current_price = float(df['close'].iloc[-1])
                    
                    if self.position == 0 and signal != 0:
                        self._execute(signal, current_price)
                    elif self.position == 1 and signal == -1:
                        self._close(current_price)
                        self._execute(-1, current_price)
                    elif self.position == -1 and signal == 1:
                        self._close(current_price)
                        self._execute(1, current_price)
                        
                time.sleep(10)
            except Exception as e:
                self.log(f"Error: {e}")
                time.sleep(10)

    def _execute(self, side, price):
        self.entry_price = price
        self.position = side
        self.position_size = (self.balance * 0.99) / price
        self.log(f"OPEN {'LONG' if side==1 else 'SHORT'} @ ${price:.2f}")

    def _close(self, price):
        entry_val = self.entry_price * self.position_size
        exit_val = price * self.position_size
        gross_pnl = (price - self.entry_price) * self.position_size if self.position == 1 else (self.entry_price - price) * self.position_size
        fees = (entry_val + exit_val) * DEFAULT_TAKER_FEE
        net_pnl = gross_pnl - fees
        
        self.balance += net_pnl
        self.trades.append({"pnl": net_pnl})
        self.log(f"CLOSE. PnL: ${net_pnl:.2f}")
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

# --- ENDPOINTS (Backtest/Certify) ---
@app.get("/api/ml/available-models", response_model=List[Dict[str, Any]])
def list_models():
    models = []
    try:
        if not os.path.exists(MODEL_DIR):
            return []
        for filename in os.listdir(MODEL_DIR):
            if filename.startswith('.'): continue
            lower = filename.lower()
            if lower.endswith('.joblib') or lower.endswith('.pkl'):
                model_id = os.path.splitext(filename)[0]
                models.append({"id": model_id, "name": model_id, "file": filename})
    except Exception as e:
        logger.error(f"Error listing models: {e}")
    return models

@app.post('/api/ml/run-combo-backtest')
async def handle_run_combo_backtest(config: BacktestConfig):
    try:
        params = config.params or {}
        strategies_config = config.strategies or []
        
        # 🚀 FIX: Normalize Params
        normalized_params = normalize_params(strategies_config, params)
        
        df_features = load_efficient_data(config.symbol, config.timeframe, config.startDate, config.endDate)
        df_features = engineer_features_for_backtest(df_features, normalized_params.get('trendFilterPeriod'), normalized_params)
        
        hybrid_mode = normalized_params.get('hybridMode', 'AND')
        sigs = []
        for i, strat in enumerate(strategies_config):
            code = strat.get('code')
            if code:
                df_features = generate_ta_signals(df_features, code, normalized_params)
                df_features[f'sig_{i}'] = df_features['ta_signal']
                sigs.append(f'sig_{i}')
        
        if len(sigs) >= 2 and hybrid_mode == 'REGIME':
            thresh = normalized_params.get('regime_threshold', 25)
            df_features['combined_final'] = np.where(df_features['ADX_14'] > thresh, df_features[sigs[0]], df_features[sigs[1]])
        elif hybrid_mode == 'AND':
             df_features['combined_final'] = df_features[sigs].apply(lambda r: 1 if (r == 1).all() else (-1 if (r == -1).all() else 0), axis=1)
        else:
             df_features['combined_final'] = df_features[sigs].apply(lambda r: 1 if (r == 1).any() else (-1 if (r == -1).any() else 0), axis=1)
             
        res = run_backtest(
            df_features, 'combined_final', config.initialBalance, config.fee,
            normalized_params.get('SL'), normalized_params.get('TP'), config.riskManagementMode, config.riskPercentage,
            None, normalized_params.get('minAtrPct'), normalized_params.get('trendFilterPeriod'), config.mlMode,
            normalized_params.get('minAdxLevel'), normalized_params.get('tslAtrMult'), None
        )
        return JSONResponse(content={"combinedResult": convert_numpy_types(res)})
    except Exception as e:
        logger.error(f"Combo Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/ml/run-backtest-on')
async def handle_run_backtest_on(config: BacktestConfig):
    try:
        # Normalize
        normalized_params = normalize_params([{'code': config.code, 'params': config.params}], config.params)
        
        df = load_efficient_data(config.symbol, config.timeframe, config.startDate, config.endDate)
        df = engineer_features_for_backtest(df, normalized_params.get('trendFilterPeriod'), normalized_params)
        df = generate_ta_signals(df, config.code, normalized_params)
        res = run_backtest(
            df, 'ta_signal', config.initialBalance, config.fee,
            normalized_params.get('SL'), normalized_params.get('TP'), config.riskManagementMode, config.riskPercentage,
            None, normalized_params.get('minAtrPct'), normalized_params.get('trendFilterPeriod'), config.mlMode,
            normalized_params.get('minAdxLevel'), normalized_params.get('tslAtrMult'), None
        )
        return JSONResponse(content=convert_numpy_types(res))
    except Exception as e:
        logger.error(f"Single Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/ml/certify-strategy')
async def handle_certify_strategy(config: CertifyConfig):
    try:
        df_full = load_efficient_data(config.symbol, config.timeframe, '2017-01-01', '2099-12-31')
        holdout_config = deepcopy(config.base_config)
        holdout_config['startDate'] = (df_full.index[-1] - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
        holdout_config['endDate'] = df_full.index[-1].strftime("%Y-%m-%d")
        
        df_holdout = df_full.loc[holdout_config['startDate']:holdout_config['endDate']].copy()
        df_holdout = engineer_features_for_backtest(df_holdout, config.best_params.get('params', {}).get('trendFilterPeriod'), config.best_params.get('params', {}))
        
        ta_code = config.base_config.get('code')
        metrics = {}
        if ta_code:
             df_holdout = generate_ta_signals(df_holdout, ta_code, config.best_params.get('params', {}))
             res = run_backtest(df_holdout, 'ta_signal', 1000, 0.001, None, None, 'standard', 1.0, None, None, None, 'off', None, None, None)
             metrics = res['metrics']
        
        return JSONResponse(content=convert_numpy_types({
            "certification_passed": metrics.get('totalTrades', 0) > 5, 
            "holdout_metrics": metrics,
            "holdout_equity_curve": res.get('equityCurve', []) if ta_code else []
        }))

    except Exception as e:
        logger.error(f"Certify Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
