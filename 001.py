# File: ml.py
# 🚀 UPGRADE: v63.0 - "Bulletproof Server"
# Fixes: 500 Crash on Null Params, Corrupt File Handling, Safe Integers

import os
import json
import pandas as pd
import numpy as np
import joblib
import logging
import traceback
import math
import threading
import time
import glob
import yfinance as yf
import ccxt
import uuid
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/root/Project/ML"
MODEL_DIR = "/root/Project/ML/app/models"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "optimizer_results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_TAKER_FEE = 0.006   # 0.1% Fee
SLIPPAGE_PCT = 0.0010       # 0.05% Slippage

app = FastAPI(title="Trading ML Server API v63.0")

# Global Bot Instance
bot_instance = None

# --- 2. STARTUP CLEANUP ---
def cleanup_corrupt_data():
    try:
        csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        for f in csv_files:
            try:
                if os.path.getsize(f) < 50:
                    os.remove(f)
            except: pass
    except: pass

cleanup_corrupt_data()

# --- 3. MODELS ---
class StrategyConfig(BaseModel):
    code: str
    params: Dict[str, Any] = {}

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
    maxPyramiding: int = 1
    riskManagementMode: str = 'standard'
    riskPercentage: float = 1.0
    growthCapitalTarget: float = 2000.0

class BotConfig(BaseModel):
    symbol: str
    timeframe: str
    capitalAllocation: float
    mlMode: str = "off"
    mlModel: str = ""
    mlThreshold: float = 0.5
    isCombo: bool = False
    strategies: List[StrategyConfig] = []
    params: Dict[str, Any] = {}
    maxPyramiding: int = 1
    currentBalance: float = 0.0
    comboConfig: Dict[str, Any] = {}

# --- 4. HELPERS ---
def safe_int(val, default=0):
    """Safely converts value to int, handling None/Null."""
    if val is None: return default
    try: return int(float(val))
    except: return default

def safe_float(val, default=0.0):
    """Safely converts value to float, handling None/Null."""
    if val is None: return default
    try: return float(val)
    except: return default

def normalize_params(config):
    """Flattens nested strategy parameters into a single dictionary."""
    merged = {}
    if hasattr(config, 'params'): merged.update(config.params)
    if hasattr(config, 'strategies') and config.strategies:
        for strat in config.strategies:
            if isinstance(strat, dict): p = strat.get('params', {})
            else: p = strat.params
            for k, v in p.items(): merged[k] = v

    key_map = {
        'trend_filter_period': 'trendFilterPeriod',
        'min_adx_level': 'minAdxLevel', 'min_adx': 'minAdxLevel',
        'min_atr_pct': 'minAtrPct',
        'tsl_atr_mult': 'tslAtrMult', 'tsl_mult': 'tslAtrMult',
        'tp': 'TP', 'sl': 'SL'
    }
    final_params = deepcopy(merged)
    for k, v in merged.items():
        if k in key_map: final_params[key_map[k]] = v
    return final_params

def find_col(df, key_fragment):
    if key_fragment in df.columns: return key_fragment
    for col in df.columns:
        if col.lower() == key_fragment.lower(): return col
    for col in df.columns:
        if col.lower().startswith(key_fragment.lower()): return col
    return None

def normalize_symbol_ccxt(symbol):
    s = symbol.upper().replace('-', '/')
    if s.endswith('USDT'): return s.replace('USDT', 'USD')
    if '/' not in s:
        if s.endswith('USD'): return s[:-3] + '/USD'
        return s + '/USD'
    return s

def fetch_live_data(symbol, timeframe, limit=1000):
    try:
        ccxt_symbol = normalize_symbol_ccxt(symbol)
        exchange = ccxt.coinbase({'enableRateLimit': True}) 
        
        if not ('/' in ccxt_symbol or 'BTC' in symbol or 'ETH' in symbol):
             raise Exception("Likely Stock Symbol")

        timeframe_map = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '4h': '4h', '1d': '1d'}
        tf = timeframe_map.get(timeframe, '1h')
        
        ohlcv = exchange.fetch_ohlcv(ccxt_symbol, tf, limit=limit)
        
        if not ohlcv: raise Exception("CCXT returned empty data")

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('datetime', inplace=True)
        df.drop(columns=['timestamp'], inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df

    except Exception as e_ccxt:
        logger.warning(f"CCXT (Coinbase) Fetch Failed: {e_ccxt}. Fallback to YFinance.")
        try:
            clean_symbol = symbol.replace('/', '-')
            interval = {'1m':'1m','5m':'5m','15m':'15m','30m':'30m','1h':'1h','4h':'1h','1d':'1d'}.get(timeframe, '1h')
            df = yf.download(tickers=clean_symbol, period="5d", interval=interval, auto_adjust=False, progress=False, multi_level_index=False)
            if df.empty: return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'}, inplace=True)
            if 'close' not in df.columns and 'Close' in df.columns: df.rename(columns={'Close': 'close'}, inplace=True)
            df.index.name = 'datetime'
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            else: df.index = df.index.tz_convert('UTC')
            if not df.empty: df = df[:-1]
            return df
        except Exception as e_yf:
            logger.error(f"❌ All Data Sources Failed: {e_yf}")
            return pd.DataFrame()

def load_efficient_data(symbol, timeframe, start_date=None, end_date=None):
    safe_symbol = symbol.replace('/', '-')
    data_filename = f"{safe_symbol}-{timeframe}.csv"
    data_path = os.path.join(DATA_DIR, data_filename)

    should_fetch = True
    if os.path.exists(data_path) and os.path.getsize(data_path) > 1000:
        file_time = datetime.fromtimestamp(os.path.getmtime(data_path))
        if datetime.now() - file_time < timedelta(hours=1):
            should_fetch = False

    if should_fetch:
        try:
            df_live = fetch_live_data(symbol, timeframe, limit=1500)
            if not df_live.empty: df_live.to_csv(data_path)
        except: pass

    if os.path.exists(data_path) and os.path.getsize(data_path) > 50:
        try:
            df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
            if df.index.tz is not None: df.index = df.index.tz_convert('UTC')
            else: df.index = df.index.tz_localize('UTC')
            if start_date and end_date:
                try:
                    s_date = pd.to_datetime(start_date, utc=True) - pd.Timedelta(days=60) 
                    e_date = pd.to_datetime(end_date, utc=True)
                    if s_date <= df.index[-1] and e_date >= df.index[0]:
                        sliced_df = df.loc[s_date:e_date].copy()
                        if not sliced_df.empty: df = sliced_df
                except: pass
            df.rename(columns=lambda x: x.lower(), inplace=True)
            return df
        except: return pd.DataFrame()
    return pd.DataFrame()

def convert_numpy_types(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, float)): return float(obj) if not (math.isnan(obj) or math.isinf(obj)) else None
    if isinstance(obj, np.ndarray): return convert_numpy_types(obj.tolist())
    if isinstance(obj, (pd.Timestamp, datetime)): return obj.isoformat()
    if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy_types(v) for v in obj]
    return obj

# --- FEATURE ENGINEERING ---
def engineer_features_for_backtest(df, params):
    if params is None: params = {}
    try:
        df.ta.atr(length=14, append=True)
        if 'ATR_14' in df.columns: df['ATR'] = df['ATR_14']
        elif 'ATRr_14' in df.columns: df['ATR'] = df['ATRr_14']

        df.ta.adx(length=14, append=True)
        if 'ADX_14' in df.columns: df['ADX'] = df['ADX_14']

        df.ta.rsi(length=14, append=True)
        if 'RSI_14' in df.columns: df['RSI'] = df['RSI_14']

        df.ta.sma(length=50, append=True)
        trend_period = safe_int(params.get('trendFilterPeriod'), 200)
        if trend_period > 0: df.ta.sma(length=trend_period, append=True)
    except Exception as e: logger.error(f"Feature Engineering Error: {e}")
    return df

# ✅ STRATEGY REGISTRY
STRATEGY_MAP = {
    "psar_signal": lambda df, p: _strat_psar(df, p),
    "bollinger_bands": lambda df, p: _strat_bb(df, p),
    "rsi_divergence": lambda df, p: _strat_rsi(df, p),
    "sma_crossover": lambda df, p: _strat_sma(df, p),
    "macd_crossover": lambda df, p: _strat_macd(df, p),
    "stochastic_crossover": lambda df, p: _strat_stoch(df, p),
    "atr_breakout": lambda df, p: _strat_atr(df, p),
    "cci_oversold": lambda df, p: _strat_cci(df, p),
    "ichimoku_cloud": lambda df, p: _strat_ichimoku(df, p),
    "obv_signal": lambda df, p: _strat_obv(df, p)
}
def _strat_psar(df, p): df.ta.psar(step=safe_float(p.get('psar_step'),0.02), max_step=safe_float(p.get('psar_max'),0.2), append=True); return _apply_sig(df, 'PSARl', 'PSARs')
def _strat_bb(df, p): df.ta.bbands(length=safe_int(p.get('bb_length'),20), std=safe_float(p.get('bb_std'),2.0), append=True); return _apply_sig_bb(df, 'BBL', 'BBU')
def _strat_rsi(df, p): 
    len_ = safe_int(p.get('rsi_length'), 14)
    col = f"RSI_{len_}"
    if col not in df.columns: df.ta.rsi(length=len_, append=True)
    col = col if col in df.columns else 'RSI'
    if col in df.columns:
        df.loc[df[col] < safe_float(p.get('oversold_level'),30), 'ta_signal'] = 1
        df.loc[df[col] > safe_float(p.get('overbought_level'),70), 'ta_signal'] = -1
    return df
def _strat_sma(df, p):
    f, s = safe_int(p.get('sma_fast_period'), 10), safe_int(p.get('sma_slow_period'), 50)
    df.ta.sma(length=f, append=True); df.ta.sma(length=s, append=True)
    return _apply_cross(df, f'SMA_{f}', f'SMA_{s}')
def _strat_macd(df, p):
    df.ta.macd(fast=safe_int(p.get('macd_fast_period'),12), slow=safe_int(p.get('macd_slow_period'),26), signal=safe_int(p.get('macd_signal_period'),9), append=True)
    return _apply_cross(df, "MACD_", "MACDs_")
def _strat_stoch(df, p):
    df.ta.stoch(k=safe_int(p.get('k_period'),14), d=safe_int(p.get('d_period'),3), append=True)
    k, d = find_col(df, "STOCHk"), find_col(df, "STOCHd")
    if k and d:
        df.loc[(df[k]>df[d]) & (df[k]<20), 'ta_signal'] = 1
        df.loc[(df[k]<df[d]) & (df[k]>80), 'ta_signal'] = -1
    return df
def _strat_atr(df, p):
    l = safe_int(p.get('atr_period'), 14)
    m = safe_float(p.get('atr_multiplier'), 2.0)
    if f"ATR_{l}" not in df.columns: df.ta.atr(length=l, append=True)
    df.ta.ema(length=20, append=True)
    atr = find_col(df, f"ATR_{l}") or 'ATR'
    ema = "EMA_20"
    if atr in df.columns and ema in df.columns:
        df.loc[df['close'] > (df[ema] + df[atr]*m), 'ta_signal'] = 1
        df.loc[df['close'] < (df[ema] - df[atr]*m), 'ta_signal'] = -1
    return df
def _strat_cci(df, p):
    l = safe_int(p.get('cci_length'), 20)
    df.ta.cci(length=l, append=True)
    c = find_col(df, f"CCI_{l}") or 'CCI'
    if c in df.columns:
        df.loc[df[c] < safe_float(p.get('cci_oversold'),-100), 'ta_signal'] = 1
        df.loc[df[c] > safe_float(p.get('cci_overbought'),100), 'ta_signal'] = -1
    return df
def _strat_ichimoku(df, p): df.ta.ichimoku(append=True); return _apply_sig_ich(df)
def _strat_obv(df, p):
    df.ta.obv(append=True)
    ma = safe_int(p.get('obv_ma_period'), 20)
    if 'OBV' in df.columns:
        df[f'OBV_MA'] = df['OBV'].rolling(window=ma).mean()
        df.loc[df['OBV'] > df['OBV_MA'], 'ta_signal'] = 1
        df.loc[df['OBV'] < df['OBV_MA'], 'ta_signal'] = -1
    return df

def _apply_sig(df, c1, c2):
    c1, c2 = find_col(df, c1), find_col(df, c2)
    if c1 and c2:
        df.loc[df[c1].notna() & (df[c1]>0), 'ta_signal'] = 1
        df.loc[df[c2].notna() & (df[c2]>0), 'ta_signal'] = -1
    return df
def _apply_sig_bb(df, l, u):
    l, u = find_col(df, l), find_col(df, u)
    if l and u:
        df.loc[df['close']<=df[l], 'ta_signal'] = 1
        df.loc[df['close']>=df[u], 'ta_signal'] = -1
    return df
def _apply_cross(df, f, s):
    f, s = find_col(df, f), find_col(df, s)
    if f and s:
        df.loc[(df[f]>df[s]) & (df[f].shift(1)<=df[s].shift(1)), 'ta_signal'] = 1
        df.loc[(df[f]<df[s]) & (df[f].shift(1)>=df[s].shift(1)), 'ta_signal'] = -1
    return df
def _apply_sig_ich(df):
    a, b = find_col(df, "ISA_"), find_col(df, "ISB_")
    if a and b:
        df.loc[(df['close']>df[a]) & (df['close']>df[b]), 'ta_signal'] = 1
        df.loc[(df['close']<df[a]) & (df['close']<df[b]), 'ta_signal'] = -1
    return df

def generate_ta_signals(df, strategy_code, params, backtest_mode=False):
    df = df.copy()
    df['ta_signal'] = 0
    if strategy_code in STRATEGY_MAP:
        try: df = STRATEGY_MAP[strategy_code](df, params)
        except Exception as e: logger.error(f"Strategy Error: {e}")
    if backtest_mode: df['ta_signal'] = df['ta_signal'].shift(1).fillna(0)
    return df

# --- PYRAMID MANAGER ---
class PyramidManager:
    def __init__(self, capital, fee, slippage, max_levels=1, tsl_mult=0, mode='standard', risk_pct=1.0, growth_target=2000.0):
        self.total_capital = float(capital)
        self.available_cash = float(capital)
        self.fee = float(fee)
        self.base_slippage = float(slippage)
        self.max_levels = int(max_levels) if int(max_levels) > 0 else 1
        self.tsl_mult = float(tsl_mult) 
        self.mode = mode 
        self.risk_pct = float(risk_pct)
        self.growth_target = float(growth_target)
        self.positions = [] 
        self.completed_trades = []
        self.current_side = None 
        self.hourly_funding_rate = 0.0001 / 8 

    def check_risk_management(self, open_price, high, low, close, current_atr, time, candle_volume=0):
        if not self.positions: return
        for i in range(len(self.positions) - 1, -1, -1):
            pos = self.positions[i]
            val = pos['qty'] * close
            fees = val * self.hourly_funding_rate
            self.positions[i]['accumulated_fees'] = self.positions[i].get('accumulated_fees', 0) + fees
            closed = False
            reason = ""
            trig = 0.0
            if pos['side'] == 'long':
                if pos['sl_price'] > 0:
                    if open_price <= pos['sl_price']: closed=True; reason="SL_GAP"; trig=open_price
                    elif low <= pos['sl_price']: closed=True; reason="SL_HIT"; trig=pos['sl_price']
                if not closed and pos['tp_price'] > 0:
                    if open_price >= pos['tp_price']: closed=True; reason="TP_GAP"; trig=open_price
                    elif high >= pos['tp_price']: closed=True; reason="TP_HIT"; trig=pos['tp_price']
                if not closed and self.tsl_mult > 0 and current_atr > 0:
                    if high > pos['highest_seen']: self.positions[i]['highest_seen'] = high
                    stop = self.positions[i]['highest_seen'] - (current_atr * self.tsl_mult)
                    if open_price <= stop: closed=True; reason="TSL_GAP"; trig=open_price
                    elif low <= stop: closed=True; reason="TSL_HIT"; trig=stop
            elif pos['side'] == 'short':
                if pos['sl_price'] > 0:
                    if open_price >= pos['sl_price']: closed=True; reason="SL_GAP"; trig=open_price
                    elif high >= pos['sl_price']: closed=True; reason="SL_HIT"; trig=pos['sl_price']
                if not closed and pos['tp_price'] > 0:
                    if open_price <= pos['tp_price']: closed=True; reason="TP_GAP"; trig=open_price
                    elif low <= pos['tp_price']: closed=True; reason="TP_HIT"; trig=pos['tp_price']
                if not closed and self.tsl_mult > 0 and current_atr > 0:
                    if low < pos['lowest_seen']: self.positions[i]['lowest_seen'] = low
                    stop = self.positions[i]['lowest_seen'] + (current_atr * self.tsl_mult)
                    if open_price >= stop: closed=True; reason="TSL_GAP"; trig=open_price
                    elif high >= stop: closed=True; reason="TSL_HIT"; trig=stop
            if closed:
                slip = self.base_slippage
                if current_atr > 0 and open_price > 0 and (current_atr/open_price) > 0.01: slip *= 2.0
                self.close_specific_position(i, trig, time, reason, slip)

    def enter(self, side, price, time, current_atr=0, tp_pct=0.0, sl_pct=0.0, candle_volume=0):
        price = float(price)
        if self.current_side and self.current_side != side: self.close_all(price, time, "REVERSAL")
        if len(self.positions) >= self.max_levels: return
        current_equity = self.get_equity(price)
        allocation = 0.0
        if self.mode == 'dynamic':
            if current_equity < self.growth_target:
                slots = self.max_levels - len(self.positions)
                if slots > 0: allocation = self.available_cash / slots 
            else: allocation = current_equity * (self.risk_pct / 100.0)
        else:
            if self.risk_pct < 100: allocation = current_equity * (self.risk_pct / 100.0)
            else: 
                slots = self.max_levels - len(self.positions)
                if slots > 0: allocation = self.available_cash / slots
        if allocation > self.available_cash: allocation = self.available_cash
        if allocation < 1.0: return
        slip = self.base_slippage
        ep = price * (1 + slip) if side == 'long' else price * (1 - slip)
        qty = (allocation - (allocation * self.fee)) / ep
        tp = ep * (1 + tp_pct/100) if side == 'long' and tp_pct > 0 else (ep * (1 - tp_pct/100) if tp_pct > 0 else 0)
        sl = ep * (1 - sl_pct/100) if side == 'long' and sl_pct > 0 else (ep * (1 + sl_pct/100) if sl_pct > 0 else 0)
        self.positions.append({ "side": side, "entry_price": ep, "qty": qty, "entry_time": time, "invested": allocation, "tp_price": tp, "sl_price": sl, "highest_seen": price, "lowest_seen": price, "accumulated_fees": 0.0 })
        self.available_cash -= allocation
        self.current_side = side

    def close_specific_position(self, index, price, time, reason="SIGNAL", slippage_override=None):
        if index < 0 or index >= len(self.positions): return
        pos = self.positions.pop(index)
        price = float(price)
        slip = slippage_override if slippage_override is not None else self.base_slippage
        xp = price * (1 - slip) if pos['side'] == 'long' else price * (1 + slip)
        raw = pos['qty'] * xp
        if pos['side'] == 'short':
            diff = pos['entry_price'] - xp
            raw = pos['invested'] + (diff * pos['qty'])
        fees = (raw * self.fee) + pos.get('accumulated_fees', 0)
        net = raw - fees
        self.available_cash += net
        self.completed_trades.append({ "entryTime": pos['entry_time'], "exitTime": time, "price": pos['entry_price'], "exitPrice": xp, "profit": net - pos['invested'], "position": pos['side'], "type": reason })
        if not self.positions: self.current_side = None

    def close_all(self, price, time, reason="FORCE_CLOSE"):
        while self.positions: self.close_specific_position(0, price, time, reason)

    def get_equity(self, current_price):
        equity = self.available_cash
        for pos in self.positions:
             fees = pos.get('accumulated_fees', 0)
             if pos['side'] == 'long': val = pos['qty'] * current_price
             else: 
                 diff = pos['entry_price'] - current_price
                 val = pos['invested'] + (diff * pos['qty'])
             equity += val - fees
        return equity

# 🚀 UPGRADE: v63.0 - Gatekeeper Logic
def run_backtest(df, signal_col, initial_balance, fee, max_pyramiding=1, tsl_mult=0, params={}):
    mode = params.get('riskManagementMode', 'standard')
    risk_pct = safe_float(params.get('riskPercentage'), 100.0) 
    target = safe_float(params.get('growthCapitalTarget'), 2000.0)
    
    min_adx = safe_float(params.get('minAdxLevel'), 0.0)
    min_atr_pct = safe_float(params.get('minAtrPct'), 0.0)
    trend_period = safe_int(params.get('trendFilterPeriod'), 200)
    
    manager = PyramidManager(initial_balance, fee, SLIPPAGE_PCT, max_pyramiding, tsl_mult, mode, risk_pct, target)
    equity_curve = [{"timestamp": df.index[0].isoformat(), "balance": initial_balance}]
    df = engineer_features_for_backtest(df, params)
    
    tp_pct = safe_float(params.get('TP'), 0.0)
    sl_pct = safe_float(params.get('SL'), 0.0)

    trend_col = f'SMA_{trend_period}' if trend_period > 0 and f'SMA_{trend_period}' in df.columns else None

    for i in range(1, len(df)):
        row = df.iloc[i]
        O, H, L, C, V = float(row['open']), float(row['high']), float(row['low']), float(row['close']), float(row['volume'])
        time_ = df.index[i].isoformat()
        
        atr = float(row['ATR']) if 'ATR' in df.columns else 0.0
        adx = float(row['ADX']) if 'ADX' in df.columns else 0.0
        sig = df[signal_col].iloc[i]
        
        # 🚀 GATEKEEPER LOGIC
        if min_adx > 0 and adx < min_adx: sig = 0 
        
        if min_atr_pct > 0 and C > 0:
            atr_pct = (atr / C) * 100
            if atr_pct < min_atr_pct: sig = 0
                
        if trend_col and trend_col in df.columns:
            trend_val = float(row[trend_col])
            if sig == 1 and C < trend_val: sig = 0
            if sig == -1 and C > trend_val: sig = 0

        if sig == 1: manager.enter('long', O, time_, atr, tp_pct, sl_pct, V)
        elif sig == -1: manager.enter('short', O, time_, atr, tp_pct, sl_pct, V)
        
        manager.check_risk_management(O, H, L, C, atr, time_, V)
        equity_curve.append({"timestamp": time_, "balance": manager.get_equity(C)})

    if not df.empty: manager.close_all(df['close'].iloc[-1], df.index[-1].isoformat(), "END_OF_DATA")
    
    final = manager.available_cash
    trades = manager.completed_trades
    wins = [t for t in trades if t['profit'] > 0]
    gross_p = sum(t['profit'] for t in wins)
    gross_l = abs(sum(t['profit'] for t in trades if t['profit'] < 0))
    pf = (gross_p / gross_l) if gross_l > 0 else 999
    
    return {
        "metrics": {
            "totalReturn": ((final - initial_balance)/initial_balance)*100, "finalBalance": final, "totalTrades": len(trades),
            "winningTrades": len(wins), "losingTrades": len(trades) - len(wins),
            "winRate": (len(wins)/len(trades)*100) if trades else 0, "maxDrawdown": 0, 
            "profitFactor": pf, "averageWin": (gross_p/len(wins)) if wins else 0, "averageLoss": (gross_l/(len(trades)-len(wins))) if (len(trades)-len(wins)) > 0 else 0
        },
        "equityCurve": equity_curve, "tradeBreakdown": trades, 
        "candleData": df[['open','high','low','close','volume']].reset_index().to_dict('records')
    }

@app.post('/api/ml/run-backtest-on') 
@app.post('/api/ml/run-combo-backtest')
async def handle_run_combo_backtest(config: BacktestConfig):
    try:
        norm_params = normalize_params(config)
        df = load_efficient_data(config.symbol, config.timeframe, config.startDate, config.endDate)
        
        if df.empty: 
            empty_res = {"metrics": {}, "equityCurve": [], "tradeBreakdown": [], "candleData": []}
            return JSONResponse(content={**empty_res, "combinedResult": empty_res})
        
        df = engineer_features_for_backtest(df, norm_params)
        
        # 🚀 FIX: Prevent crash on empty strategies
        df['comb'] = 0 
        
        if config.code:
             df = generate_ta_signals(df, config.code, norm_params, backtest_mode=True)
             df['comb'] = df['ta_signal']
        elif config.strategies:
             sigs = []
             for i, s in enumerate(config.strategies):
                 s_code = s.get('code')
                 if not s_code: continue
                 
                 s_p = norm_params.copy(); s_p.update(s.get('params', {}))
                 df = generate_ta_signals(df, s_code, s_p, backtest_mode=True)
                 df[f's_{i}'] = df['ta_signal']
                 sigs.append(f's_{i}')
             
             mode = config.params.get("hybridMode", "OR")
             
             if len(sigs) >= 2:
                 if mode == "AND":
                     def combine_and(r): return 1 if all(r[c]==1 for c in sigs) else (-1 if all(r[c]==-1 for c in sigs) else 0)
                     df['comb'] = df.apply(combine_and, axis=1)
                 elif mode == "REGIME":
                     thr = safe_int(config.params.get("regime_threshold"), 25)
                     if 'ADX' in df.columns:
                         adx_s = df['ADX'].shift(1).fillna(0)
                         df['comb'] = np.where(adx_s > thr, df[sigs[0]], df[sigs[1]])
                     else: df['comb'] = 0
                 else:
                     def combine_or(r):
                         v = [r[c] for c in sigs]
                         if 1 in v and -1 not in v: return 1
                         if -1 in v and 1 not in v: return -1
                         return 0
                     df['comb'] = df.apply(combine_or, axis=1)
             elif len(sigs) == 1: 
                 df['comb'] = df[sigs[0]]

        # 🚀 FIX: Use safe_int to prevent crash on null
        p_lvl = safe_int(config.params.get("maxPyramiding"), 1)
        if p_lvl < 1: p_lvl = 1
        
        tsl = safe_float(config.params.get('tslAtrMult'), 0.0)
        
        params_g = config.params.copy()
        params_g['riskManagementMode'] = config.riskManagementMode
        params_g['riskPercentage'] = config.riskPercentage
        params_g['growthCapitalTarget'] = config.growthCapitalTarget

        res = run_backtest(df, 'comb', config.initialBalance, config.fee, p_lvl, tsl, params_g)
        clean = convert_numpy_types(res)
        resp = deepcopy(clean)
        resp["combinedResult"] = deepcopy(clean)
        return JSONResponse(content=resp)
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/ml/available-models")
def list_models():
    if not os.path.exists(MODEL_DIR): return []
    return [{"id": f.split('.')[0], "name": f.split('.')[0]} for f in os.listdir(MODEL_DIR) if f.endswith('.joblib')]

@app.get("/api/bot/winners")
def get_winners():
    if not os.path.exists(RESULTS_DIR): return []
    winners = []
    for f in os.listdir(RESULTS_DIR):
        if f.endswith(".json"):
            try:
                path = os.path.join(RESULTS_DIR, f)
                if os.path.getsize(path) == 0: continue
                with open(path, 'r') as file: data = json.load(file)
                if isinstance(data, list): data = { "strategies": data, "params": {}, "metrics": {} }
                
                display_name = f.replace('.json', '')
                score = 0.0
                if "SCORE" in display_name:
                    try:
                        parts = display_name.split('SCORE')
                        if len(parts) > 1: score = float(parts[1].split('_')[0])
                    except: pass
                
                if score == 0.0 and 'metrics' in data:
                    score = float(data['metrics'].get('calmar', 0.0) or data['metrics'].get('score', 0.0) or 0.0)

                winners.append({ "id": f, "name": display_name, "score": score, "config": data })
            except Exception as e: logger.warning(f"Skipping corrupt file {f}: {e}")
    winners.sort(key=lambda x: x['score'], reverse=True)
    return winners

class PaperTradingBot:
    def __init__(self, config: BotConfig):
        self.config = config
        self.is_running = False
        self.manager = PyramidManager(config.capitalAllocation, DEFAULT_TAKER_FEE, SLIPPAGE_PCT, config.maxPyramiding, float(config.params.get('tslAtrMult', 0.0)), 'standard', 100.0, 0.0)
        self.logs = []
        self.trades = []
        self.df = None
        self.add_log("info", f"Bot initialized for {config.symbol}")

    def add_log(self, type_, msg):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(), 
            "type": type_, "message": msg
        }
        self.logs.append(entry)
        if len(self.logs) > 200: self.logs.pop(0)

    def get_profit(self): return sum(t['profit'] for t in self.manager.completed_trades)
    def get_win_rate(self):
        if not self.manager.completed_trades: return 0
        wins = len([t for t in self.manager.completed_trades if t['profit'] > 0])
        return (wins / len(self.manager.completed_trades)) * 100

    def stop(self):
        self.is_running = False
        self.add_log("status", "Bot stopped.")

    def run(self):
        self.is_running = True
        self.add_log("status", f"Bot started with {self.config.symbol} on {self.config.timeframe} timeframe.")
        
        while self.is_running:
            try:
                df = fetch_live_data(self.config.symbol, self.config.timeframe, limit=200)
                if df.empty:
                    self.add_log("error", "Failed to fetch market data.")
                    time.sleep(10); continue
                
                self.df = df 
                norm_params = normalize_params(self.config)
                df = engineer_features_for_backtest(df, norm_params)
                
                if self.config.strategies:
                     sigs = []
                     for i, s in enumerate(self.config.strategies):
                         s_code = s.code if hasattr(s, 'code') else s.get('code')
                         s_p_raw = s.params if hasattr(s, 'params') else s.get('params', {})
                         s_p = norm_params.copy(); s_p.update(s_p_raw)
                         df = generate_ta_signals(df, s_code, s_p, backtest_mode=False)
                         df[f's_{i}'] = df['ta_signal']
                         sigs.append(f's_{i}')
                     
                     combo_conf = self.config.comboConfig if self.config.comboConfig else {}
                     mode = combo_conf.get("combinationRule", "OR")
                     
                     if len(sigs) >= 2:
                         if mode == "AND":
                             def combine_and(r): return 1 if all(r[c]==1 for c in sigs) else (-1 if all(r[c]==-1 for c in sigs) else 0)
                             df['comb'] = df.apply(combine_and, axis=1)
                         elif mode == "REGIME":
                             thr = safe_int(self.config.params.get("regime_threshold"), 25)
                             if 'ADX' in df.columns:
                                 adx_s = df['ADX'].fillna(0)
                                 df['comb'] = np.where(adx_s > thr, df[sigs[0]], df[sigs[1]])
                             else: df['comb'] = 0
                         else:
                             def combine_or(r):
                                 v = [r[c] for c in sigs]
                                 if 1 in v and -1 not in v: return 1
                                 if -1 in v and 1 not in v: return -1
                                 return 0
                             df['comb'] = df.apply(combine_or, axis=1)
                     elif sigs: df['comb'] = df[sigs[0]]
                     else: df['comb'] = 0 
                else:
                    self.add_log("error", "No strategies configured.")
                    time.sleep(10); continue

                latest = df.iloc[-1]
                sig = latest['comb']
                
                metrics = []
                if 'RSI_14' in df.columns: metrics.append(f"RSI: {latest['RSI_14']:.1f}")
                if 'ADX_14' in df.columns: metrics.append(f"ADX: {latest['ADX_14']:.1f}")
                metric_str = " | ".join(metrics)
                
                current_adx = latest['ADX'] if 'ADX' in df.columns else 0.0
                current_atr = latest['ATR'] if 'ATR' in df.columns else 0.0
                min_adx = safe_float(norm_params.get('minAdxLevel'), 0.0)
                min_atr_pct = safe_float(norm_params.get('minAtrPct'), 0.0)
                trend_period = safe_int(norm_params.get('trendFilterPeriod'), 200)
                
                if min_adx > 0 and current_adx < min_adx:
                    if sig != 0: self.add_log("info", f"Signal BLOCKED by ADX ({current_adx:.1f} < {min_adx})")
                    sig = 0
                
                if min_atr_pct > 0 and latest['close'] > 0:
                    atr_pct = (current_atr / latest['close']) * 100
                    if atr_pct < min_atr_pct:
                        if sig != 0: self.add_log("info", f"Signal BLOCKED by Volatility ({atr_pct:.2f}% < {min_atr_pct}%)")
                        sig = 0
                
                if trend_period > 0 and f'SMA_{trend_period}' in df.columns:
                    trend_val = latest[f'SMA_{trend_period}']
                    if sig == 1 and latest['close'] < trend_val:
                        self.add_log("info", f"Long BLOCKED by Trend (Price < SMA{trend_period})")
                        sig = 0
                    if sig == -1 and latest['close'] > trend_val:
                        self.add_log("info", f"Short BLOCKED by Trend (Price > SMA{trend_period})")
                        sig = 0

                if sig == 1: 
                    self.manager.enter('long', latest['close'], datetime.now().isoformat(), current_atr)
                    self.add_log("buy", f"Entered long at {latest['close']} | {metric_str}")
                elif sig == -1: 
                    self.manager.enter('short', latest['close'], datetime.now().isoformat(), current_atr)
                    self.add_log("sell", f"Entered short at {latest['close']} | {metric_str}")                else:
                    self.add_log("info", f"Checked combo signals. Final signal: hold. | {metric_str}")

                self.manager.check_risk_management(latest['open'], latest['high'], latest['low'], latest['close'], current_atr, datetime.now().isoformat())
                
                self.trades = self.manager.completed_trades
                time.sleep(60) 
                
            except Exception as e:
                self.add_log("error", f"Loop Error: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(10)

@app.post("/api/bot/start")
async def start_bot_endpoint(config: BotConfig):
    global bot_instance
    if bot_instance and bot_instance.is_running: return {"status": "error", "message": "Running"}
    bot_instance = PaperTradingBot(config)
    threading.Thread(target=bot_instance.run, daemon=True).start()
    return {"status": "started", "config": config}

@app.post("/api/bot/stop")
async def stop_bot_endpoint():
    global bot_instance
    if bot_instance: bot_instance.stop(); return {"status": "stopped"}
    return {"status": "not_running"}

@app.get("/api/bot/status")
async def get_bot_status_endpoint():
    global bot_instance
    st = {"status": "stopped", "logs": [], "trades": [], "activePositions": [], "currentBalance": 0, "candles": [], "performanceMetrics": {}}
    
    if bot_instance:
        st["status"] = "running" if bot_instance.is_running else "stopped"
        st["logs"] = list(bot_instance.logs[-50:])
        st["trades"] = list(bot_instance.manager.completed_trades)
        st["activePositions"] = list(bot_instance.manager.positions)
        st["currentBalance"] = float(bot_instance.manager.available_cash)
        
        gp = sum(t['profit'] for t in bot_instance.trades if t['profit']>0)
        gl = abs(sum(t['profit'] for t in bot_instance.trades if t['profit']<0))
        
        if bot_instance.df is not None and not bot_instance.df.empty:
            try:
                tail = bot_instance.df.tail(100).copy().reset_index()
                date_col = next((c for c in tail.columns if str(c).lower() in ['date','datetime','timestamp','index']), None)
                if date_col:
                    tail['time'] = tail[date_col].apply(lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x))
                    st["candles"] = tail[['time','open','high','low','close']].to_dict('records')
            except Exception as e: logger.error(f"Candle Serialization Error: {e}")
            
        st["performanceMetrics"] = {
            "totalProfit": bot_instance.get_profit(),
            "winRate": bot_instance.get_win_rate(),
            "totalTrades": len(bot_instance.trades),
            "profitFactor": (gp/gl) if gl > 0 else 0,
            "maxDrawdown": 0, 
            "currentBalance": float(bot_instance.manager.available_cash)
        }
    return st

@app.get("/api/bot/logs")
async def get_logs(limit: int = 100):
    return bot_instance.logs[-limit:] if bot_instance else []

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
(venv) root@intelligent-mendel:~/Project/ML#
