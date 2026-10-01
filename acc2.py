# File: ml.py
# 🚀 UPGRADE: v40.0 - "Market Reality" (CCXT Integration + Friction Simulation)

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
import ccxt  # 👈 NEW: Professional Crypto Data
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

DEFAULT_TAKER_FEE = 0.001   # 0.1% Fee (Standard Binance Taker)
SLIPPAGE_PCT = 0.0005       # 0.05% Slippage (Base)

app = FastAPI(title="Trading ML Server API v40.0")

# Global Bot Instance
bot_instance = None

# --- 2. STARTUP CLEANUP ---
def cleanup_corrupt_data():
    try:
        csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        for f in csv_files:
            try:
                # Delete empty files
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

# --- 4. HELPERS ---
def safe_int(val, default=0):
    try: return int(float(val))
    except: return default

def safe_float(val, default=0.0):
    try: return float(val)
    except: return default

def normalize_params(config):
    merged = {}
    if hasattr(config, 'params'): merged.update(config.params)
    if hasattr(config, 'strategies') and config.strategies:
        for strat in config.strategies:
            p = strat.params if hasattr(strat, 'params') else strat.get('params', {})
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

# 🚀 NEW: Symbol Normalizer (BTC-USD -> BTC/USDT)
def normalize_symbol_ccxt(symbol):
    # Basic mapping for common styles
    s = symbol.upper().replace('-', '/')
    if s.endswith('USD') and not s.endswith('USDT') and not s.endswith('USDC'):
        # CCXT usually prefers USDT for crypto pairs on major exchanges
        return s.replace('USD', '/USDT').replace('//', '/')
    if '/' not in s:
        # If it's just BTCUSDT
        if s.endswith('USDT'): return s[:-4] + '/USDT'
        if s.endswith('USD'): return s[:-3] + '/USD'
    return s

# ✅ FIX 4: CCXT Data Fetching (The Gold Standard)
def fetch_live_data(symbol, timeframe):
    # 1. Try CCXT (Binance) First for Crypto
    try:
        # Normalize symbol for CCXT (BTC-USD -> BTC/USDT)
        ccxt_symbol = normalize_symbol_ccxt(symbol)
        
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'} # Use Futures data for better volume/funding simulation
        })
        
        # If symbol looks like a stock (AAPL, TSLA), skip to YFinance
        if not ('/' in ccxt_symbol or 'BTC' in symbol or 'ETH' in symbol):
             raise Exception("Likely Stock Symbol")

        # Pagination Loop for History
        # Fetch last ~1000 days (approx 3 years)
        # 1h candles: 24 * 1000 = 24000 candles
        # Binance limit is 1500 per call.
        
        timeframe_map = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '4h': '4h', '1d': '1d'}
        tf = timeframe_map.get(timeframe, '1h')
        
        # Calculate Start Time (ms) - Go back 2 years for robustness
        start_time = exchange.milliseconds() - (2 * 365 * 24 * 60 * 60 * 1000)
        
        all_ohlcv = []
        current_since = start_time
        
        # Simple fetch loop (limit 3 calls to avoid heavy lags in this specific script version)
        # For production, run a background job to fetch full history.
        for _ in range(5): 
            ohlcv = exchange.fetch_ohlcv(ccxt_symbol, tf, since=current_since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 1 # Next ms
            
            # Safety break if we reached now
            if current_since > exchange.milliseconds(): break
        
        if not all_ohlcv:
            raise Exception("CCXT returned empty data")

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df.drop(columns=['timestamp'], inplace=True)
        
        # Ensure float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df

    except Exception as e_ccxt:
        logger.warning(f"CCXT Fetch Failed for {symbol} ({e_ccxt}). Falling back to YFinance.")
        
        # 2. Fallback to YFinance (Robust Version from v39)
        try:
            clean_symbol = symbol.replace('/', '-')
            tf_map = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '4h': '1h', '1d': '1d'}
            interval = tf_map.get(timeframe, '1h')

            period = "2y" # Safer default
            
            df = yf.download(
                tickers=clean_symbol, 
                period=period, 
                interval=interval, 
                auto_adjust=False, 
                progress=False,
                multi_level_index=False
            )

            if df.empty: return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adj_close'
            }, inplace=True)
            
            if 'close' not in df.columns and 'Close' in df.columns:
                 df.rename(columns={'Close': 'close'}, inplace=True)

            df.index.name = 'datetime'
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            else: df.index = df.index.tz_convert('UTC')

            if not df.empty: df = df[:-1]
            return df
            
        except Exception as e_yf:
            logger.error(f"❌ All Data Sources Failed for {symbol}: {e_yf}")
            return pd.DataFrame()

def load_efficient_data(symbol, timeframe, start_date=None, end_date=None):
    safe_symbol = symbol.replace('/', '-')
    data_filename = f"{safe_symbol}-{timeframe}.csv"
    data_path = os.path.join(DATA_DIR, data_filename)

    should_fetch = True
    if os.path.exists(data_path):
        # If file exists and is reasonably large, use it (Manual Override)
        if os.path.getsize(data_path) > 1000:
            file_time = datetime.fromtimestamp(os.path.getmtime(data_path))
            # Only refetch if older than 4 hours (Funding rates update every 8h)
            if datetime.now() - file_time < timedelta(hours=4):
                should_fetch = False

    if should_fetch:
        df_live = fetch_live_data(symbol, timeframe)
        if not df_live.empty: df_live.to_csv(data_path)

    if os.path.exists(data_path) and os.path.getsize(data_path) > 50:
        try:
            df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            df.index = df.index.tz_localize('UTC')

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
    if isinstance(obj, (np.floating, np.float64, np.float32, float)):
        if math.isnan(obj) or math.isinf(obj): return None
        return float(obj)
    if isinstance(obj, np.ndarray): return convert_numpy_types(obj.tolist())
    if isinstance(obj, (pd.Timestamp, datetime, np.datetime64)):
        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
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
        if trend_period > 0:
            df.ta.sma(length=trend_period, append=True)
    except Exception as e:
        logger.error(f"Feature Engineering Error: {e}")
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

def _strat_psar(df, params):
    step = safe_float(params.get('psar_step'), 0.02)
    max_step = safe_float(params.get('psar_max'), 0.2)
    df.ta.psar(step=step, max_step=max_step, append=True)
    psarl = find_col(df, "PSARl"); psars = find_col(df, "PSARs")
    if psarl and psars:
        df.loc[df[psarl].notna() & (df[psarl] > 0), 'ta_signal'] = 1
        df.loc[df[psars].notna() & (df[psars] > 0), 'ta_signal'] = -1
    return df

def _strat_bb(df, params):
    length = safe_int(params.get('bb_length'), 20)
    std = safe_float(params.get('bb_std'), 2.0)
    df.ta.bbands(length=length, std=std, append=True)
    bbl = find_col(df, "BBL_"); bbu = find_col(df, "BBU_")
    if bbl and bbu:
        df.loc[df['close'] <= df[bbl], 'ta_signal'] = 1
        df.loc[df['close'] >= df[bbu], 'ta_signal'] = -1
    return df

def _strat_rsi(df, params):
    rsi_len = safe_int(params.get('rsi_length'), 14)
    col_name = f'RSI_{rsi_len}'
    if col_name not in df.columns: df.ta.rsi(length=rsi_len, append=True)
    
    rsi_col = find_col(df, col_name)
    if not rsi_col: rsi_col = 'RSI'
    
    if rsi_col in df.columns:
        os = safe_float(params.get('oversold_level'), 30)
        ob = safe_float(params.get('overbought_level'), 70)
        df.loc[df[rsi_col] < os, 'ta_signal'] = 1
        df.loc[df[rsi_col] > ob, 'ta_signal'] = -1
    return df

def _strat_sma(df, params):
    f = safe_int(params.get('sma_fast_period'), 10)
    s = safe_int(params.get('sma_slow_period'), 50)
    df.ta.sma(length=f, append=True); df.ta.sma(length=s, append=True)
    sma_f = find_col(df, f'SMA_{f}'); sma_s = find_col(df, f'SMA_{s}')
    if sma_f and sma_s:
        df.loc[(df[sma_f] > df[sma_s]) & (df[sma_f].shift(1) <= df[sma_s].shift(1)), 'ta_signal'] = 1
        df.loc[(df[sma_f] < df[sma_s]) & (df[sma_f].shift(1) >= df[sma_s].shift(1)), 'ta_signal'] = -1
    return df

def _strat_macd(df, params):
    f = safe_int(params.get('macd_fast_period'), 12)
    s = safe_int(params.get('macd_slow_period'), 26)
    sig = safe_int(params.get('macd_signal_period'), 9)
    df.ta.macd(fast=f, slow=s, signal=sig, append=True)
    m_col = find_col(df, "MACD_"); s_col = find_col(df, "MACDs_")
    if m_col and s_col:
        df.loc[(df[m_col] > df[s_col]) & (df[m_col].shift(1) <= df[s_col].shift(1)), 'ta_signal'] = 1
        df.loc[(df[m_col] < df[s_col]) & (df[m_col].shift(1) >= df[s_col].shift(1)), 'ta_signal'] = -1
    return df

def _strat_stoch(df, params):
    k = safe_int(params.get('k_period'), 14)
    d = safe_int(params.get('d_period'), 3)
    df.ta.stoch(k=k, d=d, append=True)
    k_col = find_col(df, "STOCHk"); d_col = find_col(df, "STOCHd")
    if k_col and d_col:
        df.loc[(df[k_col] > df[d_col]) & (df[k_col].shift(1) <= df[d_col].shift(1)) & (df[k_col] < 20), 'ta_signal'] = 1
        df.loc[(df[k_col] < df[d_col]) & (df[k_col].shift(1) >= df[d_col].shift(1)) & (df[k_col] > 80), 'ta_signal'] = -1
    return df

def _strat_atr(df, params):
    atr_p = safe_int(params.get('atr_period'), 14)
    mult = safe_float(params.get('atr_multiplier'), 2.0)
    if f"ATR_{atr_p}" not in df.columns and f"ATRr_{atr_p}" not in df.columns:
        df.ta.atr(length=atr_p, append=True)
    
    df.ta.ema(length=20, append=True)
    atr_col = find_col(df, f"ATR_{atr_p}");
    if not atr_col: atr_col = find_col(df, f"ATRr_{atr_p}")
    if not atr_col and atr_p == 14: atr_col = 'ATR'

    ema_col = find_col(df, "EMA_20")
    if atr_col and ema_col:
        upper = df[ema_col] + (df[atr_col] * mult)
        lower = df[ema_col] - (df[atr_col] * mult)
        df.loc[df['close'] > upper, 'ta_signal'] = 1
        df.loc[df['close'] < lower, 'ta_signal'] = -1
    return df

def _strat_cci(df, params):
    cci_len = safe_int(params.get('cci_length'), 20)
    df.ta.cci(length=cci_len, append=True)
    cci_col = find_col(df, f"CCI_{cci_len}")
    if not cci_col: cci_col = find_col(df, "CCI")
    if cci_col and cci_col in df.columns:
        low_t = safe_float(params.get('cci_oversold'), -100)
        high_t = safe_float(params.get('cci_overbought'), 100)
        df.loc[df[cci_col] < low_t, 'ta_signal'] = 1
        df.loc[df[cci_col] > high_t, 'ta_signal'] = -1
    return df

def _strat_ichimoku(df, params):
    df.ta.ichimoku(append=True)
    span_a = find_col(df, "ISA_"); span_b = find_col(df, "ISB_")
    if span_a and span_b:
        df.loc[(df['close'] > df[span_a]) & (df['close'] > df[span_b]) & (df['close'].shift(1) <= df[span_a].shift(1)), 'ta_signal'] = 1
        df.loc[(df['close'] < df[span_a]) & (df['close'] < df[span_b]) & (df['close'].shift(1) >= df[span_b].shift(1)), 'ta_signal'] = -1
    return df

def _strat_obv(df, params):
    obv_ma = safe_int(params.get('obv_ma_period'), 20)
    df.ta.obv(append=True)
    if 'OBV' in df.columns: df[f'OBV_SMA_{obv_ma}'] = df['OBV'].rolling(window=obv_ma).mean()
    obv_sma = find_col(df, f"OBV_SMA_{obv_ma}")
    if obv_sma and 'OBV' in df.columns:
        df.loc[df['OBV'] > df[obv_sma], 'ta_signal'] = 1
        df.loc[df['OBV'] < df[obv_sma], 'ta_signal'] = -1
    return df

def generate_ta_signals(df, strategy_code, params):
    df = df.copy()
    df['ta_signal'] = 0
    
    if strategy_code in STRATEGY_MAP:
        try:
            df = STRATEGY_MAP[strategy_code](df, params)
        except Exception as e:
            logger.error(f"Strategy Error ({strategy_code}): {e}")
    
    # ✅ FIX 1: Look-ahead Bias Prevention
    # We generate signals based on Close.
    # We trade on the NEXT Open.
    # So we shift the signal forward by 1 candle.
    df['ta_signal'] = df['ta_signal'].shift(1).fillna(0)
    
    return df

# 🚀 PYRAMID MANAGER (FRICTION + REALISTIC EXECUTION)
class PyramidManager:
    def __init__(self, capital, fee, slippage, max_levels=1, tsl_mult=0, 
                 mode='standard', risk_pct=1.0, growth_target=2000.0):
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
        
        # 🚀 NEW: Funding Rate (0.01% per 8h)
        self.hourly_funding_rate = 0.0001 / 8 

    def check_risk_management(self, open_price, high, low, close, current_atr, time, candle_volume=0):
        if not self.positions: return

        for i in range(len(self.positions) - 1, -1, -1):
            pos = self.positions[i]
            
            # 1. Funding Fees (Deduct from position virtual equity)
            position_value = pos['qty'] * close
            funding_cost = position_value * self.hourly_funding_rate
            self.positions[i]['accumulated_fees'] = self.positions[i].get('accumulated_fees', 0) + funding_cost

            closed = False
            exit_reason = ""
            trigger_price = 0.0

            # --- LONG POSITIONS ---
            if pos['side'] == 'long':
                if pos['sl_price'] > 0:
                    if open_price <= pos['sl_price']: closed = True; exit_reason = "SL_GAP"; trigger_price = open_price
                    elif low <= pos['sl_price']: closed = True; exit_reason = "SL_HIT"; trigger_price = pos['sl_price']
                
                if not closed and pos['tp_price'] > 0:
                    if open_price >= pos['tp_price']: closed = True; exit_reason = "TP_GAP"; trigger_price = open_price
                    elif high >= pos['tp_price']: closed = True; exit_reason = "TP_HIT"; trigger_price = pos['tp_price']

                if not closed and self.tsl_mult > 0 and current_atr > 0:
                    if high > pos['highest_seen']: self.positions[i]['highest_seen'] = high
                    dynamic_stop = self.positions[i]['highest_seen'] - (current_atr * self.tsl_mult)
                    if open_price <= dynamic_stop: closed = True; exit_reason = "TSL_GAP"; trigger_price = open_price
                    elif low <= dynamic_stop: closed = True; exit_reason = "TSL_HIT"; trigger_price = dynamic_stop

            # --- SHORT POSITIONS ---
            elif pos['side'] == 'short':
                if pos['sl_price'] > 0:
                    if open_price >= pos['sl_price']: closed = True; exit_reason = "SL_GAP"; trigger_price = open_price
                    elif high >= pos['sl_price']: closed = True; exit_reason = "SL_HIT"; trigger_price = pos['sl_price']

                if not closed and pos['tp_price'] > 0:
                    if open_price <= pos['tp_price']: closed = True; exit_reason = "TP_GAP"; trigger_price = open_price
                    elif low <= pos['tp_price']: closed = True; exit_reason = "TP_HIT"; trigger_price = pos['tp_price']

                if not closed and self.tsl_mult > 0 and current_atr > 0:
                    if low < pos['lowest_seen']: self.positions[i]['lowest_seen'] = low
                    dynamic_stop = self.positions[i]['lowest_seen'] + (current_atr * self.tsl_mult)
                    if open_price >= dynamic_stop: closed = True; exit_reason = "TSL_GAP"; trigger_price = open_price
                    elif high >= dynamic_stop: closed = True; exit_reason = "TSL_HIT"; trigger_price = dynamic_stop

            if closed:
                # 🚀 NEW: Volume Impact Slippage
                actual_slippage = self.base_slippage
                if current_atr > 0 and open_price > 0:
                    if (current_atr / open_price) > 0.01: actual_slippage *= 2.0
                
                # If trading huge relative to volume, increase slippage
                if candle_volume > 0 and (pos['qty'] / candle_volume) > 0.01:
                    actual_slippage *= 3.0
                
                self.close_specific_position(i, trigger_price, time, exit_reason, actual_slippage)

    def enter(self, side, price, time, current_atr=0, tp_pct=0.0, sl_pct=0.0, candle_volume=0):
        price = float(price)
        
        if self.current_side and self.current_side != side: 
            self.close_all(price, time, "REVERSAL")
        
        if len(self.positions) >= self.max_levels: return
        
        # Debounce
        if self.positions:
            last_entry = self.positions[-1]['entry_price']
            if abs(price - last_entry) / last_entry < 0.001: return 

        current_balance = self.available_cash + sum(p['invested'] for p in self.positions)
        
        allocation = 0.0
        if self.mode == 'dynamic':
            if current_balance < self.growth_target:
                slots_left = self.max_levels - len(self.positions)
                if slots_left > 0: allocation = self.available_cash / slots_left
            else:
                allocation = current_balance * (self.risk_pct / 100.0)
        else:
            if self.risk_pct < 100:
                 allocation = current_balance * (self.risk_pct / 100.0)
            else:
                 slots_left = self.max_levels - len(self.positions)
                 if slots_left > 0: allocation = self.available_cash / slots_left

        if allocation > self.available_cash: allocation = self.available_cash
        if allocation < 1.0: return 
        
        # Slippage Logic
        actual_slippage = self.base_slippage
        if current_atr > 0 and price > 0 and (current_atr / price) > 0.01: actual_slippage *= 2
        
        # Volume Impact
        est_qty = allocation / price
        if candle_volume > 0 and (est_qty / candle_volume) > 0.01: actual_slippage *= 3

        entry_price = price * (1 + actual_slippage) if side == 'long' else price * (1 - actual_slippage)
        
        fee_amt = allocation * self.fee
        net_size = allocation - fee_amt
        qty = net_size / entry_price
        
        tp_price = 0.0
        sl_price = 0.0
        if side == 'long':
            if tp_pct > 0: tp_price = entry_price * (1 + tp_pct/100)
            if sl_pct > 0: sl_price = entry_price * (1 - sl_pct/100)
        else:
            if tp_pct > 0: tp_price = entry_price * (1 - tp_pct/100)
            if sl_pct > 0: sl_price = entry_price * (1 + sl_pct/100)

        self.positions.append({
            "side": side, "entry_price": entry_price, "qty": qty,
            "entry_time": time, "invested": allocation,
            "tp_price": tp_price, "sl_price": sl_price,
            "highest_seen": price, "lowest_seen": price,
            "accumulated_fees": 0.0
        })
        
        self.available_cash -= allocation
        self.current_side = side
        
    def close_specific_position(self, index, price, time, reason="SIGNAL", slippage_override=None):
        if index < 0 or index >= len(self.positions): return
        pos = self.positions.pop(index)
        
        price = float(price)
        used_slippage = slippage_override if slippage_override is not None else self.base_slippage
        
        exit_price = price * (1 - used_slippage) if pos['side'] == 'long' else price * (1 + used_slippage)
        
        raw_val = pos['qty'] * exit_price
        if pos['side'] == 'short':
            diff = pos['entry_price'] - exit_price
            pnl = diff * pos['qty']
            raw_val = pos['invested'] + pnl

        fee_amt = raw_val * self.fee
        funding_fees = pos.get('accumulated_fees', 0)

        net_return = raw_val - fee_amt - funding_fees
        profit = net_return - pos['invested']
        
        self.available_cash += net_return 
        
        self.completed_trades.append({
            "entryTime": pos['entry_time'], "exitTime": time,
            "price": pos['entry_price'], "exitPrice": exit_price,
            "profit": profit, "position": pos['side'], "type": reason
        })
        
        if not self.positions: self.current_side = None

    def close_all(self, price, time, reason="FORCE_CLOSE"):
        while self.positions: 
            self.close_specific_position(0, price, time, reason)

    def get_equity(self, current_price):
        equity = self.available_cash
        for pos in self.positions:
             pending_fees = pos.get('accumulated_fees', 0)
             if pos['side'] == 'long': equity += (pos['qty'] * current_price) - pending_fees
             else:
                 diff = pos['entry_price'] - current_price
                 equity += (pos['invested'] + diff * pos['qty']) - pending_fees
        return equity

def run_backtest(df, signal_col, initial_balance, fee, max_pyramiding=1, tsl_mult=0, params={}):
    mode = params.get('riskManagementMode', 'standard')
    risk_pct = safe_float(params.get('riskPercentage'), 100.0) 
    target = safe_float(params.get('growthCapitalTarget'), 2000.0)

    manager = PyramidManager(initial_balance, fee, SLIPPAGE_PCT, max_pyramiding, tsl_mult, mode, risk_pct, target)
    equity_curve = [{"timestamp": df.index[0].isoformat(), "balance": initial_balance}]
    
    atr_col = 'ATR'
    
    df = engineer_features_for_backtest(df, params)

    tf_period = safe_int(params.get('trendFilterPeriod'), 200)
    trend_col = f"SMA_{tf_period}"

    min_adx = safe_int(params.get('minAdxLevel'), 0)
    min_atr_pct = safe_float(params.get('minAtrPct'), 0.0)
    tp_pct = safe_float(params.get('TP'), 0.0)
    sl_pct = safe_float(params.get('SL'), 0.0)

    for i in range(1, len(df)):
        row = df.iloc[i]
        
        open_price = float(row['open'])
        high = float(row['high'])
        low = float(row['low'])
        close = float(row['close'])
        volume = float(row['volume']) # 🚀 Pass volume
        curr_time = df.index[i].isoformat()
        current_atr = float(row[atr_col]) if atr_col in df.columns else 0.0
        
        # 2. ENTRY PHASE (Happens at OPEN)
        sig = df[signal_col].iloc[i]
        prev_row = df.iloc[i-1]
        allowed = True
        
        if trend_col in df.columns:
            trend_val = prev_row[trend_col]
            if sig == 1 and prev_row['close'] < trend_val: allowed = False 
            if sig == -1 and prev_row['close'] > trend_val: allowed = False 
        
        if min_adx > 0 and 'ADX' in df.columns and prev_row['ADX'] < min_adx: allowed = False
        
        if min_atr_pct > 0 and atr_col in df.columns and prev_row['close'] > 0:
            current_atr_pct = (prev_row[atr_col] / prev_row['close']) * 100
            if current_atr_pct < min_atr_pct: allowed = False
        
        if allowed:
            if sig == 1: 
                manager.enter('long', open_price, curr_time, current_atr, tp_pct, sl_pct, volume)
            elif sig == -1: 
                manager.enter('short', open_price, curr_time, current_atr, tp_pct, sl_pct, volume)
        
        # 3. RISK PHASE (Intraday)
        manager.check_risk_management(open_price, high, low, close, current_atr, curr_time, volume)
            
        equity_curve.append({"timestamp": curr_time, "balance": manager.get_equity(close)})

    if not df.empty:
        manager.close_all(df['close'].iloc[-1], df.index[-1].isoformat(), "END_OF_DATA")
    
    final_balance = manager.available_cash
    trades = manager.completed_trades
    wins = [t for t in trades if t['profit'] > 0]
    
    gross_profit = sum(t['profit'] for t in wins)
    gross_loss = abs(sum(t['profit'] for t in trades if t['profit'] < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999
    avg_win = (gross_profit / len(wins)) if wins else 0
    avg_loss = (gross_loss / (len(trades) - len(wins))) if (len(trades) - len(wins)) > 0 else 0

    total_return = ((final_balance - initial_balance) / initial_balance) * 100
    max_dd = 0
    balances = [x['balance'] for x in equity_curve]
    if balances:
        peak = balances[0]
        for b in balances:
            if b > peak: peak = b
            dd = (peak - b) / peak * 100
            if dd > max_dd: max_dd = dd

    return {
        "metrics": {
            "totalReturn": total_return, "finalBalance": final_balance, "totalTrades": len(trades),
            "winningTrades": len(wins), "losingTrades": len(trades) - len(wins),
            "winRate": (len(wins)/len(trades)*100) if trades else 0, "maxDrawdown": max_dd,
            "profitFactor": profit_factor, "averageWin": avg_win, "averageLoss": avg_loss
        },
        "equityCurve": equity_curve, "tradeBreakdown": trades, 
        "candleData": df[['open','high','low','close','volume']].reset_index().to_dict('records')
    }

# --- 4. LIVE BOT CLASS ---
class PaperTradingBot:
    def __init__(self, config: BotConfig):
        self.config = config
        self.is_running = False
        self.logs = []
        
        level = config.maxPyramiding
        if not level or level < 1: level = int(config.params.get('maxPyramiding', 1))
        fee = float(config.params.get('fee', DEFAULT_TAKER_FEE))
        tsl_mult = float(config.params.get('tslAtrMult', 0)) 
        
        mode = config.params.get('riskManagementMode', 'standard')
        risk_pct = safe_float(config.params.get('riskPercentage'), 100.0)
        target = safe_float(config.params.get('growthCapitalTarget'), 2000.0)

        self.manager = PyramidManager(config.capitalAllocation, fee, SLIPPAGE_PCT, level, tsl_mult, mode, risk_pct, target)
        self.df = None 

    @property
    def trades(self): return self.manager.completed_trades
    
    @property
    def current_balance(self): return self.manager.available_cash 

    def log(self, message):
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] {message}")
        self.logs.append({"timestamp": timestamp, "message": message, "type": "info"})

    def get_profit(self):
        if self.df is None or self.df.empty: return 0
        cur_price = self.df['close'].iloc[-1]
        return self.manager.get_equity(cur_price) - self.config.capitalAllocation

    def get_win_rate(self):
        trades = self.manager.completed_trades
        if not trades: return 0.0
        wins = len([t for t in trades if t['profit'] > 0])
        return (wins / len(trades)) * 100

    def stop(self):
        self.is_running = False
        self.log("Bot stopping...")

    def run(self):
        self.is_running = True
        self.log(f"Bot started. {self.config.symbol} {self.config.timeframe}. Mode: {self.manager.mode}")

        try:
            df = fetch_live_data(self.config.symbol, self.config.timeframe)
            if not df.empty:
                self.df = df
                self.log(f"Live data loaded: {len(df)} candles.")
            
            while self.is_running:
                for _ in range(60): 
                    if not self.is_running: break
                    time.sleep(1)
                if not self.is_running: break

                try:
                    df = fetch_live_data(self.config.symbol, self.config.timeframe) 
                    if df.empty: continue
                    self.df = df
                    
                    norm_params = normalize_params(self.config)
                    df = engineer_features_for_backtest(df, norm_params)

                    sigs = []
                    for i, s in enumerate(self.config.strategies):
                        s_params = norm_params.copy(); s_params.update(s.params) 
                        df = generate_ta_signals(df, s.code, s_params)
                        col = f's_{i}'; df[col] = df['ta_signal']; sigs.append(col)

                    current_live_candle = df.iloc[-1]
                    final_signal = 0
                    mode = self.config.params.get("hybridMode", "AND")
                    
                    if len(sigs) >= 2:
                        if mode == "AND":
                            if all(current_live_candle[c] == 1 for c in sigs): final_signal = 1
                            elif all(current_live_candle[c] == -1 for c in sigs): final_signal = -1
                        elif mode == "REGIME":
                            adx_col = 'ADX'
                            thresh = int(self.config.params.get("regime_threshold", 25))
                            current_adx = current_live_candle[adx_col] if adx_col in df.columns else 0
                            active_strat = sigs[0] if current_adx > thresh else sigs[1]
                            final_signal = current_live_candle[active_strat]
                        else: 
                             vals = [current_live_candle[c] for c in sigs]
                             if 1 in vals and -1 not in vals: final_signal = 1
                             elif -1 in vals and 1 not in vals: final_signal = -1
                    elif sigs: final_signal = current_live_candle[sigs[0]]

                    timestamp = datetime.now().isoformat()
                    current_price = float(current_live_candle['close']) 
                    
                    atr_col = 'ATR'
                    current_atr = float(current_live_candle[atr_col]) if atr_col in df.columns else 0.0
                    volume = float(current_live_candle['volume'])

                    self.manager.check_risk_management(current_price, current_live_candle['high'], current_live_candle['low'], current_price, current_atr, timestamp, volume)

                    allowed = True
                    tf_period = safe_int(norm_params.get('trendFilterPeriod'), 200)
                    trend_col = f"SMA_{tf_period}"
                    
                    if trend_col in df.columns:
                        trend_val = current_live_candle[trend_col]
                        if final_signal == 1 and current_price < trend_val: allowed = False
                        if final_signal == -1 and current_price > trend_val: allowed = False
                        
                    min_adx = safe_int(norm_params.get('minAdxLevel'), 0)
                    if min_adx > 0 and 'ADX' in df.columns and current_live_candle['ADX'] < min_adx: allowed = False
                    
                    tp_pct = safe_float(norm_params.get('TP'), 0.0)
                    sl_pct = safe_float(norm_params.get('SL'), 0.0)

                    if allowed:
                        if final_signal == 1:
                            if self.manager.current_side == 'short':
                                self.manager.close_all(current_price, timestamp, "REVERSAL")
                                self.log(f"🔄 FLIP TO LONG: ${current_price:.2f}")
                            self.manager.enter('long', current_price, timestamp, current_atr, tp_pct, sl_pct, volume)
                            
                        elif final_signal == -1:
                            if self.manager.current_side == 'long':
                                self.manager.close_all(current_price, timestamp, "REVERSAL")
                                self.log(f"🔄 FLIP TO SHORT: ${current_price:.2f}")
                            self.manager.enter('short', current_price, timestamp, current_atr, tp_pct, sl_pct, volume)

                except Exception as inner:
                    print(f"Loop Error: {inner}")

        except Exception as e:
             self.log(f"Crash: {e}")
             self.is_running = False

# ==============================================================================
# 5. API ENDPOINTS
# ==============================================================================

@app.post('/api/ml/run-backtest-on') 
@app.post('/api/ml/run-combo-backtest')
async def handle_run_combo_backtest(config: BacktestConfig):
    try:
        norm_params = normalize_params(config)
        
        start_date_obj = datetime.strptime(config.startDate, "%Y-%m-%d")
        days_diff = (datetime.now() - start_date_obj).days
        max_days = 730 if config.timeframe in ['1h', '30m'] else 60 
        if days_diff > max_days:
            new_start = (datetime.now() - timedelta(days=max_days - 2)).strftime("%Y-%m-%d")
            config.startDate = new_start

        df = load_efficient_data(config.symbol, config.timeframe, config.startDate, config.endDate)
        if df.empty: 
            return JSONResponse(content={"metrics": {}, "equityCurve": [], "tradeBreakdown": [], "candleData": []})
            
        df = engineer_features_for_backtest(df, norm_params)
        
        if config.code:
             df = generate_ta_signals(df, config.code, norm_params)
             df['comb'] = df['ta_signal']
        elif config.strategies:
             sigs = []
             for i, s in enumerate(config.strategies):
                 s_params = norm_params.copy()
                 s_params.update(s.get('params', {}))
                 df = generate_ta_signals(df, s['code'], s_params)
                 df[f's_{i}'] = df['ta_signal']
                 sigs.append(f's_{i}')
             
             mode = config.params.get("hybridMode", "OR")
             if len(sigs) >= 2:
                 if mode == "AND":
                     def combine_and(row):
                         vals = [row[c] for c in sigs]
                         if all(v == 1 for v in vals): return 1
                         if all(v == -1 for v in vals): return -1
                         return 0
                     df['comb'] = df.apply(combine_and, axis=1)
                 elif mode == "REGIME":
                     thresh = safe_int(config.params.get("regime_threshold"), 25)
                     if 'ADX' in df.columns:
                         # Shift ADX to match signal lag
                         adx_shifted = df['ADX'].shift(1).fillna(0)
                         df['comb'] = np.where(adx_shifted > thresh, df[sigs[0]], df[sigs[1]])
                     else:
                         df['comb'] = 0
                 else:
                     def combine_or(row):
                         vals = [row[c] for c in sigs]
                         has_buy = 1 in vals
                         has_sell = -1 in vals
                         if has_buy and not has_sell: return 1
                         if has_sell and not has_buy: return -1
                         return 0
                     df['comb'] = df.apply(combine_or, axis=1)
             elif sigs: df['comb'] = df[sigs[0]]
             else: df['comb'] = 0 
        else: df['comb'] = 0 

        pyramid_lvl = config.maxPyramiding 
        if pyramid_lvl <= 1: pyramid_lvl = int(config.params.get("maxPyramiding", 1))
        tsl_mult = float(config.params.get('tslAtrMult', 0))
        
        # Pass Parameters for Dynamic Growth
        params_with_growth = config.params.copy()
        params_with_growth['riskManagementMode'] = config.riskManagementMode
        params_with_growth['riskPercentage'] = config.riskPercentage
        params_with_growth['growthCapitalTarget'] = config.growthCapitalTarget

        res = run_backtest(df, 'comb', config.initialBalance, config.fee, pyramid_lvl, tsl_mult, params_with_growth)
        
        clean_res = convert_numpy_types(res)
        response_payload = deepcopy(clean_res)
        response_payload["combinedResult"] = deepcopy(clean_res)
        return JSONResponse(content=response_payload)
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
        if f.endswith(".json") and "winner" in f:
            try:
                path = os.path.join(RESULTS_DIR, f)
                if os.path.getsize(path) == 0: continue
                with open(path, 'r') as file: data = json.load(file)
                
                if isinstance(data, list):
                    data = { "strategies": data, "params": {}, "metrics": {"calmar": 0} }

                raw_strat = data.get('strategies', 'Unknown')
                strat_name = "Unknown Strategy"

                if isinstance(raw_strat, list):
                    names = []
                    for s in raw_strat:
                        if isinstance(s, dict): names.append(s.get('code', 'Unknown'))
                    strat_name = "+".join(names)
                elif isinstance(raw_strat, str):
                    strat_name = raw_strat

                calmar = data.get('metrics', {}).get('calmar', 0)
                winners.append({"id": f, "name": f"{strat_name} (Calmar: {calmar:.2f})", "config": data, "timestamp": data.get('timestamp', '')})
            except: pass
    winners.sort(key=lambda x: str(x.get('timestamp', '')), reverse=True)
    return winners

@app.post("/api/bot/start")
async def start_bot_endpoint(config: BotConfig):
    global bot_instance
    if bot_instance and bot_instance.is_running:
        return {"status": "error", "message": "Bot is already running"}
    
    bot_instance = PaperTradingBot(config)
    
    bot_thread = threading.Thread(target=bot_instance.run)
    bot_thread.daemon = True
    bot_thread.start()
    return {"status": "started", "config": config}

@app.post("/api/bot/stop")
async def stop_bot_endpoint():
    global bot_instance
    if bot_instance:
        bot_instance.stop()
        return {"status": "stopped"}
    return {"status": "not_running"}

@app.get("/api/bot/status")
async def get_bot_status_endpoint():
    global bot_instance
    status_data = {"status": "stopped", "logs": [], "trades": [], "activePositions": [], "currentBalance": 0, "candles": [], "performanceMetrics": {}}

    if bot_instance:
        status_data["status"] = "running" if bot_instance.is_running else "stopped"
        status_data["logs"] = list(bot_instance.logs[-50:])
        
        gross_profit = sum(t.get('profit',0) for t in bot_instance.trades if t.get('profit',0) > 0)
        gross_loss = abs(sum(t.get('profit',0) for t in bot_instance.trades if t.get('profit',0) < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        status_data["trades"] = list(bot_instance.manager.completed_trades)
        status_data["activePositions"] = list(bot_instance.manager.positions)
        status_data["currentBalance"] = float(bot_instance.manager.available_cash)
        
        if bot_instance.df is not None:
            try:
                tail = bot_instance.df.tail(100).copy()
                tail.reset_index(inplace=True)
                time_col = None
                for col in tail.columns:
                    if str(col).lower() in ['date', 'datetime', 'timestamp', 'index', 'time']:
                        time_col = col
                        break
                if time_col:
                    tail['time'] = tail[time_col].astype(str)
                    keep_cols = [c for c in ['time', 'open', 'high', 'low', 'close'] if c in tail.columns]
                    for c in keep_cols:
                        if c != 'time': tail[c] = tail[c].astype(float)
                    status_data["candles"] = tail[keep_cols].to_dict(orient='records')
            except: pass

        status_data["performanceMetrics"] = {
            "totalProfit": bot_instance.get_profit(),
            "winRate": bot_instance.get_win_rate(),
            "totalTrades": len(bot_instance.trades),
            "profitFactor": profit_factor,
            "maxDrawdown": 0,
            "currentBalance": float(bot_instance.manager.available_cash)
        }
    return status_data

@app.get("/api/bot/logs")
async def get_bot_logs_endpoint(limit: int = 100):
    if bot_instance: return bot_instance.logs[-limit:]
    return []

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
