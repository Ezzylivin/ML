# File: ml.py
# 🚀 UPGRADE: v37.1 - "True Dynamic Growth" (Compounding Logic Fixed)

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

DEFAULT_TAKER_FEE = 0.001   # 0.1% Fee
SLIPPAGE_PCT = 0.0005       # 0.05% Slippage

app = FastAPI(title="Trading ML Server API v37.1")

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
    # New Fields for Dynamic Growth
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

def fetch_live_data(symbol, timeframe):
    try:
        symbol = symbol.replace('/', '-')
        
        tf_map = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '4h': '1h', '1d': '1d'}
        interval = tf_map.get(timeframe, '1h')
        
        period = "max"
        if timeframe == '1m': period = "7d"
        elif timeframe in ['5m', '15m']: period = "60d"
        elif timeframe in ['30m', '1h']: period = "730d" 
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty: return pd.DataFrame()
            
        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        df.index.name = 'datetime'
        
        if df.index.tz is None: df.index = df.index.tz_localize('UTC')
        else: df.index = df.index.tz_convert('UTC')
        
        return df
    except Exception as e:
        print(f"❌ YFinance Error: {e}")
        return pd.DataFrame()

def load_efficient_data(symbol, timeframe, start_date=None, end_date=None):
    safe_symbol = symbol.replace('/', '-')
    data_filename = f"{safe_symbol}-{timeframe}.csv"
    data_path = os.path.join(DATA_DIR, data_filename)
    
    should_fetch = True
    if os.path.exists(data_path):
        if os.path.getsize(data_path) < 100: should_fetch = True
        else:
            file_time = datetime.fromtimestamp(os.path.getmtime(data_path))
            if datetime.now() - file_time < timedelta(hours=1):
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
            
            df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True, errors='ignore')
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
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.sma(length=50, append=True)
        trend_period = safe_int(params.get('trendFilterPeriod'), 200)
        if trend_period > 0: df.ta.sma(length=trend_period, append=True) 
    except: pass
    return df

def generate_ta_signals(df, strategy_code, params):
    df = df.copy()
    df['ta_signal'] = 0
    if params is None: params = {}
    try:
        df = engineer_features_for_backtest(df, params)

        if strategy_code == "psar_signal":
            step = safe_float(params.get('psar_step'), 0.02)
            max_step = safe_float(params.get('psar_max'), 0.2)
            df.ta.psar(step=step, max_step=max_step, append=True)
            psarl = find_col(df, "PSARl"); psars = find_col(df, "PSARs")
            if psarl and psars:
                df.loc[df[psarl].notna() & (df[psarl] > 0), 'ta_signal'] = 1
                df.loc[df[psars].notna() & (df[psars] > 0), 'ta_signal'] = -1
        
        elif strategy_code == "bollinger_bands":
            length = safe_int(params.get('bb_length'), 20)
            std = safe_float(params.get('bb_std'), 2.0)
            df.ta.bbands(length=length, std=std, append=True)
            bbl = find_col(df, "BBL_"); bbu = find_col(df, "BBU_")
            if bbl and bbu:
                df.loc[df['close'] <= df[bbl], 'ta_signal'] = 1
                df.loc[df['close'] >= df[bbu], 'ta_signal'] = -1
        
        elif strategy_code == "rsi_divergence":
             rsi_len = safe_int(params.get('rsi_length'), 14)
             df.ta.rsi(length=rsi_len, append=True)
             rsi_col = find_col(df, f'RSI_{rsi_len}') 
             if not rsi_col: rsi_col = find_col(df, 'RSI')
             if rsi_col:
                 os = safe_float(params.get('oversold_level'), 30)
                 ob = safe_float(params.get('overbought_level'), 70)
                 df.loc[df[rsi_col] < os, 'ta_signal'] = 1
                 df.loc[df[rsi_col] > ob, 'ta_signal'] = -1
        
        elif strategy_code == "sma_crossover":
            f = safe_int(params.get('sma_fast_period'), 10)
            s = safe_int(params.get('sma_slow_period'), 50)
            df.ta.sma(length=f, append=True); df.ta.sma(length=s, append=True)
            sma_f = find_col(df, f'SMA_{f}'); sma_s = find_col(df, f'SMA_{s}')
            if sma_f and sma_s:
                df.loc[(df[sma_f] > df[sma_s]) & (df[sma_f].shift(1) <= df[sma_s].shift(1)), 'ta_signal'] = 1
                df.loc[(df[sma_f] < df[sma_s]) & (df[sma_f].shift(1) >= df[sma_s].shift(1)), 'ta_signal'] = -1
        
        elif strategy_code == "macd_crossover":
             f = safe_int(params.get('macd_fast_period'), 12)
             s = safe_int(params.get('macd_slow_period'), 26)
             sig = safe_int(params.get('macd_signal_period'), 9)
             df.ta.macd(fast=f, slow=s, signal=sig, append=True)
             m_col = find_col(df, "MACD_"); s_col = find_col(df, "MACDs_")
             if m_col and s_col:
                 df.loc[(df[m_col] > df[s_col]) & (df[m_col].shift(1) <= df[s_col].shift(1)), 'ta_signal'] = 1
                 df.loc[(df[m_col] < df[s_col]) & (df[m_col].shift(1) >= df[s_col].shift(1)), 'ta_signal'] = -1
        
        elif strategy_code == "stochastic_crossover":
             k = safe_int(params.get('k_period'), 14)
             d = safe_int(params.get('d_period'), 3)
             df.ta.stoch(k=k, d=d, append=True)
             k_col = find_col(df, "STOCHk"); d_col = find_col(df, "STOCHd")
             if k_col and d_col:
                 df.loc[(df[k_col] > df[d_col]) & (df[k_col].shift(1) <= df[d_col].shift(1)) & (df[k_col] < 20), 'ta_signal'] = 1
                 df.loc[(df[k_col] < df[d_col]) & (df[k_col].shift(1) >= df[d_col].shift(1)) & (df[k_col] > 80), 'ta_signal'] = -1
        
        elif strategy_code == "atr_breakout":
             atr_p = safe_int(params.get('atr_period'), 14)
             mult = safe_float(params.get('atr_multiplier'), 2.0)
             df.ta.atr(length=atr_p, append=True)
             df.ta.ema(length=20, append=True)
             atr_col = find_col(df, f"ATR_{atr_p}"); 
             if not atr_col: atr_col = find_col(df, f"ATRr_{atr_p}")
             ema_col = find_col(df, "EMA_20")
             if atr_col and ema_col:
                 upper = df[ema_col] + (df[atr_col] * mult)
                 lower = df[ema_col] - (df[atr_col] * mult)
                 df.loc[df['close'] > upper, 'ta_signal'] = 1
                 df.loc[df['close'] < lower, 'ta_signal'] = -1
        
        elif strategy_code == "cci_oversold":
             cci_len = safe_int(params.get('cci_length'), 20)
             df.ta.cci(length=cci_len, append=True)
             cci_col = find_col(df, f"CCI_{cci_len}")
             if not cci_col: cci_col = find_col(df, "CCI")
             if cci_col and cci_col in df.columns:
                 low_t = safe_float(params.get('cci_oversold'), -100)
                 high_t = safe_float(params.get('cci_overbought'), 100)
                 df.loc[df[cci_col] < low_t, 'ta_signal'] = 1
                 df.loc[df[cci_col] > high_t, 'ta_signal'] = -1
        
        elif strategy_code == "ichimoku_cloud":
             df.ta.ichimoku(append=True)
             span_a = find_col(df, "ISA_"); span_b = find_col(df, "ISB_")
             if span_a and span_b:
                 df.loc[(df['close'] > df[span_a]) & (df['close'] > df[span_b]) & (df['close'].shift(1) <= df[span_a].shift(1)), 'ta_signal'] = 1
                 df.loc[(df['close'] < df[span_a]) & (df['close'] < df[span_b]) & (df['close'].shift(1) >= df[span_b].shift(1)), 'ta_signal'] = -1
        
        elif strategy_code == "obv_signal":
             obv_ma = safe_int(params.get('obv_ma_period'), 20)
             df.ta.obv(append=True)
             if 'OBV' in df.columns: df[f'OBV_SMA_{obv_ma}'] = df['OBV'].rolling(window=obv_ma).mean()
             obv_sma = find_col(df, f"OBV_SMA_{obv_ma}")
             if obv_sma and 'OBV' in df.columns:
                 df.loc[df['OBV'] > df[obv_sma], 'ta_signal'] = 1
                 df.loc[df['OBV'] < df[obv_sma], 'ta_signal'] = -1
    except Exception as e:
        print(f"❌ Signal Gen Error ({strategy_code}): {e}")
    return df

# 🚀 PYRAMID MANAGER (UPDATED FOR DYNAMIC GROWTH)
class PyramidManager:
    def __init__(self, capital, fee, slippage, max_levels=1, tsl_mult=0, 
                 mode='standard', risk_pct=1.0, growth_target=2000.0):
        self.total_capital = float(capital)
        self.available_cash = float(capital)
        self.fee = float(fee)
        self.slippage = float(slippage)
        self.max_levels = int(max_levels) if int(max_levels) > 0 else 1
        self.tsl_mult = float(tsl_mult) 
        
        # Risk Management Settings
        self.mode = mode # 'standard' or 'dynamic'
        self.risk_pct = float(risk_pct)
        self.growth_target = float(growth_target)
        
        self.positions = [] 
        self.completed_trades = []
        self.current_side = None 

    # 1. RISK CHECK 
    def check_risk_management(self, current_price, high, low, current_atr, time):
        if not self.positions: return

        for i in range(len(self.positions) - 1, -1, -1):
            pos = self.positions[i]
            closed = False
            exit_reason = ""
            trigger_price = 0.0

            if pos['side'] == 'long':
                if pos['sl_price'] > 0 and low <= pos['sl_price']:
                    closed = True; exit_reason = "SL_HIT"; trigger_price = pos['sl_price']
                elif pos['tp_price'] > 0 and high >= pos['tp_price']:
                    closed = True; exit_reason = "TP_HIT"; trigger_price = pos['tp_price']
                
            elif pos['side'] == 'short':
                if pos['sl_price'] > 0 and high >= pos['sl_price']:
                    closed = True; exit_reason = "SL_HIT"; trigger_price = pos['sl_price']
                elif pos['tp_price'] > 0 and low <= pos['tp_price']:
                    closed = True; exit_reason = "TP_HIT"; trigger_price = pos['tp_price']

            if not closed and self.tsl_mult > 0 and current_atr > 0:
                if pos['side'] == 'long':
                    if high > pos['highest_seen']: self.positions[i]['highest_seen'] = high
                    dynamic_stop = self.positions[i]['highest_seen'] - (current_atr * self.tsl_mult)
                    if low <= dynamic_stop:
                        closed = True; exit_reason = "TSL_HIT"; trigger_price = dynamic_stop

                elif pos['side'] == 'short':
                    if low < pos['lowest_seen']: self.positions[i]['lowest_seen'] = low
                    dynamic_stop = self.positions[i]['lowest_seen'] + (current_atr * self.tsl_mult)
                    if high >= dynamic_stop:
                        closed = True; exit_reason = "TSL_HIT"; trigger_price = dynamic_stop

            if closed:
                self.close_specific_position(i, trigger_price, time, exit_reason)

    # 2. ENTRY (UPDATED SIZING LOGIC)
    def enter(self, side, price, time, tp_pct=0.0, sl_pct=0.0):
        price = float(price)
        
        if self.current_side and self.current_side != side: 
            self.close_all(price, time, "REVERSAL")
        
        if len(self.positions) >= self.max_levels: return
        
        if self.positions:
            last_entry = self.positions[-1]['entry_price']
            if abs(price - last_entry) / last_entry < 0.001: return 

        # 🚀 LOGIC FIX: CALCULATE ALLOCATION DYNAMICALLY
        # First, calculate roughly total equity (Cash + Invested) to check target
        # We can approximate Equity = available_cash + invested_in_open_positions
        # (This ignores unrealized PnL for sizing safety, effectively using 'Balance' not 'Equity')
        current_balance = self.available_cash + sum(p['invested'] for p in self.positions)
        
        allocation = 0.0
        
        if self.mode == 'dynamic':
            if current_balance < self.growth_target:
                # PHASE 1: AGGRESSIVE GROWTH
                # Allocate all available cash divided by remaining slots
                # (Compounding logic: Uses the profits that were added to available_cash)
                slots_left = self.max_levels - len(self.positions)
                if slots_left > 0:
                    allocation = self.available_cash / slots_left
            else:
                # PHASE 2: WEALTH PRESERVATION
                # Use Risk % (e.g. 1% of Balance)
                # But typically risk% refers to RISK amount, here we simulate Position Size % for simplicity
                # If risk_pct is 1, we use 1% of balance. 
                # If user meant 1% RISK (SL distance), we'd need SL distance. 
                # Assuming "Position Size %" based on inputs usually seen in simple bots.
                # However, usually users set risk_pct=100 for full use.
                # If risk_pct is small (1-5), it's likely per-trade risk.
                # Let's interpret riskPercentage as "Allocation % of Balance"
                allocation = current_balance * (self.risk_pct / 100.0)
        else:
            # STANDARD MODE:
            # If risk_pct is provided and < 100, use it. Otherwise 100% split.
            if self.risk_pct < 100:
                 allocation = current_balance * (self.risk_pct / 100.0)
            else:
                 # Default split logic
                 slots_left = self.max_levels - len(self.positions)
                 if slots_left > 0: allocation = self.available_cash / slots_left

        # Sanity Check
        if allocation > self.available_cash: allocation = self.available_cash
        if allocation < 10: return 
        
        entry_price = price * (1 + self.slippage) if side == 'long' else price * (1 - self.slippage)
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
            "highest_seen": price, "lowest_seen": price
        })
        
        self.available_cash -= allocation
        self.current_side = side
        
    def close_specific_position(self, index, price, time, reason="SIGNAL"):
        if index < 0 or index >= len(self.positions): return
        pos = self.positions.pop(index)
        
        price = float(price)
        exit_price = price * (1 - self.slippage) if pos['side'] == 'long' else price * (1 + self.slippage)
        
        raw_val = pos['qty'] * exit_price
        if pos['side'] == 'short':
            diff = pos['entry_price'] - exit_price
            pnl = diff * pos['qty']
            raw_val = pos['invested'] + pnl

        fee_amt = raw_val * self.fee
        net_return = raw_val - fee_amt
        profit = net_return - pos['invested']
        
        # 🚀 CRITICAL: PROFITS ARE ADDED TO CASH FOR REINVESTMENT
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
             if pos['side'] == 'long': equity += pos['qty'] * current_price
             else:
                 diff = pos['entry_price'] - current_price
                 equity += (pos['invested'] + diff * pos['qty'])
        return equity

def run_backtest(df, signal_col, initial_balance, fee, max_pyramiding=1, tsl_mult=0, params={}):
    # Extract Dynamic Params
    mode = params.get('riskManagementMode', 'standard')
    risk_pct = safe_float(params.get('riskPercentage'), 100.0) # Default to 100% usage if not dynamic
    target = safe_float(params.get('growthCapitalTarget'), 2000.0)

    # Pass them to Manager
    manager = PyramidManager(initial_balance, fee, SLIPPAGE_PCT, max_pyramiding, tsl_mult, mode, risk_pct, target)
    
    equity_curve = [{"timestamp": df.index[0].isoformat(), "balance": initial_balance}]
    
    atr_col = find_col(df, "ATR")
    if not atr_col: 
        df.ta.atr(length=14, append=True)
        atr_col = find_col(df, "ATR")
    
    adx_col = find_col(df, "ADX")
    
    tf_period = safe_int(params.get('trendFilterPeriod'), 200)
    trend_col = f"SMA_{tf_period}"
    if trend_col not in df.columns and tf_period > 0: 
        try: df.ta.sma(length=tf_period, append=True)
        except: pass

    min_adx = safe_int(params.get('minAdxLevel'), 0)
    min_atr_pct = safe_float(params.get('minAtrPct'), 0.0)
    tp_pct = safe_float(params.get('TP'), 0.0)
    sl_pct = safe_float(params.get('SL'), 0.0)

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = float(row['close'])
        high = float(row['high'])
        low = float(row['low'])
        curr_time = df.index[i].isoformat()
        current_atr = float(row[atr_col]) if atr_col else 0.0
        
        manager.check_risk_management(price, high, low, current_atr, curr_time)
        
        sig = df[signal_col].iloc[i]
        allowed = True
        
        if min_adx > 0 and adx_col and row[adx_col] < min_adx: allowed = False
        if min_atr_pct > 0 and atr_col:
            current_atr_pct = (row[atr_col] / price) * 100
            if current_atr_pct < min_atr_pct: allowed = False
        if trend_col in df.columns and tf_period > 0:
            trend_val = row[trend_col]
            if sig == 1 and price < trend_val: allowed = False 
            if sig == -1 and price > trend_val: allowed = False 
        
        if allowed:
            if sig == 1: 
                manager.enter('long', price, curr_time, tp_pct, sl_pct)
            elif sig == -1: 
                manager.enter('short', price, curr_time, tp_pct, sl_pct)
            
        equity_curve.append({"timestamp": curr_time, "balance": manager.get_equity(price)})

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
        
        # LIVE BOT PARAM EXTRATION
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
                    sigs = []
                    for i, s in enumerate(self.config.strategies):
                        s_params = norm_params.copy(); s_params.update(s.params) 
                        df = generate_ta_signals(df, s.code, s_params)
                        col = f's_{i}'; df[col] = df['ta_signal']; sigs.append(col)

                    last_completed_candle = df.iloc[-2] 
                    current_live_candle = df.iloc[-1]
                    current_price = float(current_live_candle['close'])
                    
                    final_signal = 0
                    mode = self.config.params.get("hybridMode", "AND")
                    
                    if len(sigs) >= 2:
                        if mode == "AND":
                            if all(last_completed_candle[c] == 1 for c in sigs): final_signal = 1
                            elif all(last_completed_candle[c] == -1 for c in sigs): final_signal = -1
                        elif mode == "REGIME":
                            if 'ADX_14' not in df.columns:
                                try: df.ta.adx(length=14, append=True); last_completed_candle = df.iloc[-2]
                                except: pass
                            adx_col = find_col(df, "ADX")
                            thresh = int(self.config.params.get("regime_threshold", 25))
                            current_adx = last_completed_candle[adx_col] if adx_col else 0
                            active_strat = sigs[0] if current_adx > thresh else sigs[1]
                            final_signal = last_completed_candle[active_strat]
                        else: 
                             vals = [last_completed_candle[c] for c in sigs]
                             if 1 in vals and -1 not in vals: final_signal = 1
                             elif -1 in vals and 1 not in vals: final_signal = -1
                    elif sigs: final_signal = last_completed_candle[sigs[0]]

                    timestamp = datetime.now().isoformat()
                    
                    tp_pct = safe_float(norm_params.get('TP'), 0.0)
                    sl_pct = safe_float(norm_params.get('SL'), 0.0)
                    
                    atr_col = find_col(df, "ATR")
                    current_atr = float(last_completed_candle[atr_col]) if atr_col else 0.0
                    
                    self.manager.check_risk_management(current_price, current_live_candle['high'], current_live_candle['low'], current_atr, timestamp)

                    allowed = True
                    tf_period = safe_int(norm_params.get('trendFilterPeriod'), 200)
                    trend_col = f"SMA_{tf_period}"
                    if trend_col in last_completed_candle:
                        trend_val = last_completed_candle[trend_col]
                        if final_signal == 1 and current_price < trend_val: allowed = False
                        if final_signal == -1 and current_price > trend_val: allowed = False
                    min_adx = safe_int(norm_params.get('minAdxLevel'), 0)
                    adx_col = find_col(df, "ADX")
                    if min_adx > 0 and adx_col and last_completed_candle[adx_col] < min_adx: allowed = False
                    min_atr_pct = safe_float(norm_params.get('minAtrPct'), 0.0)
                    if min_atr_pct > 0 and atr_col:
                        curr_atr_pct = (last_completed_candle[atr_col] / current_price) * 100
                        if curr_atr_pct < min_atr_pct: allowed = False

                    if allowed:
                        if final_signal == 1:
                            if self.manager.current_side == 'short':
                                self.manager.close_all(current_price, timestamp, "REVERSAL")
                                self.log(f"🔄 FLIP TO LONG: ${current_price:.2f}")
                            self.manager.enter('long', current_price, timestamp, tp_pct, sl_pct)
                            
                        elif final_signal == -1:
                            if self.manager.current_side == 'long':
                                self.manager.close_all(current_price, timestamp, "REVERSAL")
                                self.log(f"🔄 FLIP TO SHORT: ${current_price:.2f}")
                            self.manager.enter('short', current_price, timestamp, tp_pct, sl_pct)

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
                     adx_col = find_col(df, "ADX")
                     if not adx_col: adx_col = "ADX_14"
                     if adx_col in df.columns:
                         df['comb'] = np.where(df[adx_col] > thresh, df[sigs[0]], df[sigs[1]])
                     else:
                         df['comb'] = df.apply(lambda r: 1 if all(r[c]==1 for c in sigs) else (-1 if all(r[c]==-1 for c in sigs) else 0), axis=1)
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
