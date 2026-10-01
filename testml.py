# File: ml.py
# 🚀 UPGRADE: v29.0 - DEBUG EDITION (Verbose Logging Enabled)
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
import yfinance as yf 
from fastapi import FastAPI, HTTPException, Request
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

app = FastAPI(title="Trading ML Server API v29.0 (Debug)")

# Global Bot Instance
bot_instance = None

# --- 2. MODELS ---
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

def normalize_params(config):
    if hasattr(config, 'params'):
        merged = deepcopy(config.params)
        if hasattr(config, 'strategies') and config.strategies:
            for strat in config.strategies:
                p = strat.params if hasattr(strat, 'params') else strat.get('params', {})
                for k, v in p.items(): merged[k] = v
        return merged
    return config

def find_col(df, key_fragment):
    # 1. Exact match
    if key_fragment in df.columns: return key_fragment
    # 2. Case-insensitive exact
    for col in df.columns:
        if col.lower() == key_fragment.lower(): return col
    # 3. Starts-with
    for col in df.columns:
        if col.lower().startswith(key_fragment.lower()): return col
    
    # ❌ DEBUG: Log missing columns if not found
    # print(f"⚠️ [DEBUG] Could not find column matching '{key_fragment}' in {list(df.columns)}")
    return None

# 🚀 LIVE DATA FETCHING
def fetch_live_data(symbol, timeframe):
    try:
        tf_map = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '4h': '1h', '1d': '1d'}
        interval = tf_map.get(timeframe, '1h')
        period = "5d" if timeframe in ['1m', '5m', '15m'] else "1mo"
        
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

# LOCAL DATA LOADING
def load_efficient_data(symbol, timeframe, start_date=None, end_date=None):
    safe_symbol = symbol.replace('/', '-')
    data_filename = f"{safe_symbol}-{timeframe}.csv"
    data_path = os.path.join(DATA_DIR, data_filename)
    
    if not os.path.exists(data_path):
        print(f"📥 Fetching missing data for {symbol}...")
        df = fetch_live_data(symbol, timeframe) 
        if not df.empty:
            df.to_csv(data_path)
            return df
        return pd.DataFrame()
    
    df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df.index = df.index.tz_localize('UTC')
    
    if start_date and end_date:
        try: 
            buffer = pd.to_datetime(start_date, utc=True) - pd.Timedelta(days=365)
            df = df.loc[buffer:pd.to_datetime(end_date, utc=True)].copy()
        except: pass
    
    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True, errors='ignore')
    return df

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

# --- SIGNAL GENERATION ---
def engineer_features_for_backtest(df, params):
    if params is None: params = {} 
    try:
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.sma(length=50, append=True)
    except: pass
    return df

def generate_ta_signals(df, strategy_code, params):
    df = df.copy()
    df['ta_signal'] = 0
    if params is None: params = {}
    
    # 🔍 DEBUG PRINT
    print(f"⚙️ GEN SIGNALS: {strategy_code} | Params: {params}")
    
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
             
             # DEBUG
             if not rsi_col: print(f"❌ RSI Column Missing! Looked for RSI_{rsi_len}")

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
             
             # DEBUG
             if not atr_col: print(f"❌ ATR Column Missing! Looked for ATR_{atr_p}")

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
        traceback.print_exc()
    return df

class PyramidManager:
    def __init__(self, capital, fee, slippage, max_levels=1):
        self.total_capital = float(capital)
        self.available_cash = float(capital)
        self.fee = float(fee)
        self.slippage = float(slippage)
        self.max_levels = int(max_levels) if int(max_levels) > 0 else 1
        self.positions = [] 
        self.completed_trades = []
        self.current_side = None 

    def enter(self, side, price, time):
        price = float(price)
        if self.current_side and self.current_side != side:
            self.close_all(price, time)
        
        if len(self.positions) >= self.max_levels: return
        
        if self.positions:
            last_price = self.positions[-1]['entry_price']
            diff_pct = abs(price - last_price) / last_price
            if diff_pct < 0.005: 
                return 

        allocation = self.total_capital / self.max_levels
        if allocation > self.available_cash: allocation = self.available_cash
        if allocation < 10: return 
        
        entry_price = price * (1 + self.slippage) if side == 'long' else price * (1 - self.slippage)
        fee_amt = allocation * self.fee
        net_size = allocation - fee_amt
        qty = net_size / entry_price
        
        self.positions.append({
            "side": side, "entry_price": entry_price, "qty": qty,
            "entry_time": time, "invested": allocation 
        })
        self.available_cash -= allocation
        self.current_side = side

    def reduce_position(self, price, time):
        if not self.positions: return
        
        pos = self.positions.pop(0) 
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
        self.available_cash += net_return
        
        self.completed_trades.append({
            "entryTime": pos['entry_time'], "exitTime": time,
            "price": pos['entry_price'], "exitPrice": exit_price,
            "profit": profit, "position": pos['side'],
            "type": "scale_out"
        })
        
        if not self.positions: self.current_side = None

    def close_all(self, price, time):
        while self.positions:
            self.reduce_position(price, time)

    def get_equity(self, current_price):
        equity = self.available_cash
        for pos in self.positions:
             if pos['side'] == 'long': equity += pos['qty'] * current_price
             else:
                 diff = pos['entry_price'] - current_price
                 equity += (pos['invested'] + diff * pos['qty'])
        return equity

def run_backtest(df, signal_col, initial_balance, fee, max_pyramiding=1):
    manager = PyramidManager(initial_balance, fee, SLIPPAGE_PCT, max_pyramiding)
    equity_curve = [{"timestamp": df.index[0].isoformat(), "balance": initial_balance}]

    for i in range(1, len(df)):
        price = float(df['close'].iloc[i])
        curr_time = df.index[i].isoformat()
        sig = df[signal_col].iloc[i]
        
        if sig == 1: manager.enter('long', price, curr_time)
        elif sig == -1: 
            if manager.current_side == 'long': 
                manager.reduce_position(price, curr_time)
            else: 
                manager.enter('short', price, curr_time)
            
        equity_curve.append({"timestamp": curr_time, "balance": manager.get_equity(price)})

    if not df.empty:
        manager.close_all(df['close'].iloc[-1], df.index[-1].isoformat())
    
    final_balance = manager.available_cash
    trades = manager.completed_trades
    wins = [t for t in trades if t['profit'] > 0]
    
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
            "winRate": (len(wins)/len(trades)*100) if trades else 0, "maxDrawdown": max_dd
        },
        "equityCurve": equity_curve, "tradeBreakdown": trades, 
        "candleData": df[['open','high','low','close','volume']].reset_index().to_dict('records')
    }

class PaperTradingBot:
    def __init__(self, config: BotConfig):
        self.config = config
        self.is_running = False
        self.logs = []
        
        level = config.maxPyramiding
        if not level or level < 1: level = int(config.params.get('maxPyramiding', 1))
        fee = float(config.params.get('fee', DEFAULT_TAKER_FEE))
        
        self.manager = PyramidManager(config.capitalAllocation, fee, SLIPPAGE_PCT, level)
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
        self.log(f"Bot started. {self.config.symbol} {self.config.timeframe}. Pyramiding: {self.manager.max_levels}x")

        try:
            df = fetch_live_data(self.config.symbol, self.config.timeframe)
            if not df.empty:
                self.df = df
                self.log(f"Live data loaded: {len(df)} candles. Last close: {df['close'].iloc[-1]:.2f}")
            
            while self.is_running:
                for _ in range(60): 
                    if not self.is_running: break
                    time.sleep(1)
                if not self.is_running: break

                try:
                    df = fetch_live_data(self.config.symbol, self.config.timeframe) 
                    if df.empty: continue
                    self.df = df
                    
                    norm_params = self.config.params
                    sigs = []
                    for i, s in enumerate(self.config.strategies):
                        s_params = norm_params.copy(); s_params.update(s.params) 
                        df = generate_ta_signals(df, s.code, s_params)
                        col = f's_{i}'; df[col] = df['ta_signal']; sigs.append(col)

                    last_row = df.iloc[-1]
                    final_signal = 0
                    mode = self.config.params.get("hybridMode", "AND")
                    
                    if len(sigs) >= 2:
                        if mode == "AND":
                            if all(last_row[c] == 1 for c in sigs): final_signal = 1
                            elif all(last_row[c] == -1 for c in sigs): final_signal = -1
                        elif mode == "REGIME":
                            if 'ADX_14' not in df.columns:
                                try: df.ta.adx(length=14, append=True); last_row = df.iloc[-1]
                                except: pass
                            adx_col = find_col(df, "ADX")
                            thresh = int(self.config.params.get("regime_threshold", 25))
                            current_adx = last_row[adx_col] if adx_col else 0
                            active_strat = sigs[0] if current_adx > thresh else sigs[1]
                            final_signal = last_row[active_strat]
                        else: 
                             vals = [last_row[c] for c in sigs]
                             if 1 in vals and -1 not in vals: final_signal = 1
                             elif -1 in vals and 1 not in vals: final_signal = -1
                    elif sigs: final_signal = last_row[sigs[0]]

                    current_price = float(last_row['close'])
                    timestamp = datetime.now().isoformat()
                    
                    if final_signal == 1:
                        self.manager.enter('long', current_price, timestamp)
                        self.log(f"🟢 SIGNAL BUY: ${current_price:.2f}")
                    elif final_signal == -1:
                        if self.manager.current_side == 'long':
                            self.manager.reduce_position(current_price, timestamp)
                            self.log(f"🔴 SIGNAL SELL (Scale Out): ${current_price:.2f}")
                        else:
                            self.manager.enter('short', current_price, timestamp)
                            self.log(f"🔻 SIGNAL SHORT: ${current_price:.2f}")

                except Exception as inner:
                    print(f"Loop Error: {inner}")

        except Exception as e:
             self.log(f"Crash: {e}")
             self.is_running = False

@app.post('/api/ml/run-backtest-on') 
@app.post('/api/ml/run-combo-backtest')
async def handle_run_combo_backtest(config: BacktestConfig):
    # 🔍 DEBUG: PRINT RECEIVED PAYLOAD
    print(f"\n🔵 DEBUG: Received Config: {config.dict()}")

    try:
        norm_params = normalize_params(config)
        df = load_efficient_data(config.symbol, config.timeframe, config.startDate, config.endDate)
        if df.empty: raise HTTPException(400, "No data found")
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
        
        res = run_backtest(df, 'comb', config.initialBalance, config.fee, pyramid_lvl)
        
        clean_res = convert_numpy_types(res)
        response_payload = deepcopy(clean_res)
        response_payload["combinedResult"] = deepcopy(clean_res)
        return JSONResponse(content=response_payload)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))

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
async def start_bot_endpoint(config: BotConfig, background_tasks: BackgroundTasks):
    global bot_instance
    if bot_instance and bot_instance.is_running:
        return {"status": "error", "message": "Bot is already running"}
    
    # DEBUG
    print(f"🚀 BOT START REQUEST: {config.dict()}")

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
    status_data = {"status": "stopped", "logs": [], "trades": [], "currentBalance": 0, "candles": []}

    if bot_instance:
        status_data["status"] = "running" if bot_instance.is_running else "stopped"
        status_data["logs"] = list(bot_instance.logs[-50:])
        status_data["trades"] = list(bot_instance.manager.completed_trades) # Use Manager Trades
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
                    status_data["candles"] = tail[keep_cols].to_dict(orient='records')
            except: pass

        status_data["performanceMetrics"] = {
            "totalProfit": bot_instance.get_profit(),
            "winRate": bot_instance.get_win_rate(),
            "totalTrades": len(bot_instance.trades),
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
