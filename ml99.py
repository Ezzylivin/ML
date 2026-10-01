# File: ml.py
# 🚀 UPGRADE: v8.0 - Master Server (Bot Engine + Backtester + API)
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
from fastapi import FastAPI, HTTPException, BackgroundTasks
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

app = FastAPI(title="Trading ML Server API v8.0")

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
    # If config is BotConfig or BacktestConfig, extract params
    if hasattr(config, 'params'):
        merged = deepcopy(config.params)
        if hasattr(config, 'strategies') and config.strategies:
            for strat in config.strategies:
                # Handle Pydantic object or Dict
                p = strat.params if hasattr(strat, 'params') else strat.get('params', {})
                for k, v in p.items(): merged[k] = v
        return merged
    return config

def find_col(df, key_fragment):
    if key_fragment in df.columns: return key_fragment
    for col in df.columns:
        if col.lower() == key_fragment.lower(): return col
    return None

def load_efficient_data(symbol, timeframe, start_date=None, end_date=None):
    safe_symbol = symbol.replace('/', '-')
    data_filename = f"{safe_symbol}-{timeframe}.csv"
    data_path = os.path.join(DATA_DIR, data_filename)
    if not os.path.exists(data_path): return pd.DataFrame()
    
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

# --- FEATURE ENGINEERING & SIGNALS ---
def engineer_features_for_backtest(df, params):
    if params is None: params = {} 
    rsi_len = safe_int(params.get('rsi_length'), 14)
    bb_len = safe_int(params.get('bb_length'), 20)
    bb_std = safe_float(params.get('bb_std'), 2.0)
    sma_fast = safe_int(params.get('sma_fast_period'), 10)
    sma_slow = safe_int(params.get('sma_slow_period'), 50)
    k_per = safe_int(params.get('k_period'), 14)
    d_per = safe_int(params.get('d_period'), 3)
    cci_len = safe_int(params.get('cci_length'), 20)
    atr_len = safe_int(params.get('atr_period'), 14)

    df = df.copy()
    df[['open','high','low','close']] = df[['open','high','low','close']].fillna(method='ffill')
    
    try:
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.sma(length=50, append=True)
        
        if rsi_len != 14: df.ta.rsi(length=rsi_len, append=True)
        df.ta.bbands(length=bb_len, std=bb_std, append=True)
        df.ta.stoch(k=k_per, d=d_per, smooth_k=3, append=True)
        df.ta.cci(length=cci_len, append=True)
        df.ta.psar(append=True)
        if atr_len != 14: df.ta.atr(length=atr_len, append=True)
        
        df.ta.obv(append=True)
        obv_ma = safe_int(params.get('obv_ma_period'), 20)
        if 'OBV' in df.columns: df[f'OBV_SMA_{obv_ma}'] = df['OBV'].rolling(window=obv_ma).mean()

        df.ta.ichimoku(append=True)
        if sma_fast > 0: df.ta.sma(length=sma_fast, append=True)
        if sma_slow > 0: df.ta.sma(length=sma_slow, append=True)
        df.ta.ema(length=20, append=True)
        
        # MACD Explicit
        mf = safe_int(params.get('macd_fast_period'), 12)
        ms = safe_int(params.get('macd_slow_period'), 26)
        msig = safe_int(params.get('macd_signal_period'), 9)
        df.ta.macd(fast=mf, slow=ms, signal=msig, append=True)

    except: pass
    return df

def generate_ta_signals(df, strategy_code, params):
    df = df.copy()
    df['ta_signal'] = 0
    if params is None: params = {}
    
    try:
        # Ensure indicators exist (lazy load)
        df = engineer_features_for_backtest(df, params)

        if strategy_code == "psar_signal":
            psarl = find_col(df, "PSARl"); psars = find_col(df, "PSARs")
            if psarl and psars:
                df.loc[df[psarl].notna() & (df[psarl] > 0), 'ta_signal'] = 1
                df.loc[df[psars].notna() & (df[psars] > 0), 'ta_signal'] = -1
        elif strategy_code == "bollinger_bands":
            bbl = find_col(df, "BBL_"); bbu = find_col(df, "BBU_")
            if bbl and bbu:
                df.loc[df['close'] <= df[bbl], 'ta_signal'] = 1
                df.loc[df['close'] >= df[bbu], 'ta_signal'] = -1
        elif strategy_code == "rsi_divergence":
             rsi_len = safe_int(params.get('rsi_length'), 14)
             rsi_col = find_col(df, f'RSI_{rsi_len}')
             if rsi_col:
                 os = safe_float(params.get('oversold_level'), 30)
                 ob = safe_float(params.get('overbought_level'), 70)
                 df.loc[df[rsi_col] < os, 'ta_signal'] = 1
                 df.loc[df[rsi_col] > ob, 'ta_signal'] = -1
        elif strategy_code == "sma_crossover":
            f = safe_int(params.get('sma_fast_period'), 10)
            s = safe_int(params.get('sma_slow_period'), 50)
            sma_f = find_col(df, f'SMA_{f}'); sma_s = find_col(df, f'SMA_{s}')
            if sma_f and sma_s:
                df.loc[(df[sma_f] > df[sma_s]) & (df[sma_f].shift(1) <= df[sma_s].shift(1)), 'ta_signal'] = 1
                df.loc[(df[sma_f] < df[sma_s]) & (df[sma_f].shift(1) >= df[sma_s].shift(1)), 'ta_signal'] = -1
        elif strategy_code == "macd_crossover":
             m_col = find_col(df, "MACD_"); s_col = find_col(df, "MACDs_")
             if m_col and s_col:
                 df.loc[(df[m_col] > df[s_col]) & (df[m_col].shift(1) <= df[s_col].shift(1)), 'ta_signal'] = 1
                 df.loc[(df[m_col] < df[s_col]) & (df[m_col].shift(1) >= df[s_col].shift(1)), 'ta_signal'] = -1
        elif strategy_code == "stochastic_crossover":
             k_col = find_col(df, "STOCHk"); d_col = find_col(df, "STOCHd")
             if k_col and d_col:
                 df.loc[(df[k_col] > df[d_col]) & (df[k_col].shift(1) <= df[d_col].shift(1)) & (df[k_col] < 20), 'ta_signal'] = 1
                 df.loc[(df[k_col] < df[d_col]) & (df[k_col].shift(1) >= df[d_col].shift(1)) & (df[k_col] > 80), 'ta_signal'] = -1
        elif strategy_code == "atr_breakout":
             atr_p = safe_int(params.get('atr_period'), 14)
             mult = safe_float(params.get('atr_multiplier'), 2.0)
             atr_col = find_col(df, f"ATR_{atr_p}"); ema_col = find_col(df, "EMA_20")
             if atr_col and ema_col:
                 upper = df[ema_col] + (df[atr_col] * mult)
                 lower = df[ema_col] - (df[atr_col] * mult)
                 df.loc[df['close'] > upper, 'ta_signal'] = 1
                 df.loc[df['close'] < lower, 'ta_signal'] = -1
        elif strategy_code == "cci_oversold":
             cci_len = safe_int(params.get('cci_length'), 20)
             col_name = f"CCI_MANUAL_{cci_len}"
             cci_col = find_col(df, "CCI")
             if not cci_col: cci_col = col_name # fallback logic simplified
             if cci_col in df.columns:
                 low_t = safe_float(params.get('cci_oversold'), -100)
                 high_t = safe_float(params.get('cci_overbought'), 100)
                 df.loc[df[cci_col] < low_t, 'ta_signal'] = 1
                 df.loc[df[cci_col] > high_t, 'ta_signal'] = -1
        elif strategy_code == "ichimoku_cloud":
             span_a = find_col(df, "ISA_"); span_b = find_col(df, "ISB_")
             if span_a and span_b:
                 df.loc[(df['close'] > df[span_a]) & (df['close'] > df[span_b]) & (df['close'].shift(1) <= df[span_a].shift(1)), 'ta_signal'] = 1
                 df.loc[(df['close'] < df[span_a]) & (df['close'] < df[span_b]) & (df['close'].shift(1) >= df[span_b].shift(1)), 'ta_signal'] = -1
        elif strategy_code == "obv_signal":
             obv_ma = safe_int(params.get('obv_ma_period'), 20)
             obv_sma = find_col(df, f"OBV_SMA_{obv_ma}")
             if obv_sma and 'OBV' in df.columns:
                 df.loc[df['OBV'] > df[obv_sma], 'ta_signal'] = 1
                 df.loc[df['OBV'] < df[obv_sma], 'ta_signal'] = -1

    except: pass
    return df

def run_backtest(df, signal_col, initial_balance, fee):
    balance = float(initial_balance)
    fee_pct = float(fee)
    slippage_pct = float(SLIPPAGE_PCT)
    position = 0
    trades = []
    equity_curve = []
    entry_price = 0.0
    entry_time = None
    entry_balance = 0.0
    
    equity_curve.append({"timestamp": df.index[0].isoformat(), "balance": balance})

    for i in range(1, len(df)):
        price = float(df['close'].iloc[i])
        curr_time = df.index[i]
        sig = df[signal_col].iloc[i]
        
        if position == 0:
            if sig == 1: 
                position = 1; entry_price = price * (1 + slippage_pct); entry_time = curr_time; entry_balance = balance
            elif sig == -1: 
                position = -1; entry_price = price * (1 - slippage_pct); entry_time = curr_time; entry_balance = balance
        elif position == 1 and sig == -1:
            exit_price = price * (1 - slippage_pct)
            raw_pnl = (exit_price - entry_price) / entry_price
            total_fees = (entry_balance * fee_pct) + (balance * (1 + raw_pnl) * fee_pct)
            profit = (entry_balance * raw_pnl) - total_fees
            balance += profit
            trades.append({"entryTime": entry_time.isoformat(), "exitTime": curr_time.isoformat(), "profit": profit, "result": "win" if profit > 0 else "loss"})
            position = 0
        elif position == -1 and sig == 1:
            exit_price = price * (1 + slippage_pct)
            raw_pnl = (entry_price - exit_price) / entry_price
            total_fees = (entry_balance * fee_pct) + (balance * (1 + raw_pnl) * fee_pct)
            profit = (entry_balance * raw_pnl) - total_fees
            balance += profit
            trades.append({"entryTime": entry_time.isoformat(), "exitTime": curr_time.isoformat(), "profit": profit, "result": "win" if profit > 0 else "loss"})
            position = 0
            
        equity_curve.append({"timestamp": curr_time.isoformat(), "balance": balance})

    total_return = ((balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0
    wins = [t for t in trades if t['profit'] > 0]
    max_dd = 0.0
    balances = [p['balance'] for p in equity_curve]
    if balances:
        peak = balances[0]
        for b in balances:
            if b > peak: peak = b
            dd = (peak - b) / peak * 100
            if dd > max_dd: max_dd = dd

    return {
        "metrics": {
            "totalReturn": total_return, "finalBalance": balance, "totalTrades": len(trades),
            "winningTrades": len(wins), "losingTrades": len(trades) - len(wins),
            "winRate": (len(wins)/len(trades)*100) if trades else 0, "maxDrawdown": max_dd
        },
        "equityCurve": equity_curve, "tradeBreakdown": trades, 
        "candleData": df[['open','high','low','close','volume']].reset_index().to_dict('records')
    }

# --- PAPER TRADING BOT CLASS (The Engine) ---
class PaperTradingBot:
    def __init__(self, config: BotConfig):
        self.config = config
        self.is_running = False
        self.logs = []
        self.trades = []
        self.current_balance = config.capitalAllocation
        self.position = None 
        self.entry_price = 0.0
        self.df = None 

    def log(self, message):
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] {message}")
        self.logs.append({"timestamp": timestamp, "message": message, "type": "info"})

    def get_profit(self):
        return self.current_balance - self.config.capitalAllocation

    def get_win_rate(self):
        if not self.trades: return 0.0
        wins = len([t for t in self.trades if t['profit'] > 0])
        return (wins / len(self.trades)) * 100

    def run(self):
        self.is_running = True
        self.log(f"Bot started with {self.config.symbol} on {self.config.timeframe} timeframe.")
        print(f"✅ [BOT START] {self.config.symbol} | Strategies: {len(self.config.strategies)}")

        try:
            # 🚀 1. Initial Data Load
            print("⏳ Fetching initial data...")
            df = load_efficient_data(self.config.symbol, self.config.timeframe)
            if not df.empty:
                self.df = df
                self.log(f"Initial data loaded: {len(df)} candles.")
            else:
                self.log("Warning: Initial data fetch returned empty.")

            # 🚀 2. The Loop
            while self.is_running:
                try:
                    # Refresh Data
                    df = load_efficient_data(self.config.symbol, self.config.timeframe) 
                    if df.empty:
                        self.log("Error: No data received.")
                        time.sleep(10)
                        continue
                    
                    self.df = df
                    
                    # --- SIGNAL GENERATION ---
                    norm_params = self.config.params
                    sigs = []
                    
                    for i, s in enumerate(self.config.strategies):
                        s_params = norm_params.copy()
                        s_params.update(s.params) 
                        
                        df = generate_ta_signals(df, s.code, s_params)
                        col = f's_{i}'
                        df[col] = df['ta_signal']
                        sigs.append(col)

                    # --- COMBINATION LOGIC ---
                    last_row = df.iloc[-1]
                    final_signal = 0
                    mode = self.config.params.get("hybridMode", "AND")
                    
                    if len(sigs) >= 2:
                        if mode == "AND":
                            if all(last_row[c] == 1 for c in sigs): final_signal = 1
                            elif all(last_row[c] == -1 for c in sigs): final_signal = -1
                        elif mode == "REGIME":
                            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
                            if adx is not None and not adx.empty:
                                current_adx = adx.iloc[-1, 0] 
                                thresh = int(self.config.params.get("regime_threshold", 25))
                                active_strat = sigs[0] if current_adx > thresh else sigs[1]
                                final_signal = last_row[active_strat]
                        else: 
                             vals = [last_row[c] for c in sigs]
                             if 1 in vals and -1 not in vals: final_signal = 1
                             elif -1 in vals and 1 not in vals: final_signal = -1
                    elif sigs:
                        final_signal = last_row[sigs[0]]

                    # --- ML FILTER ---
                    if self.config.mlMode in ['on', 'predictions'] and self.config.mlModel:
                        pass 

                    # --- EXECUTION ---
                    current_price = last_row['close']
                    self.log(f"Price: {current_price:.2f} | Sig: {final_signal} | Mode: {mode}")

                    if self.position is None and final_signal == 1:
                        self.position = 'long'
                        self.entry_price = current_price
                        self.log(f"🟢 BUY EXECUTED at ${current_price:.2f}")
                        
                    elif self.position == 'long' and final_signal == -1:
                        profit = (current_price - self.entry_price) * (self.config.capitalAllocation / self.entry_price)
                        self.current_balance += profit
                        self.trades.append({
                            "entryTime": datetime.now().isoformat(),
                            "exitTime": datetime.now().isoformat(),
                            "price": self.entry_price,
                            "exitPrice": current_price,
                            "profit": profit,
                            "position": "long"
                        })
                        self.position = None
                        self.log(f"🔴 SELL EXECUTED. Profit: ${profit:.2f}")

                    time.sleep(60)

                except Exception as inner_e:
                    self.log(f"Loop Error: {str(inner_e)}")
                    time.sleep(10)

        except Exception as e:
             self.log(f"Critical Bot Crash: {str(e)}")
             self.is_running = False

# ==============================================================================
# 5. API ENDPOINTS
# ==============================================================================

@app.post('/api/ml/run-backtest-on') 
@app.post('/api/ml/run-combo-backtest')
async def handle_run_combo_backtest(config: BacktestConfig):
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

        res = run_backtest(df, 'comb', config.initialBalance, config.fee)
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
    
    bot_instance = PaperTradingBot(config)
    
    # 🚀 FIX: Use Threading (Not BackgroundTasks) for infinite loop
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
        status_data["logs"] = bot_instance.logs[-50:]
        status_data["trades"] = bot_instance.trades
        status_data["currentBalance"] = bot_instance.current_balance
        
        if bot_instance.df is not None:
            try:
                tail = bot_instance.df.tail(100).copy()
                if isinstance(tail.index, pd.DatetimeIndex): tail['time'] = tail.index.astype(str)
                elif 'timestamp' in tail.columns: tail['time'] = tail['timestamp'].astype(str)
                status_data["candles"] = tail.to_dict(orient='records')
            except: pass

        status_data["performanceMetrics"] = {
            "totalProfit": bot_instance.get_profit(),
            "winRate": bot_instance.get_win_rate(),
            "totalTrades": len(bot_instance.trades),
            "maxDrawdown": 0
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
