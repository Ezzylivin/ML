# File: ml.py
# 🚀 UPGRADE: v5.8 - Fixed 'CertifyConfig' NameError + All Math/Logic Fixes
import os
import json
import pandas as pd
import numpy as np
import joblib
import logging
import typing
import math
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas_ta as ta
import warnings
from copy import deepcopy
from decimal import Decimal, getcontext
from datetime import datetime, timezone, timedelta

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
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Trading Constants
DEFAULT_TAKER_FEE = 0.006   # 0.1%
SLIPPAGE_PCT = 0.001        # 0.1%

app = FastAPI(title="Trading ML Server API v5.8")

# ==============================================================================
# 2. PYDANTIC MODELS (DEFINED BEFORE USE)
# ==============================================================================

class BacktestConfig(BaseModel):
    symbol: str
    timeframe: str
    startDate: str
    endDate: str
    mlMode: str = 'off'
    mlModel: Optional[str] = None
    mlThreshold: float = 0.65
    code: Optional[str] = None                  # For Single Strategy
    strategies: Optional[List[Dict[str, Any]]] = None # For Combo Strategy
    initialBalance: float = 1000.0
    fee: float = DEFAULT_TAKER_FEE
    params: Dict[str, Any] = Field(default_factory=dict)

# 🚀 FIX: Defined CertifyConfig here so it exists for the endpoint below
class CertifyConfig(BaseModel):
    symbol: str
    timeframe: str
    best_params: Dict[str, Any] 

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

def safe_int(val, default=0):
    try: return int(float(val))
    except: return default

def safe_float(val, default=0.0):
    try: return float(val)
    except: return default

def normalize_params(config: BacktestConfig) -> Dict:
    """Merges global params with strategy-specific params for feature engineering"""
    merged = deepcopy(config.params)
    
    # If combo, extract params from list
    if config.strategies:
        for strat in config.strategies:
            p = strat.get('params', {})
            for k, v in p.items():
                merged[k] = v
    return merged

def find_col(df, key_fragment):
    """Robust column finder that prefers exact matches but falls back to starts_with"""
    if key_fragment in df.columns: return key_fragment
    for col in df.columns:
        if col.lower() == key_fragment.lower(): return col
    cols = sorted(df.columns, key=len, reverse=True)
    for col in cols:
        if col.upper().startswith(key_fragment.upper()): return col
    return None

def load_efficient_data(symbol, timeframe, start_date, end_date):
    safe_symbol = symbol.replace('/', '-')
    data_filename = f"{safe_symbol}-{timeframe}.csv"
    data_path = os.path.join(DATA_DIR, data_filename)
    if not os.path.exists(data_path): raise FileNotFoundError(f"Data not found: {data_path}")
    
    # Read CSV
    df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    
    # Timezone Safety
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.tz_localize('UTC')
    
    # Filter Date Range
    try: 
        buffer_date = pd.to_datetime(start_date, utc=True) - pd.Timedelta(days=365)
        df = df.loc[buffer_date:pd.to_datetime(end_date, utc=True)].copy()
    except Exception as e:
        logger.error(f"Date filtering error: {e}")

    if df.empty:
        raise ValueError("Dataframe empty after date filtering.")

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

# --- FEATURE ENGINEERING ---
def engineer_features_for_backtest(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    if params is None: params = {} 
    
    # Extract params with defaults
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
    # Fill NaNs in price data first
    df[['open','high','low','close']] = df[['open','high','low','close']].fillna(method='ffill')
    df.replace([np.inf, -np.inf], np.nan, inplace=True) 

    try:
        # Core Indicators (Always generated for ML)
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True) 
        df.ta.sma(length=50, append=True)
        df.ta.sma(length=200, append=True)
        
        # Aliasing
        if 'RSI_14' in df.columns: df['RSI'] = df['RSI_14']
        if 'ATR_14' in df.columns: df['ATR'] = df['ATR_14']
        if 'ADX_14' in df.columns: df['ADX'] = df['ADX_14']
        if 'SMA_50' in df.columns: df['SMA50'] = df['SMA_50']
        if 'SMA_200' in df.columns: df['SMA200'] = df['SMA_200']
        if 'volume' in df.columns: df['Volume'] = df['volume']

        # Strategy Specifics
        if rsi_len != 14: df.ta.rsi(length=rsi_len, append=True)
        df.ta.bbands(length=bb_len, std=bb_std, append=True)
        df.ta.stoch(k=k_per, d=d_per, smooth_k=3, append=True)
        df.ta.cci(length=cci_len, append=True)
        df.ta.psar(append=True) 
        if atr_len != 14: df.ta.atr(length=atr_len, append=True)
        
        df.ta.obv(append=True)
        obv_ma = safe_int(params.get('obv_ma_period'), 20)
        if 'OBV' in df.columns: 
            df[f'OBV_SMA_{obv_ma}'] = df['OBV'].rolling(window=obv_ma).mean()

        df.ta.ichimoku(append=True)
        if sma_fast > 0: df.ta.sma(length=sma_fast, append=True)
        if sma_slow > 0: df.ta.sma(length=sma_slow, append=True)
        df.ta.ema(length=20, append=True)

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
        # PSAR
        if strategy_code == "psar_signal":
            psarl = find_col(df, "PSARl") 
            psars = find_col(df, "PSARs")
            if psarl and psars:
                df.loc[df[psarl].notna() & (df[psarl] > 0), 'ta_signal'] = 1
                df.loc[df[psars].notna() & (df[psars] > 0), 'ta_signal'] = -1

        # Bollinger
        elif strategy_code == "bollinger_bands":
            bbl = find_col(df, "BBL_")
            bbu = find_col(df, "BBU_")
            if bbl and bbu:
                df.loc[df['close'] <= df[bbl], 'ta_signal'] = 1 
                df.loc[df['close'] >= df[bbu], 'ta_signal'] = -1 
        
        # RSI
        elif strategy_code == "rsi_divergence":
             rsi_len = safe_int(params.get('rsi_length'), 14)
             rsi_col = find_col(df, f'RSI_{rsi_len}')
             oversold = safe_float(params.get('oversold_level'), 30)
             overbought = safe_float(params.get('overbought_level'), 70)
             if rsi_col:
                 df.loc[df[rsi_col] < oversold, 'ta_signal'] = 1
                 df.loc[df[rsi_col] > overbought, 'ta_signal'] = -1

        # SMA
        elif strategy_code == "sma_crossover":
            f = safe_int(params.get('sma_fast_period'), 10)
            s = safe_int(params.get('sma_slow_period'), 50)
            sma_f = find_col(df, f'SMA_{f}')
            sma_s = find_col(df, f'SMA_{s}')
            if sma_f and sma_s:
                df.loc[(df[sma_f] > df[sma_s]) & (df[sma_f].shift(1) <= df[sma_s].shift(1)), 'ta_signal'] = 1
                df.loc[(df[sma_f] < df[sma_s]) & (df[sma_f].shift(1) >= df[sma_s].shift(1)), 'ta_signal'] = -1

        # Stochastic
        elif strategy_code == "stochastic_crossover":
             k_col = find_col(df, "STOCHk")
             d_col = find_col(df, "STOCHd")
             if k_col and d_col:
                 df.loc[(df[k_col] > df[d_col]) & (df[k_col].shift(1) <= df[d_col].shift(1)) & (df[k_col] < 20), 'ta_signal'] = 1
                 df.loc[(df[k_col] < df[d_col]) & (df[k_col].shift(1) >= df[d_col].shift(1)) & (df[k_col] > 80), 'ta_signal'] = -1
        
        # MACD
        elif strategy_code == "macd_crossover":
             m_col = find_col(df, "MACD_")
             s_col = find_col(df, "MACDs_")
             # Force calc if missing
             if not m_col or not s_col:
                 mf = safe_int(params.get('macd_fast_period'), 12)
                 ms = safe_int(params.get('macd_slow_period'), 26)
                 msig = safe_int(params.get('macd_signal_period'), 9)
                 macd_df = df.ta.macd(fast=mf, slow=ms, signal=msig)
                 if macd_df is not None:
                     df = pd.concat([df, macd_df], axis=1)
                     m_col = find_col(df, "MACD_")
                     s_col = find_col(df, "MACDs_")

             if m_col and s_col:
                 df.loc[(df[m_col] > df[s_col]) & (df[m_col].shift(1) <= df[s_col].shift(1)), 'ta_signal'] = 1
                 df.loc[(df[m_col] < df[s_col]) & (df[m_col].shift(1) >= df[s_col].shift(1)), 'ta_signal'] = -1

        # Ichimoku
        elif strategy_code == "ichimoku_cloud":
             span_a = find_col(df, "ISA_")
             span_b = find_col(df, "ISB_")
             if span_a and span_b:
                 df.loc[(df['close'] > df[span_a]) & (df['close'] > df[span_b]) & (df['close'].shift(1) <= df[span_a].shift(1)), 'ta_signal'] = 1
                 df.loc[(df['close'] < df[span_a]) & (df['close'] < df[span_b]) & (df['close'].shift(1) >= df[span_b].shift(1)), 'ta_signal'] = -1

        # OBV
        elif strategy_code == "obv_signal":
             obv_ma = safe_int(params.get('obv_ma_period'), 20)
             obv_sma = find_col(df, f"OBV_SMA_{obv_ma}")
             if obv_sma and 'OBV' in df.columns:
                 df.loc[df['OBV'] > df[obv_sma], 'ta_signal'] = 1
                 df.loc[df['OBV'] < df[obv_sma], 'ta_signal'] = -1
        
        # CCI
        elif strategy_code == "cci_oversold":
             cci_len = safe_int(params.get('cci_length'), 20)
             col_name = f"CCI_MANUAL_{cci_len}"
             if col_name not in df.columns and 'CCI_14_0.015' not in df.columns:
                 try:
                     tp = (df['high'] + df['low'] + df['close']) / 3
                     sma_tp = tp.rolling(cci_len).mean()
                     mad = tp.rolling(cci_len).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
                     df[col_name] = (tp - sma_tp) / (0.015 * mad)
                     df[col_name] = df[col_name].fillna(0)
                 except: pass
             
             cci_col = find_col(df, "CCI")
             if not cci_col: cci_col = col_name

             low_t = safe_float(params.get('cci_oversold'), -100)
             high_t = safe_float(params.get('cci_overbought'), 100)
             if cci_col in df.columns:
                 df.loc[df[cci_col] < low_t, 'ta_signal'] = 1
                 df.loc[df[cci_col] > high_t, 'ta_signal'] = -1

        # ATR Breakout
        elif strategy_code == "atr_breakout":
             atr_p = safe_int(params.get('atr_period'), 14)
             mult = safe_float(params.get('atr_multiplier'), 2.0)
             atr_col = find_col(df, f"ATR_{atr_p}")
             if not atr_col: atr_col = find_col(df, "ATR")
             ema_col = find_col(df, "EMA_20")
             
             if atr_col and ema_col:
                 upper = df[ema_col] + (df[atr_col] * mult)
                 lower = df[ema_col] - (df[atr_col] * mult)
                 df.loc[df['close'] > upper, 'ta_signal'] = 1
                 df.loc[df['close'] < lower, 'ta_signal'] = -1

    except Exception as e: 
        logger.error(f"Signal Gen Error ({strategy_code}): {e}")
        
    return df

# ==============================================================================
# 4. BACKTEST ENGINE (CORRECTED MATH)
# ==============================================================================

def run_backtest(df, signal_col, initial_balance, fee):
    balance = float(initial_balance)
    fee_pct = float(fee)
    slippage_pct = float(SLIPPAGE_PCT)
    position = 0 # 0=Flat, 1=Long, -1=Short
    
    trades = []          
    equity_curve = []    
    
    entry_price = 0.0
    entry_time = None
    entry_balance = 0.0 # To calculate Trade Size
    
    equity_curve.append({
        "timestamp": df.index[0].isoformat(),
        "balance": balance
    })

    for i in range(1, len(df)):
        price = float(df['close'].iloc[i])
        curr_time = df.index[i]
        sig = df[signal_col].iloc[i]
        
        # --- ENTRY LOGIC ---
        if position == 0:
            if sig == 1: # GO LONG
                position = 1
                entry_price = price * (1 + slippage_pct)
                entry_time = curr_time
                entry_balance = balance 
                
            elif sig == -1: # GO SHORT
                position = -1
                entry_price = price * (1 - slippage_pct)
                entry_time = curr_time
                entry_balance = balance

        # --- EXIT LOGIC ---
        elif position == 1: # Close Long
            if sig == -1:
                exit_price = price * (1 - slippage_pct)
                raw_pnl_pct = (exit_price - entry_price) / entry_price
                total_fees = (entry_balance * fee_pct) + (balance * (1 + raw_pnl_pct) * fee_pct)
                net_profit = (entry_balance * raw_pnl_pct) - total_fees
                balance += net_profit
                
                trades.append({
                    "entryTime": entry_time.isoformat(),
                    "exitTime": curr_time.isoformat(),
                    "entryPrice": entry_price,
                    "exitPrice": exit_price,
                    "position": "long",
                    "size": entry_balance, 
                    "profit": net_profit,
                    "result": "win" if net_profit > 0 else "loss"
                })
                position = 0
                
        elif position == -1: # Close Short
            if sig == 1: 
                exit_price = price * (1 + slippage_pct)
                raw_pnl_pct = (entry_price - exit_price) / entry_price
                total_fees = (entry_balance * fee_pct) + (balance * (1 + raw_pnl_pct) * fee_pct)
                net_profit = (entry_balance * raw_pnl_pct) - total_fees
                balance += net_profit

                trades.append({
                    "entryTime": entry_time.isoformat(),
                    "exitTime": curr_time.isoformat(),
                    "entryPrice": entry_price,
                    "exitPrice": exit_price,
                    "position": "short",
                    "size": entry_balance,
                    "profit": net_profit,
                    "result": "win" if net_profit > 0 else "loss"
                })
                position = 0
                
        equity_curve.append({
            "timestamp": curr_time.isoformat(),
            "balance": balance
        })
    
    # Close Final Position
    if position != 0:
        last_price = float(df['close'].iloc[-1])
        if position == 1:
            exit_price = last_price * (1 - slippage_pct)
            raw_pnl_pct = (exit_price - entry_price) / entry_price
        else:
            exit_price = last_price * (1 + slippage_pct)
            raw_pnl_pct = (entry_price - exit_price) / entry_price
            
        total_fees = (entry_balance * fee_pct) + (balance * (1 + raw_pnl_pct) * fee_pct)
        net_profit = (entry_balance * raw_pnl_pct) - total_fees
        balance += net_profit
        
        trades.append({
            "entryTime": entry_time.isoformat(),
            "exitTime": df.index[-1].isoformat(),
            "entryPrice": entry_price,
            "exitPrice": exit_price,
            "position": "long" if position == 1 else "short",
            "size": entry_balance,
            "profit": net_profit,
            "result": "win" if net_profit > 0 else "loss"
        })

    # --- METRICS ---
    total_return = 0.0
    if initial_balance > 0:
        total_return = ((balance - initial_balance) / initial_balance) * 100
        
    wins = [t for t in trades if t['profit'] > 0]
    losses = [t for t in trades if t['profit'] <= 0]
    win_rate = (len(wins) / len(trades) * 100) if trades else 0

    balances = [p['balance'] for p in equity_curve]
    peak = balances[0]
    max_dd = 0.0
    for b in balances:
        if b > peak: peak = b
        dd = (peak - b) / peak * 100
        if dd > max_dd: max_dd = dd

    candle_data = []
    temp_df = df.copy()
    temp_df['time'] = temp_df.index.map(lambda x: x.isoformat())
    cols_to_keep = ['time', 'open', 'high', 'low', 'close', 'volume']
    if signal_col in temp_df.columns: cols_to_keep.append(signal_col)
    temp_df = temp_df.fillna(0)
    candle_data = temp_df[cols_to_keep].to_dict('records')

    result_structure = {
        "metrics": {
            "totalReturn": total_return,
            "finalBalance": balance,
            "totalTrades": len(trades),
            "winningTrades": len(wins),
            "losingTrades": len(losses),
            "winRate": win_rate,
            "maxDrawdown": max_dd,
        },
        "equityCurve": equity_curve,
        "tradeBreakdown": trades,  
        "candleData": candle_data  
    }
    return result_structure

# --- ENDPOINTS ---

@app.post('/api/ml/run-backtest-on') 
@app.post('/api/ml/run-combo-backtest')
async def handle_run_combo_backtest(config: BacktestConfig):
    try:
        norm_params = normalize_params(config)
        df = load_efficient_data(config.symbol, config.timeframe, config.startDate, config.endDate)
        
        if df.empty:
             raise HTTPException(400, "No data found for date range")

        df = engineer_features_for_backtest(df, norm_params)
        
        # LOGIC SPLIT
        if config.code:
             df = generate_ta_signals(df, config.code, norm_params)
             df['comb'] = df['ta_signal']
        elif config.strategies:
             sigs = []
             for i, s in enumerate(config.strategies):
                 df = generate_ta_signals(df, s['code'], norm_params)
                 df[f's_{i}'] = df['ta_signal']
                 sigs.append(f's_{i}')
             
             if len(sigs) >= 2:
                 df['comb'] = df[sigs].apply(lambda r: 1 if (r==1).any() else (-1 if (r==-1).any() else 0), axis=1)
             elif sigs:
                 df['comb'] = df[sigs[0]]
             else:
                 df['comb'] = 0
        else:
             df['comb'] = 0 

        # ML INFERENCE
        if config.mlMode in ['on', 'predictions'] and config.mlModel:
            try:
                model_path = os.path.join(MODEL_DIR, f"{config.mlModel}.joblib")
                if os.path.exists(model_path):
                    loaded = joblib.load(model_path)
                    if isinstance(loaded, dict):
                        model = loaded.get('model')
                        scaler = loaded.get('scaler')
                        feat_names = loaded.get('feature_names')
                    else:
                        model = loaded
                        scaler = None
                        feat_names = getattr(model, "feature_names_in_", None)
                    
                    X = df.copy().fillna(0)
                    if feat_names is not None:
                        for c in feat_names:
                            if c not in X.columns: X[c] = 0.0
                        X = X[feat_names]
                    else:
                        try: X = X.iloc[:, :model.n_features_in_]
                        except: pass

                    if scaler:
                        try: X = scaler.transform(X)
                        except: pass

                    ml_sig = np.zeros(len(df))
                    classes = getattr(model, "classes_", [0, 1])
                    
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X)
                        if len(classes) == 2:
                            ml_sig = np.where(probs[:, 1] > config.mlThreshold, 1, 
                                     np.where(probs[:, 0] > config.mlThreshold, -1, 0))
                        elif len(classes) >= 3:
                            ml_sig = np.where(probs[:, 2] > config.mlThreshold, 1, 
                                     np.where(probs[:, 0] > config.mlThreshold, -1, 0))
                    elif hasattr(model, "predict"):
                        raw = model.predict(X)
                        if raw.max() == 1: ml_sig = np.where(raw == 1, 1, -1)
                        else: ml_sig = np.where(raw == 2, 1, np.where(raw == 0, -1, 0))
                    
                    if config.mlMode == 'on': df['comb'] = ml_sig
                    elif config.mlMode == 'predictions':
                         df['comb'] = np.where((df['comb']==1)&(ml_sig==1), 1, 
                                      np.where((df['comb']==-1)&(ml_sig==-1), -1, 0))
            except Exception as e:
                logger.error(f"ML Error: {e}")
                pass 

        res = run_backtest(df, 'comb', config.initialBalance, config.fee)
        
        # FIX: Return DEEPCOPY
        clean_res = convert_numpy_types(res)
        response_payload = deepcopy(clean_res)
        response_payload["combinedResult"] = deepcopy(clean_res) 
        
        return JSONResponse(content=response_payload)
        
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))

@app.get("/api/ml/available-models")
def list_models():
    return [{"id": f.split('.')[0], "name": f.split('.')[0]} for f in os.listdir(MODEL_DIR) if f.endswith('.joblib')]

@app.post('/api/ml/certify-strategy')
async def handle_certify(config: CertifyConfig):
    return await handle_run_combo_backtest(BacktestConfig(
        symbol=config.symbol, timeframe=config.timeframe, 
        startDate=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        endDate=datetime.now().strftime("%Y-%m-%d"),
        strategies=[{'code': c, 'params': config.best_params} for c in config.best_params.get('combo_strategies', '').split(',')],
        params=config.best_params
    ))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
