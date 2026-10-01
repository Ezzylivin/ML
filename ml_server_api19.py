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
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas_ta as ta
import warnings
from copy import deepcopy

# 🚀 UPGRADE: Import Decimal for precision money handling
from decimal import Decimal, getcontext, ROUND_HALF_UP
getcontext().prec = 28  # High precision for internal calculations

# ----------------------------------------------------------------------
# 1. SETUP & CONSTANTS
# ----------------------------------------------------------------------
try:
    from config.constants import (
        SEED, RISK_FREE_RATE, MAX_LEVERAGE,
        SLIPPAGE_PCT, OPTIMIZER_CACHE_DIR,
        MODEL_DIR, DATA_DIR, EPSILON, PROJECT_ROOT
    )
    logging.getLogger().warning("Loaded config.constants successfully.")
except Exception as e:
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.getLogger().warning(f"Could not import config.constants.py, using fallback values: {e}")
    SEED = 12345
    RISK_FREE_RATE = 0.02
    MAX_LEVERAGE = 10
    SLIPPAGE_PCT = 0.001
    OPTIMIZER_CACHE_DIR = os.path.join(os.getcwd(), "data", "optimizer_cache")
    MODEL_DIR = os.path.join(os.getcwd(), "models")
    DATA_DIR = os.path.join(os.getcwd(), "data")
    EPSILON = 1e-9
    PROJECT_ROOT = os.getcwd()

os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None
logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
os.makedirs(OPTIMIZER_CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

PRICE_PRECISION = 8 
SIZE_PRECISION = 8   
CURRENCY_PRECISION = 4

# ----------------------------------------------------------------------
# 2. APP DEFINITION
# ----------------------------------------------------------------------
app = FastAPI(
    title="Trading ML Server API",
    description="API for running backtests, certification, and accessing ML models",
    version="2.2.6" 
)

# ----------------------------------------------------------------------
# 3. PYDANTIC MODELS
# ----------------------------------------------------------------------
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
    fee: float = 0.001
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

# ----------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ----------------------------------------------------------------------

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
    if not found_cols:
         raise KeyError(f"API: Could not find required col for key: {key} in columns: {df.columns.tolist()}")
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
    try:
        if df.index.tz is None: df.index = df.index.tz_localize('UTC')
        else: df.index = df.index.tz_convert('UTC')
    except Exception as e:
        logger.warning(f"Could not normalize timezone: {e}. Assuming UTC.")
        pass
    df_sliced = df.loc[buffer_start_dt:(end_dt + pd.Timedelta(days=1))].copy()
    if df_sliced.empty:
        if str(end_date).startswith('2099'): df_sliced = df.copy()
        if df_sliced.empty: raise ValueError(f"No data in {data_path} for date range {start_date} to {end_date}.")
    df_sliced.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True, errors='ignore')
    return df_sliced

def convert_numpy_types(obj):
    if isinstance(obj, (np.bool_, bool)): return bool(obj)
    if isinstance(obj, (np.floating, np.float64)):
        if np.isnan(obj) or np.isinf(obj): return None 
        return float(obj)
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, np.ndarray): return convert_numpy_types(obj.tolist()) 
    if isinstance(obj, (pd.Timestamp, datetime)): return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        try: return convert_numpy_types(obj.to_dict('records')) 
        except Exception: return []
    if isinstance(obj, pd.Series):
        try: return convert_numpy_types(obj.tolist()) 
        except Exception: return []
    if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()} 
    if isinstance(obj, list): return [convert_numpy_types(v) for v in obj] 
    return obj

def engineer_features_for_backtest(df: pd.DataFrame, trend_filter_period: typing.Optional[int], params: dict = None) -> pd.DataFrame:
    df = df.copy()
    if df['close'].isnull().any(): df['close'] = df['close'].fillna(method='ffill')
    df.replace([np.inf, -np.inf], np.nan, inplace=True) 
    if params is None: params = {} 

    # === 1. STATIC ML FEATURES ===
    try:
        df.ta.rsi(length=14, append=True, col_names=('RSI_14'))
        df.ta.macd(fast=12, slow=26, signal=9, append=True, col_names=('MACD_12_26_9', 'MACDH_12_26_9', 'MACDS_12_26_9'))
        df.ta.stoch(k=14, d=3, smooth_k=3, append=True, col_names=('STOCHk_14_3_3', 'STOCHd_14_3_3', 'STOCHs_14_3_3'))
        df.ta.cci(length=20, append=True, col_names=('CCI_20_0.015'))
        df.ta.bbands(length=20, std=2.0, append=True, col_names=('BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0'))
        df.ta.atr(length=14, append=True, col_names=('ATR_14'))
        df.ta.sma(length=10, append=True, col_names=('SMA_10'))
        df.ta.sma(length=50, append=True, col_names=('SMA_50'))
        df.ta.sma(length=200, append=True, col_names=('SMA_200'))
        df.ta.psar(append=True)
        df.ta.ichimoku(conversion=9, base=26, span=52, append=True)
        df.ta.obv(append=True)
        df['OBV_SMA_20'] = df['OBV'].rolling(window=20).mean()
        df.ta.adx(length=14, append=True)
    except Exception as e:
        logger.warning(f"Error generating static ML features: {e}")
        pass 

    # === 2. DYNAMIC STRATEGY FEATURES ===
    rsi_len_dyn = params.get('rsi_length', 14)
    macd_fast_dyn = params.get('macd_fast_period', 12)
    macd_slow_dyn = params.get('macd_slow_period', 26)
    macd_sig_dyn = params.get('macd_signal_period', 9)
    stoch_k_dyn = params.get('k_period', 14)
    stoch_d_dyn = params.get('d_period', 3)
    cci_len_dyn = params.get('cci_length', 20)
    bb_len_dyn = params.get('bb_length', 20)
    bb_std_dyn = params.get('bb_std', 2.0)
    sma_fast_dyn = params.get('sma_fast_period', 10)
    sma_slow_dyn = params.get('sma_slow_period', 50)

    if rsi_len_dyn != 14: df.ta.rsi(length=rsi_len_dyn, append=True, col_names=(f'RSI_{rsi_len_dyn}'))
    if macd_fast_dyn != 12 or macd_slow_dyn != 26 or macd_sig_dyn != 9:
        df.ta.macd(fast=macd_fast_dyn, slow=macd_slow_dyn, signal=macd_sig_dyn, append=True, col_names=(f'MACD_{macd_fast_dyn}_{macd_slow_dyn}_{macd_sig_dyn}', f'MACDH_{macd_fast_dyn}_{macd_slow_dyn}_{macd_sig_dyn}', f'MACDS_{macd_fast_dyn}_{macd_slow_dyn}_{macd_sig_dyn}'))
    if stoch_k_dyn != 14 or stoch_d_dyn != 3:
        df.ta.stoch(k=stoch_k_dyn, d=stoch_d_dyn, smooth_k=3, append=True, col_names=(f'STOCHk_{stoch_k_dyn}_{stoch_d_dyn}_3', f'STOCHd_{stoch_k_dyn}_{stoch_d_dyn}_3', f'STOCHs_{stoch_k_dyn}_{stoch_d_dyn}_3'))
    if cci_len_dyn != 20: df.ta.cci(length=cci_len_dyn, append=True, col_names=(f'CCI_{cci_len_dyn}_0.015'))
    if bb_len_dyn != 20 or bb_std_dyn != 2.0:
        df.ta.bbands(length=bb_len_dyn, std=bb_std_dyn, append=True, col_names=(f'BBL_{bb_len_dyn}_{bb_std_dyn}', f'BBM_{bb_len_dyn}_{bb_std_dyn}', f'BBU_{bb_len_dyn}_{bb_std_dyn}', f'BBB_{bb_len_dyn}_{bb_std_dyn}', f'BBP_{bb_len_dyn}_{bb_std_dyn}'))
    if sma_fast_dyn != 10 and sma_fast_dyn != 50 and sma_fast_dyn != 200:
        df.ta.sma(length=sma_fast_dyn, append=True, col_names=(f'SMA_{sma_fast_dyn}'))
    if sma_slow_dyn != 10 and sma_slow_dyn != 50 and sma_slow_dyn != 200:
        df.ta.sma(length=sma_slow_dyn, append=True, col_names=(f'SMA_{sma_slow_dyn}'))
        
    if trend_filter_period and trend_filter_period > 0:
        sma_trend_col = f'SMA_{trend_filter_period}'
        if sma_trend_col not in df.columns: df.ta.sma(length=trend_filter_period, append=True, col_names=(sma_trend_col))

    # === 3. Get Column Names ===
    try:
        rsi_col = find_col(df, f'RSI_{rsi_len_dyn}')
        macd_col = find_col(df, f'MACD_{macd_fast_dyn}_{macd_slow_dyn}_{macd_sig_dyn}', 'MACDS')
        macd_signal_col = find_col(df, f'MACDS_{macd_fast_dyn}_{macd_slow_dyn}_{macd_sig_dyn}')
        stoch_k_col = find_col(df, f'STOCHk_{stoch_k_dyn}_{stoch_d_dyn}_3') 
        stoch_d_col = find_col(df, f'STOCHd_{stoch_k_dyn}_{stoch_d_dyn}_3') 
        cci_col = find_col(df, f'CCI_{cci_len_dyn}_0.015')
        bbl_col = find_col(df, f'BBL_{bb_len_dyn}_{bb_std_dyn}')
        bbu_col = find_col(df, f'BBU_{bb_len_dyn}_{bb_std_dyn}')
        bbm_col = find_col(df, f'BBM_{bb_len_dyn}_{bb_std_dyn}')
        sma10_col = find_col(df, f'SMA_{sma_fast_dyn}') 
        sma50_col = find_col(df, f'SMA_{sma_slow_dyn}')
        
        atr_col = find_col(df, 'ATR_14')
        sma200_col = find_col(df, 'SMA_200')
        psar_col = find_col(df, 'PSARr')
        adx_col = find_col(df, 'ADX_14')
        obv_col = find_col(df, 'OBV')
        spanA_col = find_col(df, 'ISA_9')  
        spanB_col = find_col(df, 'ISB_26') 
    except KeyError as e:
        logger.error(f"Failed to find a base TA column: {e}. Stopping.")
        raise e

    # === 4. Features ===
    df['price_vs_sma50'] = (df['close'] - df[sma50_col]) / (df[sma50_col] + EPSILON) 
    df['price_vs_sma200'] = (df['close'] - df[sma200_col]) / (df[sma200_col] + EPSILON)
    df['sma10_vs_sma50'] = (df[sma10_col] - df[sma50_col]) / (df[sma50_col] + EPSILON) 
    df['sma50_vs_sma200'] = (df[sma50_col] - df[sma200_col]) / (df[sma200_col] + EPSILON)
    df['price_vs_bbu'] = (df[bbu_col] - df['close']) / (df['close'] + EPSILON)
    df['price_vs_bbl'] = (df[bbl_col] - df['close']) / (df['close'] + EPSILON)
    df['bb_width'] = (df[bbu_col] - df[bbl_col]) / (df[bbm_col] + EPSILON)
    df['rsi_state'] = np.select([df[rsi_col] > 70, df[rsi_col] < 30], [1, -1], 0) 
    df['macd_hist_norm'] = (df[macd_col] - df[macd_signal_col]) / (df['close'] + EPSILON)
    df['stoch_cross'] = np.sign(df[stoch_k_col] - df[stoch_d_col]).diff()
    df['cci_state'] = np.select([df[cci_col] > 100, df[cci_col] < -100], [1, -1], 0) 
    df['atr_pct'] = (df[atr_col] / (df['close'] + EPSILON)) * 100
    df['adx_strong_trend'] = (df[adx_col] > 25).astype(int)
    df['rsi_roc_3'] = (df[rsi_col] - df[rsi_col].shift(3)) / (df[rsi_col].shift(3) + EPSILON)
    df['volume_roc_10'] = (df['volume'] - df['volume'].shift(10)) / (df['volume'].shift(10) + EPSILON)
    df['price_roc_5'] = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + EPSILON)

    lag_features_to_create = [
        'RSI_14', 'ADX_14', 'OBV', 'ATR_14', 'price_vs_sma200', 'sma50_vs_sma200',
        f'RSI_{rsi_len_dyn}', 'rsi_state', 'atr_pct', 'macd_hist_norm', 
        'volume_roc_10', 'price_roc_5', 'bb_width', 
        'adx_strong_trend', 'stoch_cross', 'cci_state', 'rsi_roc_3'
    ]
    final_lag_list = []
    for feat_base in lag_features_to_create:
        try:
            col_name = find_col(df, feat_base) 
            if col_name not in final_lag_list: final_lag_list.append(col_name)
        except KeyError: logger.warning(f"Could not find base feature '{feat_base}' for lagging.")
            
    for feat_col in final_lag_list:
        if feat_col in df.columns:
            for lag in [1, 2, 3]: df[f"{feat_col}_lag{lag}"] = df[feat_col].shift(lag)

    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    return df

def generate_ta_signals(df: pd.DataFrame, strategy_code: str, params: dict = None) -> pd.DataFrame:
    df = df.copy()
    df['ta_signal'] = 0
    if params is None: params = {} 

    try:
        rsi_len = params.get('rsi_length', 14)
        stoch_k = params.get('k_period', 14)
        stoch_d = params.get('d_period', 3)
        cci_len = params.get('cci_length', 20)
        bb_len = params.get('bb_length', 20)
        bb_std = params.get('bb_std', 2.0)
        macd_fast = params.get('macd_fast_period', 12)
        macd_slow = params.get('macd_slow_period', 26)
        macd_sig = params.get('macd_signal_period', 9)
        sma_fast = params.get('sma_fast_period', 10)
        sma_slow = params.get('sma_slow_period', 50)
        
        rsi_col = find_col(df, f'RSI_{rsi_len}')
        macd_col = find_col(df, f'MACD_{macd_fast}_{macd_slow}_{macd_sig}', 'MACDS')
        macd_signal_col = find_col(df, f'MACDS_{macd_fast}_{macd_slow}_{macd_sig}')
        stoch_k_col = find_col(df, f'STOCHk_{stoch_k}_{stoch_d}_3') 
        stoch_d_col = find_col(df, f'STOCHd_{stoch_k}_{stoch_d}_3') 
        cci_col = find_col(df, f'CCI_{cci_len}_0.015')
        bbl_col = find_col(df, f'BBL_{bb_len}_{bb_std}')
        bbu_col = find_col(df, f'BBU_{bb_len}_{bb_std}')
        atr_col = find_col(df, 'ATR_14')
        sma10_col = find_col(df, f'SMA_{sma_fast}') 
        sma50_col = find_col(df, f'SMA_{sma_slow}') 
        psar_col = find_col(df, 'PSARr')
        spanA_col = find_col(df, 'ISA_9')
        spanB_col = find_col(df, 'ISB_26')
        obv_col = find_col(df, 'OBV')

        df['close_prev'] = df['close'].shift(1)
        df['rsi_prev'] = df[rsi_col].shift(1)
        df['cci_prev'] = df[cci_col].shift(1)
        df['sma10_prev'] = df[sma10_col].shift(1) 
        df['sma50_prev'] = df[sma50_col].shift(1) 
        df['macd_hist'] = df[macd_col] - df[macd_signal_col]
        df['macd_hist_prev'] = df['macd_hist'].shift(1)
        df['stoch_k_prev'] = df[stoch_k_col].shift(1)
        df['stoch_d_prev'] = df[stoch_d_col].shift(1)
        df['atr_prev'] = df[atr_col].shift(1)
        df['obv_prev'] = df[obv_col].shift(1)

        if strategy_code == "sma_crossover":
            df.loc[(df['sma10_prev'] <= df['sma50_prev']) & (df[sma10_col] > df[sma50_col]), 'ta_signal'] = 1
            df.loc[(df['sma10_prev'] >= df['sma50_prev']) & (df[sma10_col] < df[sma50_col]), 'ta_signal'] = -1
        elif strategy_code == "rsi_divergence":
            oversold = params.get('oversold_level', 30)
            overbought = params.get('overbought_level', 70)
            df.loc[(df['rsi_prev'] >= oversold) & (df[rsi_col] < oversold), 'ta_signal'] = 1
            df.loc[(df['rsi_prev'] <= overbought) & (df[rsi_col] > overbought), 'ta_signal'] = -1
        elif strategy_code == "macd_crossover":
            df.loc[(df['macd_hist_prev'] <= 0) & (df['macd_hist'] > 0), 'ta_signal'] = 1
            df.loc[(df['macd_hist_prev'] >= 0) & (df['macd_hist'] < 0), 'ta_signal'] = -1
        elif strategy_code == "stochastic_crossover":
            df.loc[(df['stoch_k_prev'] <= df['stoch_d_prev']) & (df[stoch_k_col] > df[stoch_d_col]), 'ta_signal'] = 1
            df.loc[(df['stoch_k_prev'] >= df['stoch_d_prev']) & (df[stoch_k_col] < df[stoch_d_col]), 'ta_signal'] = -1
        elif strategy_code == "cci_oversold":
            df.loc[(df['cci_prev'] >= -100) & (df[cci_col] < -100), 'ta_signal'] = 1
            df.loc[(df['cci_prev'] <= 100) & (df[cci_col] > 100), 'ta_signal'] = -1
        elif strategy_code == "bollinger_bands":
            df.loc[(df['close_prev'] >= df[bbl_col].shift(1)) & (df['close'] < df[bbl_col]), 'ta_signal'] = 1
            df.loc[(df['close_prev'] <= df[bbu_col].shift(1)) & (df['close'] > df[bbu_col]), 'ta_signal'] = -1
        elif strategy_code == "ichimoku_cloud":
            df.loc[((df['close_prev'] <= df[spanA_col].shift(1)) | (df['close_prev'] <= df[spanB_col].shift(1))) & ((df['close'] > df[spanA_col]) & (df['close'] > df[spanB_col])), 'ta_signal'] = 1
            df.loc[((df['close_prev'] >= df[spanA_col].shift(1)) | (df['close_prev'] >= df[spanB_col].shift(1))) & ((df['close'] < df[spanA_col]) & (df['close'] < df[spanB_col])), 'ta_signal'] = -1
        elif strategy_code == "atr_signal":
            df.loc[(df['atr_prev'] >= df[atr_col]), 'ta_signal'] = -1
            df.loc[(df['atr_prev'] < df[atr_col]), 'ta_signal'] = 1
        elif strategy_code == "obv_signal":
            df.loc[(df['obv_prev'] >= df[obv_col]), 'ta_signal'] = -1
            df.loc[(df['obv_prev'] < df[obv_col]), 'ta_signal'] = 1
        elif strategy_code == "psar_signal":
            df['ta_signal'] = df[psar_col].fillna(0)
        else:
            logger.warning(f"Strategy code '{strategy_code}' not found. Defaulting to 'Hold' (0).")
            df['ta_signal'] = 0
    except KeyError as e:
        logger.error(f"KeyError in generate_ta_signals: {e}. One of your TA columns is missing. Defaulting to 'Hold' (0).")
        df['ta_signal'] = 0
    except Exception as e:
        logger.error(f"Error in generate_ta_signals: {e}. Defaulting to 'Hold' (0).")
        df['ta_signal'] = 0
    return df

# ----------------------------------------------------------------------
# 🛡️ SAFE BACKTEST ENGINE (Float + Logic Guards)
# ----------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    signal_column: str,
    initial_balance: float,
    fee: float,
    stop_loss_pct: typing.Optional[float],
    take_profit_pct: typing.Optional[float],
    risk_mode: str,
    risk_percent: float,
    growth_target: typing.Optional[float],
    min_atr_pct: typing.Optional[float],
    trend_filter_period: typing.Optional[int],
    ml_mode: str,
    min_adx_level: typing.Optional[float],
    trailing_stop_atr_mult: typing.Optional[float],
    trailing_stop_pct: typing.Optional[float],
    max_leverage: float = MAX_LEVERAGE,
    **kwargs
) -> typing.Dict[str, typing.Any]:
    
    if signal_column not in df.columns: raise ValueError(f"Signal column '{signal_column}' not found.")

    # Helper safely get columns
    def get_col_safe(name):
        return find_col(df, name) if name in df.columns or any(x in name for x in ['ATR', 'ADX', 'SMA']) else None

    # 💡 FIX: Standardized variable names
    atr_col = get_col_safe('ATR_14')
    adx_col = get_col_safe('ADX_14')
    trend_col = get_col_safe(f'SMA_{trend_filter_period}') if trend_filter_period else None

    # 🚀 UPGRADE: Use Decimal for Money
    DEC_EPSILON = Decimal(str(EPSILON))
    balance = Decimal(str(initial_balance))
    equity = Decimal(str(initial_balance))
    
    # Fee and Slippage as Decimals
    fee_dec = Decimal(str(fee))
    slippage_pct_dec = Decimal(str(SLIPPAGE_PCT))
    
    position = 0
    position_size = Decimal('0.0')
    entry_price = Decimal('0.0')
    entry_time = None
    equity_curve = []
    trades = []
    
    stop_loss_price = Decimal('0.0')
    take_profit_price = Decimal('0.0')
    current_risk_percent = Decimal(str(risk_percent))
    
    df_signals = df[signal_column].to_numpy()
    df_low = df["low"].to_numpy()
    df_high = df["high"].to_numpy()
    df_close = df["close"].to_numpy()
    df_open = df["open"].to_numpy()
    
    # Helper to get ATR/ADX if available
    def get_val(arr, idx):
        return Decimal(str(arr[idx])) if arr is not None else Decimal('0')

    # 💡 FIX: Use standardized variable names
    df_atr = df[atr_col].to_numpy(dtype=float) if atr_col else None
    df_adx = df[adx_col].to_numpy(dtype=float) if adx_col else None
    df_trend = df[trend_col].to_numpy(dtype=float) if trend_col else None

    for i in range(1, len(df) - 1):  
        current_low = Decimal(str(df_low[i]))
        current_high = Decimal(str(df_high[i]))
        current_close = Decimal(str(df_close[i]))
        current_atr = get_val(df_atr, i)
        current_adx = get_val(df_adx, i)
        next_open = Decimal(str(df_open[i+1]))
        current_timestamp = df.index[i]

        if position != 0:
            exit_price = Decimal('0.0')
            pnl_reason = "Signal"
            
            # Trailing Stop Update
            if position == 1:
                if trailing_stop_atr_mult and trailing_stop_atr_mult > 0:
                    new_stop = current_close - (current_atr * Decimal(str(trailing_stop_atr_mult)))
                    stop_loss_price = max(stop_loss_price, new_stop)
                elif trailing_stop_pct and trailing_stop_pct > 0: 
                    new_stop = current_close * (Decimal('1') - Decimal(str(trailing_stop_pct)) / Decimal('100'))
                    stop_loss_price = max(stop_loss_price, new_stop)
            elif position == -1:
                if trailing_stop_atr_mult and trailing_stop_atr_mult > 0:
                    new_stop = current_close + (current_atr * Decimal(str(trailing_stop_atr_mult)))
                    if stop_loss_price == 0: stop_loss_price = new_stop
                    else: stop_loss_price = min(stop_loss_price, new_stop)
                elif trailing_stop_pct and trailing_stop_pct > 0:
                    new_stop = current_close * (Decimal('1') + Decimal(str(trailing_stop_pct)) / Decimal('100'))
                    if stop_loss_price == 0: stop_loss_price = new_stop
                    else: stop_loss_price = min(stop_loss_price, new_stop)

            # Check Exit
            if position == 1:
                sl_hit = current_low <= stop_loss_price
                tp_hit = take_profit_pct is not None and current_high >= take_profit_price
                signal_exit = (df_signals[i] == -1)
                if sl_hit: exit_price, pnl_reason = stop_loss_price, "Stop Loss"
                elif tp_hit: exit_price, pnl_reason = take_profit_price, "Take Profit"
                elif signal_exit: exit_price, pnl_reason = next_open * (Decimal('1') - slippage_pct_dec), "Signal"
            elif position == -1:
                sl_hit = current_high >= stop_loss_price
                tp_hit = take_profit_pct is not None and current_low <= take_profit_price
                signal_exit = (df_signals[i] == 1)
                if sl_hit: exit_price, pnl_reason = stop_loss_price, "Stop Loss"
                elif tp_hit: exit_price, pnl_reason = take_profit_price, "Take Profit"
                elif signal_exit: exit_price, pnl_reason = next_open * (Decimal('1') + slippage_pct_dec), "Signal"

            if exit_price > 0:
                if position == 1: 
                    profit_usd = (exit_price * (Decimal('1') - fee_dec) - entry_price * (Decimal('1') + fee_dec)) * position_size
                else: 
                    profit_usd = (entry_price * (Decimal('1') - fee_dec) - exit_price * (Decimal('1') + fee_dec)) * position_size
                
                balance += profit_usd
                
                # 💡 FIX: Use DEC_EPSILON instead of global EPSILON (float)
                pnl_pct = (profit_usd / ((position_size * entry_price) + DEC_EPSILON)) * 100

                trades.append({
                    "action": "sell" if position == 1 else "cover",
                    "price": float(exit_price),
                    "time": current_timestamp,
                    "size": float(position_size),
                    "pnl_pct": float(pnl_pct),
                    "profit_usd": float(profit_usd),
                    "reason": pnl_reason,
                    "entryPrice": float(entry_price), "entryTime": entry_time, "position": "long" if position == 1 else "short",
                    "result": "win" if profit_usd > 0 else "loss" if profit_usd < 0 else "breakeven"
                })
                position = 0
                position_size = Decimal('0.0')

        if position == 0 and balance > 0:
            signal = df_signals[i]
            passes_vol_filter = True
            if min_atr_pct is not None and min_atr_pct > 0:
                if (current_atr / (current_close + DEC_EPSILON)) * 100 < min_atr_pct: passes_vol_filter = False

            passes_trend_filter = True
            if trend_col and trend_col in df.columns: # Use correct variable
                trend_val = Decimal(str(df_trend[i])) # Use correct array
                if (signal == 1 and current_close < trend_val) or (signal == -1 and current_close > trend_val): passes_trend_filter = False

            passes_adx_filter = True
            if min_adx_level is not None and min_adx_level > 0:
                if current_adx < min_adx_level: passes_adx_filter = False

            if (signal == 1 or signal == -1) and passes_vol_filter and passes_trend_filter and passes_adx_filter:
                if next_open <= 0: continue

                risk_amount_usd = balance * (current_risk_percent / Decimal('100'))
                sl_distance_usd = Decimal('0.0')
                
                if trailing_stop_atr_mult and trailing_stop_atr_mult > 0: 
                    sl_distance_usd = current_atr * Decimal(str(trailing_stop_atr_mult))
                elif trailing_stop_pct and trailing_stop_pct > 0: 
                    sl_distance_usd = next_open * (Decimal(str(trailing_stop_pct)) / Decimal('100'))
                elif stop_loss_pct and stop_loss_pct > 0: 
                    sl_distance_usd = next_open * (Decimal(str(stop_loss_pct)) / Decimal('100'))
                
                if sl_distance_usd <= 0:
                    sl_distance_usd = next_open * Decimal('0.01') # Safety fallback

                position_size = risk_amount_usd / sl_distance_usd

                # Max Leverage Cap
                max_lev = Decimal(str(max_leverage))
                max_size_allowed = (balance * max_lev) / next_open
                
                if position_size > max_size_allowed:
                    position_size = max_size_allowed
                
                if position_size <= Decimal('0.00000001'): continue

                entry_time = df.index[i+1]
                
                if signal == 1:
                    position = 1
                    entry_price = next_open * (Decimal('1') + slippage_pct_dec)
                    if trailing_stop_atr_mult and trailing_stop_atr_mult > 0: 
                        stop_loss_price = entry_price - (current_atr * Decimal(str(trailing_stop_atr_mult)))
                    elif trailing_stop_pct and trailing_stop_pct > 0: 
                        stop_loss_price = entry_price * (Decimal('1') - Decimal(str(trailing_stop_pct)) / Decimal('100'))
                    elif stop_loss_pct: 
                        stop_loss_price = entry_price * (Decimal('1') - Decimal(str(stop_loss_pct)) / Decimal('100'))
                    if take_profit_pct: 
                        take_profit_price = entry_price * (Decimal('1') + Decimal(str(take_profit_pct)) / Decimal('100'))
                    
                    # 💡 FIX: Added 'position' to Entry Record
                    trades.append({
                        "action": "buy", 
                        "price": float(entry_price), 
                        "time": entry_time, 
                        "size": float(position_size),
                        "position": "long" 
                    })

                elif signal == -1:
                    position = -1
                    entry_price = next_open * (Decimal('1') - slippage_pct_dec)
                    if trailing_stop_atr_mult and trailing_stop_atr_mult > 0: 
                        stop_loss_price = entry_price + (current_atr * Decimal(str(trailing_stop_atr_mult)))
                    elif trailing_stop_pct and trailing_stop_pct > 0: 
                        stop_loss_price = entry_price * (Decimal('1') + Decimal(str(trailing_stop_pct)) / Decimal('100'))
                    elif stop_loss_pct: 
                        stop_loss_price = entry_price * (Decimal('1') + Decimal(str(stop_loss_pct)) / Decimal('100'))
                    if take_profit_pct: 
                        take_profit_price = entry_price * (Decimal('1') - Decimal(str(take_profit_pct)) / Decimal('100'))
                    
                    # 💡 FIX: Added 'position' to Entry Record
                    trades.append({
                        "action": "sell_short", 
                        "price": float(entry_price), 
                        "time": entry_time, 
                        "size": float(position_size),
                        "position": "short" 
                    })

        equity = balance
        if position == 1: 
            equity += (current_close - entry_price * (Decimal('1') + fee_dec)) * position_size
        elif position == -1: 
            equity += (entry_price * (Decimal('1') - fee_dec) - current_close) * position_size
        
        equity_curve.append({"timestamp": df.index[i], "balance": float(equity)})

    # Metrics Calculation
    final_balance = float(equity)
    initial_balance_flt = float(initial_balance)
    total_return = ((final_balance / initial_balance_flt) - 1) * 100
    
    wins = [t for t in trades if t.get('result') == 'win']
    losses = [t for t in trades if t.get('result') == 'loss']
    
    # 💡 FIX: Use only closed trades for win rate denominator
    total_closed_trades = len(wins) + len(losses)
    win_rate = (len(wins) / total_closed_trades) * 100 if total_closed_trades > 0 else 0
    
    total_profit = sum(t['profit_usd'] for t in wins)
    total_loss = abs(sum(t['profit_usd'] for t in losses))
    profit_factor = total_profit / total_loss if total_loss > 0 else 999

    # Drawdown
    eq_series = pd.Series([e['balance'] for e in equity_curve])
    peak = eq_series.cummax()
    dd = (eq_series - peak) / (peak + 1e-9)
    max_dd = abs(dd.min()) * 100 if not dd.empty else 0
    
    sharpe_ratio = 0.0
    sortino_ratio = 0.0
    calmar_ratio = 0.0

    if len(equity_curve) > 1:
        daily_returns = eq_series.pct_change().fillna(0)
        days = max((df.index[-1] - df.index[0]).days, 1)
        annual_return_rate = ((final_balance / (initial_balance_flt + 1e-9)) ** (365.25 / days)) - 1
        annual_std_dev = daily_returns.std() * np.sqrt(365.25)
        sharpe_ratio = (annual_return_rate - RISK_FREE_RATE) / (annual_std_dev + 1e-9)
        downside_returns = daily_returns[daily_returns < 0]
        annual_downside_std = downside_returns.std() * np.sqrt(365.25) if not downside_returns.empty else 0
        sortino_ratio = (annual_return_rate - RISK_FREE_RATE) / (annual_downside_std + 1e-9)
        calmar_ratio = (annual_return_rate * 100) / (max_dd + 1e-9)

    return {
        "metrics": {
            "totalReturn": total_return,
            "profitFactor": profit_factor,
            "maxDrawdown": max_dd,
            "winRate": win_rate,
            "totalTrades": total_closed_trades, # Report only closed
            "finalBalance": final_balance,
            "averageWin": total_profit / len(wins) if wins else 0,
            "averageLoss": total_loss / len(losses) if losses else 0,
            "winningTrades": len(wins),
            "losingTrades": len(losses),
            "sharpeRatio": sharpe_ratio, 
            "sortinoRatio": sortino_ratio,
            "calmarRatio": calmar_ratio
        },
        "equityCurve": equity_curve,
        "tradeBreakdown": trades,
        "candleData": df.reset_index().rename(columns={'datetime':'timestamp'})[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')
    }

# ----------------------------------------------------------------------
# 5. CERTIFICATION LOGIC
# ----------------------------------------------------------------------

def run_certification(best_params: dict, full_df: pd.DataFrame, base_config: dict) -> dict:
    logger.warning("[Certifier] Starting certification...")
    
    if full_df.empty: return {"certification_passed": False, "reason": "DataFrame is empty."}
    last_date = full_df.index[-1]
    start_date_dt = last_date - pd.Timedelta(days=180)
    CERTIFICATION_START_DATE = start_date_dt.strftime("%Y-%m-%d")
    CERTIFICATION_END_DATE = last_date.strftime("%Y-%m-%d")
    
    holdout_config = deepcopy(base_config)
    holdout_config['startDate'] = CERTIFICATION_START_DATE
    holdout_config['endDate'] = CERTIFICATION_END_DATE
    holdout_config['optimizer_mode'] = False
    holdout_config['params'] = {}
    
    for key, value in best_params.items():
        if value is None or (isinstance(value, float) and np.isnan(value)): continue
        if key in ["CalmarRatio", "MaxDrawdown"]: continue
        if key.startswith("params."):
            param_key = key.split('.', 1)[1] 
            set_nested_value(holdout_config['params'], param_key, value)
        else:
            set_nested_value(holdout_config, key, value)
    
    if "combo_strategies" in best_params and isinstance(best_params["combo_strategies"], str):
        combo_codes = [c.strip() for c in best_params["combo_strategies"].split(',')]
        holdout_config['strategies'] = [{"code": code, "params": {}} for code in combo_codes]
        
    try:
        cert_params = holdout_config.get('params', {})
        trend_period = cert_params.get('trendFilterPeriod')
        df_holdout_features = engineer_features_for_backtest(full_df, trend_period, cert_params)
        signal_column = 'final_signal'
        
        if holdout_config.get('strategies'):
            hybrid_mode = cert_params.get('hybridMode', 'AND')
            signal_columns = []
            for i, strat in enumerate(holdout_config['strategies']):
                ta_code = strat.get('code')
                if ta_code:
                    sig_col = f'ta_signal_{i}'
                    df_holdout_features = generate_ta_signals(df_holdout_features, ta_code, cert_params)
                    df_holdout_features[sig_col] = df_holdout_features.get('ta_signal', 0)
                    signal_columns.append(sig_col)
            
            if signal_columns:
                if hybrid_mode == 'REGIME' and len(signal_columns) >= 2:
                    adx_threshold = cert_params.get('regime_threshold', 25)
                    try: adx_col = find_col(df_holdout_features, 'ADX')
                    except KeyError: adx_col = 'ADX_14'
                    df_holdout_features['combined_ta'] = np.where(
                        df_holdout_features[adx_col] > adx_threshold,
                        df_holdout_features[signal_columns[0]], 
                        df_holdout_features[signal_columns[1]] 
                    )
                elif hybrid_mode == 'AND':
                    df_holdout_features['combined_ta'] = df_holdout_features[signal_columns].apply(lambda r: 1 if (r == 1).all() else (-1 if (r == -1).all() else 0), axis=1)
                else: 
                    df_holdout_features['combined_ta'] = df_holdout_features[signal_columns].apply(lambda r: 1 if (r == 1).any() else (-1 if (r == -1).any() else 0), axis=1)
            else:
                df_holdout_features['combined_ta'] = 0
            
            df_holdout_features['high_conf_prediction'] = 0 
            if holdout_config.get('mlMode') == 'on': df_holdout_features[signal_column] = df_holdout_features['high_conf_prediction']
            elif holdout_config.get('mlMode') == 'predictions':
                 df_holdout_features.loc[(df_holdout_features['combined_ta'] == 1) & (df_holdout_features['high_conf_prediction'] == 1), signal_column] = 1
                 df_holdout_features.loc[(df_holdout_features['combined_ta'] == -1) & (df_holdout_features['high_conf_prediction'] == -1), signal_column] = -1
            else: df_holdout_features[signal_column] = df_holdout_features['combined_ta']
        else:
            ta_code = holdout_config.get('code')
            if ta_code:
                df_holdout_features = generate_ta_signals(df_holdout_features, ta_code, cert_params)
                df_holdout_features[signal_column] = df_holdout_features.get('ta_signal', 0).fillna(0)
            else:
                df_holdout_features['high_conf_prediction'] = 0 
                df_holdout_features[signal_column] = df_holdout_features['high_conf_prediction']
                
        df_sliced = df_holdout_features.loc[CERTIFICATION_START_DATE:CERTIFICATION_END_DATE].copy().fillna(0)
        if df_sliced.empty: return {"certification_passed": False, "reason": "No data in hold-out period."}
        
        bt_params = {
            "df": df_sliced, "signal_column": signal_column, "initial_balance": holdout_config.get('initialBalance', 1000), "fee": holdout_config.get('fee', 0.001), "stop_loss_pct": cert_params.get('SL'), "take_profit_pct": cert_params.get('TP'), "risk_mode": holdout_config.get('riskManagementMode', 'standard'), "risk_percent": holdout_config.get('riskPercentage', 1.0), "growth_target": holdout_config.get('growthCapitalTarget'), "min_atr_pct": cert_params.get('minAtrPct'), "trend_filter_period": cert_params.get('trendFilterPeriod'), "ml_mode": holdout_config.get('mlMode', 'off'), "min_adx_level": cert_params.get('minAdxLevel'), "trailing_stop_atr_mult": cert_params.get('tslAtrMult'), "trailing_stop_pct": cert_params.get('tslPct')
        }
            
        holdout_results = run_backtest(**bt_params)
        holdout_metrics = holdout_results.get('metrics', {})
        
        certification_passed = holdout_metrics.get('calmarRatio', -1) > 0 and holdout_metrics.get('totalTrades', 0) > 5
        
        return {
            "certification_passed": certification_passed, "holdout_metrics": holdout_metrics, "holdout_equity_curve": holdout_results.get('equityCurve'), "parameters_certified": best_params 
        }

    except Exception as e:
        logger.error(f"[Certifier] Certification run failed: {e}", exc_info=True)
        return {"certification_passed": False, "reason": str(e), "traceback": traceback.format_exc()}

# ----------------------------------------------------------------------
# 7. API ENDPOINTS
# ----------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ML Server API is running"}

@app.get("/api/ml/available-models", response_model=List[Dict[str, Any]])
def list_models():
    models = []
    try:
        if not os.path.exists(MODEL_DIR):
            logger.warning(f"MODEL_DIR does not exist: {MODEL_DIR}")
            return []
        for filename in os.listdir(MODEL_DIR):
            if filename.startswith('.'): continue
            lower = filename.lower()
            if lower.endswith('.joblib') or lower.endswith('.pkl') or lower.endswith('.model'):
                model_id = os.path.splitext(filename)[0]
                model_name = model_id.replace('_', ' ').title()
                models.append({"id": model_id, "name": model_name, "file": filename})
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")
    return models

@app.get("/api/ml/models", response_model=List[Dict[str, Any]])
def list_models_alias():
    return list_models()

@app.post('/api/ml/run-backtest-on')
async def handle_run_backtest_on(config: BacktestConfig):
    try:
        params = config.params or {}
        sl_pct = params.get('SL')
        tp_pct = params.get('TP')
        min_atr_pct = params.get('minAtrPct')
        trend_period = params.get('trendFilterPeriod')
        min_adx_level = params.get('minAdxLevel')
        trailing_stop_atr_mult = params.get('tslAtrMult')
        trailing_stop_pct = params.get('tslPct')

        df = load_efficient_data(config.symbol, config.timeframe, config.startDate, config.endDate)
        df = engineer_features_for_backtest(df, trend_period, params)
        signal_column = 'final_signal'

        if config.mlMode == 'off':
            if not config.code: raise ValueError("TA Strategy 'code' is required for 'off' mode.")
            df = generate_ta_signals(df, config.code, params)
            df[signal_column] = df.get('ta_signal', pd.Series(0, index=df.index)).fillna(0)

        else: 
            if not config.mlModel: raise ValueError(f"ML Model name is required for '{config.mlMode}' mode.")
            model_path = os.path.join(MODEL_DIR, f"{config.mlModel}.joblib")
            if not os.path.exists(model_path):
                model_path = os.path.join(MODEL_DIR, f"{config.mlModel}.pkl")
                if not os.path.exists(model_path): raise FileNotFoundError(f"Model file '{config.mlModel}.joblib' or '.pkl' not found in {MODEL_DIR}.")

            pipeline = joblib.load(model_path)
            class_indices = {'buy': 0, 'hold': 1, 'sell': 2} 

            if isinstance(pipeline, dict):
                model = pipeline.get('model')
                scaler = pipeline.get('scaler')
                model_feature_names = pipeline.get('feature_names', [])
                class_indices = pipeline.get('class_indices', class_indices)
                missing_cols = [col for col in model_feature_names if col not in df.columns]
                if missing_cols: raise KeyError(f"Missing feature columns for ML model: {missing_cols}")
                df_features = df[model_feature_names].copy().fillna(0)
                X_scaled_array = scaler.transform(df_features)
                if hasattr(model, 'feature_names_in_') or 'LGBM' in str(type(model)): X_scaled_input = pd.DataFrame(X_scaled_array, columns=model_feature_names, index=df_features.index)
                else: X_scaled_input = X_scaled_array
                probabilities = model.predict_proba(X_scaled_input)

            else:
                probabilities = pipeline.predict_proba(df.fillna(0))
            
            df['prob_buy'] = probabilities[:, class_indices.get('buy', 0)]
            df['prob_sell'] = probabilities[:, class_indices.get('sell', 2)]
            
            df['high_conf_prediction'] = 0
            df.loc[(df['prob_buy'] > config.mlThreshold), 'high_conf_prediction'] = 1
            df.loc[(df['prob_sell'] > config.mlThreshold), 'high_conf_prediction'] = -1

            if config.mlMode == 'on': df[signal_column] = df['high_conf_prediction']
            else: 
                if not config.code: raise ValueError("TA Strategy 'code' is required for 'predictions' (hybrid) mode.")
                df = generate_ta_signals(df, config.code, params)
                df[signal_column] = 0
                hybrid_mode = config.params.get('hybridMode_single', 'AND')
                if hybrid_mode == 'AND':
                    df.loc[(df.get('ta_signal', 0) == 1) & (df['high_conf_prediction'] == 1), signal_column] = 1
                    df.loc[(df.get('ta_signal', 0) == -1) & (df['high_conf_prediction'] == -1), signal_column] = -1
                else: 
                    df.loc[(df.get('ta_signal', 0) == 1) | (df['high_conf_prediction'] == 1), signal_column] = 1
                    df.loc[(df.get('ta_signal', 0) == -1) | (df['high_conf_prediction'] == -1), signal_column] = -1

        df_sliced = df.loc[config.startDate:config.endDate].copy().fillna(0)
        if df_sliced.empty: raise ValueError("No data for date range after all processing.")

        results = run_backtest(
            df=df_sliced,
            signal_column=signal_column,
            initial_balance=config.initialBalance,
            fee=config.fee,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            risk_mode=config.riskManagementMode,
            risk_percent=config.riskPercentage,
            growth_target=config.growthCapitalTarget,
            min_atr_pct=min_atr_pct,
            trend_filter_period=trend_period,
            ml_mode=config.mlMode,
            min_adx_level=min_adx_level,
            trailing_stop_atr_mult=trailing_stop_atr_mult,
            trailing_stop_pct=trailing_stop_pct
        )
        
        safe_results = convert_numpy_types(results)
        return JSONResponse(content=safe_results)

    except Exception as e:
        logger.error(f"General Error in /run-backtest-on: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")

@app.post('/api/ml/run-combo-backtest')
async def handle_run_combo_backtest(config: BacktestConfig):
    try:
        params = config.params or {}
        strategies_config = config.strategies or []
        
        sl_pct = params.get('SL')
        tp_pct = params.get('TP')
        min_atr_pct = params.get('minAtrPct')
        trend_period = params.get('trendFilterPeriod')
        min_adx_level = params.get('minAdxLevel')
        trailing_stop_atr_mult = params.get('tslAtrMult')
        trailing_stop_pct = params.get('tslPct')
        hybrid_mode = params.get('hybridMode', 'AND')

        df_features = load_efficient_data(config.symbol, config.timeframe, config.startDate, config.endDate)
        df_features = engineer_features_for_backtest(df_features, trend_period, params)

        df_features['high_conf_prediction'] = 0
        if config.mlMode in ('predictions', 'on'):
            if not config.mlModel: raise ValueError("ML Model name is required for 'predictions'/'on' mode.")
            model_path = os.path.join(MODEL_DIR, f"{config.mlModel}.joblib")
            if not os.path.exists(model_path):
                model_path = os.path.join(MODEL_DIR, f"{config.mlModel}.pkl")
                if not os.path.exists(model_path): raise FileNotFoundError(f"Model file '{config.mlModel}.joblib' or '.pkl' not found in {MODEL_DIR}.")

            pipeline = joblib.load(model_path)
            class_indices = {'buy': 0, 'hold': 1, 'sell': 2} 

            if isinstance(pipeline, dict):
                model = pipeline.get('model')
                scaler = pipeline.get('scaler')
                model_feature_names = pipeline.get('feature_names', [])
                class_indices = pipeline.get('class_indices', class_indices)
                missing_cols = [c for c in model_feature_names if c not in df_features.columns]
                if missing_cols: raise KeyError(f"Missing feature columns for ML model: {missing_cols}")
                df_model_features = df_features[model_feature_names].copy().fillna(0)
                X_scaled_array = scaler.transform(df_model_features)
                if hasattr(model, 'feature_names_in_') or 'LGBM' in str(type(model)): X_scaled_input = pd.DataFrame(X_scaled_array, columns=model_feature_names, index=df_model_features.index)
                else: X_scaled_input = X_scaled_array
                probabilities = model.predict_proba(X_scaled_input)
            else:
                probabilities = pipeline.predict_proba(df_features.fillna(0))

            df_features['prob_buy'] = probabilities[:, class_indices.get('buy', 0)]
            df_features['prob_sell'] = probabilities[:, class_indices.get('sell', 2)]
            
            df_features.loc[(df_features['prob_buy'] > config.mlThreshold), 'high_conf_prediction'] = 1
            df_features.loc[(df_features['prob_sell'] > config.mlThreshold), 'high_conf_prediction'] = -1

        signal_columns_to_combine = []
        for i, strat_config in enumerate(strategies_config):
            ta_code = strat_config.get('code')
            if not ta_code: continue
            signal_col = f'ta_signal_{i}'
            df_features = generate_ta_signals(df_features, ta_code, params)
            df_features[signal_col] = df_features.get('ta_signal', pd.Series(0, index=df_features.index))
            signal_columns_to_combine.append(signal_col)

        if signal_columns_to_combine:
            if hybrid_mode == 'REGIME' and len(signal_columns_to_combine) >= 2:
                adx_threshold = params.get('regime_threshold', 25)
                try: adx_col = find_col(df_features, 'ADX')
                except KeyError: adx_col = 'ADX_14' 
                df_features['combined_ta'] = np.where(df_features[adx_col] > adx_threshold, df_features[signal_columns_to_combine[0]], df_features[signal_columns_to_combine[1]])
            elif hybrid_mode == 'AND':
                df_features['combined_ta'] = df_features[signal_columns_to_combine].apply(lambda row: 1 if (row == 1).all() else (-1 if (row == -1).all() else 0), axis=1)
            else: 
                df_features['combined_ta'] = df_features[signal_columns_to_combine].apply(lambda row: 1 if (row == 1).any() else (-1 if (row == -1).any() else 0), axis=1)
        else:
            df_features['combined_ta'] = 0

        signal_column = 'combined_final_signal'
        if config.mlMode == 'predictions':
            df_features[signal_column] = 0
            if hybrid_mode == 'AND':
                df_features.loc[(df_features['combined_ta'] == 1) & (df_features['high_conf_prediction'] == 1), signal_column] = 1
                df_features.loc[(df_features['combined_ta'] == -1) & (df_features['high_conf_prediction'] == -1), signal_column] = -1
            else:
                df_features.loc[(df_features['combined_ta'] == 1) | (df_features['high_conf_prediction'] == 1), signal_column] = 1
                df_features.loc[(df_features['combined_ta'] == -1) | (df_features['high_conf_prediction'] == -1), signal_column] = -1
        elif config.mlMode == 'on':
            df_features[signal_column] = df_features['high_conf_prediction']
        else: 
            df_features[signal_column] = df_features['combined_ta']

        df_sliced_combined = df_features.loc[config.startDate:config.endDate].copy().fillna(0)        
        combined_result = run_backtest(
            df=df_sliced_combined,
            signal_column=signal_column,
            initial_balance=config.initialBalance,
            fee=config.fee,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            risk_mode=config.riskManagementMode,
            risk_percent=config.riskPercentage,
            growth_target=config.growthCapitalTarget,
            min_atr_pct=min_atr_pct,
            trend_filter_period=trend_period,
            ml_mode=config.mlMode,
            min_adx_level=min_adx_level,
            trailing_stop_atr_mult=trailing_stop_atr_mult,
            trailing_stop_pct=trailing_stop_pct
        )

        if config.optimizer_mode:
            ticket_id = str(uuid.uuid4())
            cache_path = os.path.join(OPTIMIZER_CACHE_DIR, f"{ticket_id}.json")
            optimizer_results = {"metrics": combined_result.get("metrics"), "equityCurve": combined_result.get("equityCurve") }
            try:
                safe_results = convert_numpy_types(optimizer_results)
                with open(cache_path, 'w') as f: json.dump(safe_results, f)
                return {"optimizer_ticket_id": ticket_id}
            except Exception as e:
                logger.error(f"Failed to write optimizer cache file: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to write cache file: {e}")
        else:
            # --- FIX: Wrap Combo Result for Node.js Service Compatibility ---
            safe_results = convert_numpy_types(combined_result)
            return JSONResponse(content={"combinedResult": safe_results, "individualResults": []})

    except Exception as e:
        logger.error(f"General Error in /run-combo-backtest: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")

# -------------------------
# Main entry point for Uvicorn
# -------------------------
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
