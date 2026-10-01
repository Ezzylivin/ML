import os
import sys
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import logging
from xgboost import XGBClassifier
import ccxt
import warnings
import time

# 🟢 INTERNAL IMPORTS
sys.path.append(os.getcwd())
# Ensure this path matches your config
from app.config2 import MODEL_STORAGE_DIR 

# 🟢 SETTINGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("OmniTrainer")

# 🟢 CONFIGURATION
SYMBOLS = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'PEPE/USD']
TIMEFRAMES = ['1h', '4h']

def fetch_fresh_data(symbol, timeframe):
    """Fetch robust dataset for training."""
    logger.info(f"📡 Fetching fresh data for {symbol} {timeframe}...")
    try:
        ex = ccxt.binanceus()
        # Fetch significant history
        all_candles = []
        since = ex.parse8601('2023-01-01T00:00:00Z')
        
        # Limit loop to prevent infinite runs
        for _ in range(5): 
            candles = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not candles: break
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            time.sleep(0.5)
        
        if len(all_candles) < 500:
            logger.warning(f"⚠️ Not enough data for {symbol}")
            return None

        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Failed to download {symbol}: {e}")
        return None

def train_and_save(symbol, timeframe):
    clean_sym = symbol.split('/')[0].lower()
    
    # 1. Get Data
    df = fetch_fresh_data(symbol, timeframe)
    if df is None: return

    # 2. Engineer Features (EXACT MATCH to Backtester)
    # Trend
    df['sma_50'] = ta.sma(df['close'], length=50)
    df['sma_200'] = ta.sma(df['close'], length=200)
    st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
    df['st_trend'] = st.iloc[:, 1] if st is not None else 0
    
    # Momentum & Volatility
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx.iloc[:, 0] if adx is not None else 0
    
    # Range
    bb = ta.bbands(df['close'], length=20, std=2)
    df = pd.concat([df, bb], axis=1)

    # 3. Create Target (1 = Up, 0 = Down)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)

    # 4. Dynamic Feature Selection
    # (Captures BBL/BBU columns regardless of exact naming)
    bbl_col = [c for c in df.columns if c.startswith("BBL")][0]
    bbu_col = [c for c in df.columns if c.startswith("BBU")][0]
    
    features = [
        'sma_50', 'sma_200', 'st_trend', 
        'rsi', 'atr', 'adx',
        bbl_col, bbu_col
    ]
    
    # Filter valid columns
    valid_features = [f for f in features if f in df.columns]
    X = df[valid_features]
    y = df['target']
    
    logger.info(f"📊 {clean_sym} {timeframe}: Training on {len(valid_features)} features")

    # 5. Train Model
    model = XGBClassifier(n_estimators=100, max_depth=3, eval_metric='logloss')
    model.fit(X, y)

    # 6. Save with METADATA (Critical Step!)
    payload = {
        "model": model,
        "feature_names": valid_features, 
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    filename = f"{clean_sym}_{timeframe}_stacking_model.joblib"
    save_path = os.path.join(MODEL_STORAGE_DIR, filename)
    
    # Atomic Overwrite
    if os.path.exists(save_path):
        os.remove(save_path)
        
    joblib.dump(payload, save_path)
    logger.info(f"✅ SAVED: {filename}")

# 🟢 MAIN LOOP
if __name__ == "__main__":
    logger.info("🚀 STARTING GLOBAL MODEL UPDATE")
    if not os.path.exists(MODEL_STORAGE_DIR): os.makedirs(MODEL_STORAGE_DIR)

    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                train_and_save(sym, tf)
            except Exception as e:
                logger.error(f"❌ Failed on {sym} {tf}: {e}")
                continue
    
    logger.info("🏁 ALL MODELS UPDATED SUCCESSFULLY.")
