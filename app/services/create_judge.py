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

# 🟢 INTERNAL IMPORTS
sys.path.append(os.getcwd())
from app.config2 import MODEL_STORAGE_DIR, DATA_DIR

# 🟢 SETTINGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("UniversalRepair")

SYMBOL = "BTC/USD"
TIMEFRAME = "1h"
# We try to load all these. If one is missing, we skip it.
EXPERTS_TO_LOAD = ["RandomForest", "XGBoost", "LightGBM", "MLPClassifier"]

def fetch_fresh_data(symbol, timeframe):
    """Fetch enough data to cover all indicator lookbacks."""
    logger.info(f"📡 Fetching fresh data for {symbol}...")
    try:
        ex = ccxt.binanceus()
        since = ex.parse8601('2023-01-01T00:00:00Z')
        all_candles = []
        while True:
            candles = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not candles: break
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            if len(all_candles) > 6000: break 
        
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        return None

def robust_engineer_features(df):
    """
    Generates A SUPERSET of all features previously used.
    The models will pick what they need from this buffet.
    """
    df = df.copy()
    
    # 1. Standard Price
    # (open, high, low, close, volume already exist)

    # 2. Moving Averages
    df['sma_50'] = ta.sma(df['close'], length=50)
    df['sma_200'] = ta.sma(df['close'], length=200)
    df['sma_logic'] = df['sma_200'] 

    # 3. Momentum
    df['rsi'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df['macd'] = macd.iloc[:, 0]
    df['macds'] = macd.iloc[:, 2]

    # 4. Volatility
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_logic'] = df['atr'] 
    
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx.iloc[:, 0]
    df['adx_logic'] = df['adx'] 

    # 5. Bollinger Bands
    bb = ta.bbands(df['close'], length=20, std=2)
    df = pd.concat([df, bb], axis=1)

    # 6. SuperTrend
    st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
    if st is not None:
        st_col_name = st.columns[0] 
        df['st_trend'] = st[st_col_name]

    return df.dropna()

def load_expert_direct(name, symbol, timeframe):
    """
    Directly loads the joblib payload to retrieve the model AND its metadata.
    """
    # 🟢 FIX: Use split instead of replace to match file format (btc_1h...)
    clean_sym = symbol.split('/')[0].lower()
    
    fname = f"{clean_sym}_{timeframe}_{name.lower()}_model.joblib"
    path = os.path.join(MODEL_STORAGE_DIR, fname)
    
    if not os.path.exists(path):
        logger.warning(f"   ❌ File not found: {fname}")
        return None

    try:
        payload = joblib.load(path)
        if isinstance(payload, dict) and 'model' in payload:
             # This is our custom trained dict
             return payload
        else:
             logger.warning(f"   ⚠️ {name} is not in the correct dictionary format.")
             return None
    except Exception as e:
        logger.error(f"   ❌ Error loading {name}: {e}")
        return None

def repair_process():
    # 🟢 FIX: Use split logic for saving too
    clean_sym = SYMBOL.split('/')[0].lower()
    
    # 1. Get Data
    df_raw = fetch_fresh_data(SYMBOL, TIMEFRAME)
    if df_raw is None: return
    
    df = robust_engineer_features(df_raw)
    logger.info(f"✅ Data Engineered. Columns available: {len(df.columns)}")

    # 2. Load Experts Directly
    experts = {}
    valid_expert_names = []
    
    logger.info("🧠 Loading Experts and Metadata from disk...")
    
    for name in EXPERTS_TO_LOAD:
        payload = load_expert_direct(name, SYMBOL, TIMEFRAME)
        
        if payload:
            model = payload['model']
            scaler = payload.get('scaler')
            feature_names = payload.get('feature_names', [])
            
            # Check if we have the columns
            missing = [f for f in feature_names if f not in df.columns]
            
            if missing:
                logger.warning(f"   ⚠️ {name} missing columns in current data: {missing}")
            else:
                experts[name] = {
                    'model': model,
                    'scaler': scaler,
                    'features': feature_names
                }
                valid_expert_names.append(name)
                logger.info(f"   ✅ {name} loaded (Needs {len(feature_names)} features)")

    if len(experts) < 2:
        logger.error("🛑 Not enough valid experts to build a Judge. Aborting.")
        return

    # 3. Generate Votes
    logger.info(f"⚖️  Asking {len(experts)} experts to vote on {len(df)} candles...")
    
    votes_data = []
    targets = []
    
    count = 0
    # Start loop where we have enough history
    for i in range(200, len(df)-1):
        try:
            row_votes = []
            
            for name, expert in experts.items():
                # Extract exactly the columns this model learned on
                cols = expert['features']
                row_data = df[cols].iloc[i].values.reshape(1, -1)
                
                # Scale if scaler exists
                if expert['scaler']:
                    row_data = expert['scaler'].transform(row_data)
                
                # Predict (2 = Buy in our class indices)
                # Some models output [Sell, Hold, Buy], others just probabilities
                probs = expert['model'].predict_proba(row_data)[0]
                
                # We want the probability of BUY (index 2 usually, or last index)
                buy_prob = probs[-1] 
                row_votes.append(buy_prob)
            
            # Ground Truth: Price went up next candle
            target = 1 if df.iloc[i+1]['close'] > df.iloc[i]['close'] else 0
            
            votes_data.append(row_votes)
            targets.append(target)
            
            count += 1
            if count % 1000 == 0: print(f"   Processed {count} candles...", end="\r")
            
        except Exception as e:
            continue

    if len(votes_data) == 0:
        logger.error("No votes collected.")
        return

    X_meta = np.array(votes_data)
    y_meta = np.array(targets)
    
    logger.info(f"\n📊 Calibration Data Shape: {X_meta.shape}")

    # 4. Train The Judge
    logger.info("🔨 Training Stacking Judge (XGBoost)...")
    judge = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        objective='binary:logistic',
        n_jobs=1
    )
    judge.fit(X_meta, y_meta)

    # 5. Save
    # 🟢 FIX: Save as btc_1h_stacking_model.joblib (underscore)
    filename = f"{clean_sym}_{TIMEFRAME}_stacking_model.joblib"
    save_path = os.path.join(MODEL_STORAGE_DIR, filename)
    
    # Save simply
    joblib.dump(judge, save_path)
    
    logger.info(f"✅ JUDGE SAVED: {filename}")
    logger.info(f"   Experts used: {valid_expert_names}")

if __name__ == "__main__":
    repair_process()
