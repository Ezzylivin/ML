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
from app.predictors.model_factory import ModelFactory

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
    df['sma_logic'] = df['sma_200'] # Alias used in some versions

    # 3. Momentum
    df['rsi'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df['macd'] = macd.iloc[:, 0]
    df['macds'] = macd.iloc[:, 2]

    # 4. Volatility
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_logic'] = df['atr'] # Alias
    
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx.iloc[:, 0]
    df['adx_logic'] = df['adx'] # Alias

    # 5. Bollinger Bands (The likely culprit for mismatch)
    bb = ta.bbands(df['close'], length=20, std=2)
    # Rename columns to match what training usually expects just in case
    # pandas_ta outputs: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
    df = pd.concat([df, bb], axis=1)

    # 6. SuperTrend (The other likely culprit)
    st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
    # Supertrend output usually has 2 cols: SUPERT_10_3.0 and SUPERTd_10_3.0
    # We need to explicitly name one 'st_trend'
    st_col_name = st.columns[0] 
    df['st_trend'] = st[st_col_name]

    return df.dropna()

def repair_process():
    clean_sym = SYMBOL.replace('/', '-').lower()
    
    # 1. Get Data
    df_raw = fetch_fresh_data(SYMBOL, TIMEFRAME)
    if df_raw is None: return
    
    df = robust_engineer_features(df_raw)
    logger.info(f"✅ Data Engineered. Columns available: {len(df.columns)}")

    # 2. Load Experts & Check Requirements
    experts = {}
    valid_expert_names = []
    
    logger.info("🧠 Checking Experts and their requirements...")
    
    for name in EXPERTS_TO_LOAD:
        # Load using factory
        model_wrapper = ModelFactory.load_model(name, symbol=clean_sym, timeframe=TIMEFRAME)
        
        if model_wrapper:
            # CRITICAL STEP: Identify what this specific model wants
            needed_features = model_wrapper.feature_names
            missing = [f for f in needed_features if f not in df.columns]
            
            if missing:
                logger.warning(f"⚠️ {name} requires features missing from dataframe: {missing}")
                logger.warning(f"   Skipping {name} to prevent shape mismatch.")
            else:
                experts[name] = model_wrapper
                valid_expert_names.append(name)
                logger.info(f"   ✅ {name} ready (expects {len(needed_features)} features)")
        else:
            logger.warning(f"   ❌ {name} not found on disk.")

    if len(experts) < 2:
        logger.error("🛑 Not enough valid experts to build a Judge. Aborting.")
        return

    # 3. Generate Votes (The Calibration)
    logger.info(f"⚖️  Asking {len(experts)} experts to vote on {len(df)} candles...")
    
    votes_data = []
    targets = []
    
    # Use a safe slice loop
    # We start at 200 to ensure even sma_200 is valid
    count = 0
    for i in range(200, len(df)-1):
        try:
            # We assume ModelFactory handles the scaling internally if we pass the specific row
            # But ModelFactory.predict_direction usually expects a DataFrame slice
            row_slice = df.iloc[i-100:i+1] # Give it context
            
            row_votes = []
            for name, expert in experts.items():
                # This calls the internal predict using the features IT specifically wants
                prob = expert.predict_direction(row_slice)
                row_votes.append(prob)
            
            # Ground Truth: Next candle Close > Current Close
            target = 1 if df.iloc[i+1]['close'] > df.iloc[i]['close'] else 0
            
            votes_data.append(row_votes)
            targets.append(target)
            count += 1
            if count % 1000 == 0: print(f"   Processed {count} candles...", end="\r")
            
        except Exception as e:
            # If a specific row fails, skip it rather than crashing
            continue

    if len(votes_data) == 0:
        logger.error("No votes collected. Check data alignment.")
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
    filename = f"{clean_sym}_{TIMEFRAME}_stacking_model.joblib"
    save_path = os.path.join(MODEL_STORAGE_DIR, filename)
    
    # We save a simple dict wrapper
    payload = {
        'model': judge,
        'is_stacking': True,
        'expert_names': valid_expert_names
    }
    joblib.dump(payload, save_path)
    
    logger.info(f"✅ JUDGE SAVED: {filename}")
    logger.info(f"   Experts used: {valid_expert_names}")

if __name__ == "__main__":
    repair_process()
