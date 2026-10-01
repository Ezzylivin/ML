import pandas as pd
import pandas_ta as ta
import joblib
from xgboost import XGBClassifier
import os
import logging

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ForceTrainer")

# 🟢 CONFIG: Exact path used by the API
MODEL_DIR = "app/models"
MODEL_PATH = os.path.join(MODEL_DIR, "sol_1h_stacking_model.joblib")
DATA_FILE = "data/SOL-USD-1h.csv"

def train_now():
    logger.info(f"🚀 FORCE TRAINING: {MODEL_PATH}")
    
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        logger.error(f"❌ Data file missing: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # 2. Engineer Features (Must match Backtester logic)
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
    
    # Range (Bollinger Bands)
    bb = ta.bbands(df['close'], length=20, std=2)
    df = pd.concat([df, bb], axis=1)

    # 3. Create Target (1 = Price goes UP)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)

    # 4. Select Features
    # We explicitly define the list so we can save it in metadata
    features = [
        'sma_50', 'sma_200', 'st_trend', 
        'rsi', 'atr', 'adx',
        'BBL_20_2.0', 'BBU_20_2.0'  # Correct Pandas-TA names
    ]
    
    # Verify columns exist
    valid_features = [f for f in features if f in df.columns]
    X = df[valid_features]
    y = df['target']
    
    logger.info(f"📊 Training on {len(valid_features)} features: {valid_features}")

    # 5. Train Model
    model = XGBClassifier(n_estimators=100, max_depth=3, eval_metric='logloss')
    model.fit(X, y)

    # 6. Save with METADATA (The Critical Part)
    payload = {
        "model": model,
        "feature_names": valid_features, # <--- The Backtester needs this!
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    # Ensure dir exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Force delete old file if exists
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        
    joblib.dump(payload, MODEL_PATH)
    logger.info(f"✅ SUCCESS: Overwrote {MODEL_PATH}")
    logger.info("   Metadata saved. Backtester will now work.")

if __name__ == "__main__":
    train_now()
