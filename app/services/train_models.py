import os
import sys
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import logging
from datetime import datetime, timezone
import time
import ccxt
import warnings

# --- Machine Learning Imports ---
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

# --- Deep Learning Imports ---
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping

# 🟢 STABILITY: Force CPU mode for training to ensure system stability
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# --- Configuration ---
TOP_SYMBOLS = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'SOL/USD', 'PEPE/USD']
TIMEFRAMES = ['1h', '4h']
START_DATE_DOWNLOAD = '2020-01-01'
EXCHANGES_TO_TRY = ['binanceus', 'coinbase', 'kraken']

MODEL_SAVE_DIR = '/root/Project/ML/app/models'
DATA_DIR = '/root/Project/ML/data'

LOOK_BACK_WINDOW = 50  # For Transformer Sequence
LOOK_FORWARD_CANDLES = 24
PROFIT_TARGET_PCT = 1.5
STOP_LOSS_PCT = 1.0
MIN_ROWS_TO_TRAIN = 1000

# Registry for the Decision Council
MODEL_CONFIGS = [
    {
        'name': 'RandomForest',
        'type': 'sklearn',
        'estimator': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        'param_grid': {'n_estimators': [100, 300], 'max_depth': [10, 30]}
    },
    {
        'name': 'XGBoost',
        'type': 'sklearn',
        'estimator': xgb.XGBClassifier(objective='multi:softprob', num_class=3, random_state=42),
        'param_grid': {'n_estimators': [100, 300], 'learning_rate': [0.01, 0.05]}
    },
    {
        'name': 'LightGBM',
        'type': 'sklearn',
        'estimator': lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42),
        'param_grid': {'n_estimators': [100, 300], 'learning_rate': [0.01, 0.05]}
    },
    {
        'name': 'MLPClassifier',
        'type': 'sklearn',
        'estimator': MLPClassifier(random_state=42, max_iter=500),
        'param_grid': {'hidden_layer_sizes': [(100, 50), (50, 50)]}
    },
    {
        'name': 'Transformer',
        'type': 'keras',
        'lookback': 50
    }
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("SovereignTrainer")

# --- 🛠️ CORE FUNCTIONS ---

def build_transformer(input_shape):
    """Deep Learning Attention model for time-series forecasting."""
    inputs = Input(shape=input_shape)
    x = LayerNormalization(epsilon=1e-6)(inputs)
    att = MultiHeadAttention(num_heads=4, key_dim=input_shape[-1])(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + att)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(3, activation="softmax")(x) # 3-Class: Sell, Hold, Buy
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model

def create_sequences(X, y, window):
    """Transforms tabular data into 3D tensors for the Transformer."""
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X.iloc[i:(i + window)].values)
        ys.append(y.iloc[i + window])
    return np.array(Xs), np.array(ys)

def fetch_data(symbol, timeframe):
    """CCXT robust downloader."""
    since = int(datetime.strptime(START_DATE_DOWNLOAD, '%Y-%m-%d').timestamp() * 1000)
    for ex_id in EXCHANGES_TO_TRY:
        try:
            ex = getattr(ccxt, ex_id)()
            candles = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=5000)
            if len(candles) > 500:
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('datetime', inplace=True)
                return df
        except: continue
    return None

def engineer_features(df):
    """Calculates all Sovereign Signal indicators to match the Backtester."""
    df = df.copy()
    
    # --- TREND ---
    # We rename them to match the Backtester's keys exactly
    df['sma_50'] = ta.sma(df['close'], length=50)   # Used for Breakout Strategy
    df['sma_200'] = ta.sma(df['close'], length=200) # Used for Trend Filter
    df['sma_logic'] = df['sma_200']                 # Alias
    
    st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
    df['st_trend'] = st.iloc[:, 1]
    
    # --- MOMENTUM ---
    df['rsi'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df['macd'] = macd.iloc[:, 0]
    df['macds'] = macd.iloc[:, 2]
    
    # --- VOLATILITY & RANGE ---
    # Pandas TA returns columns like BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
    bb = ta.bbands(df['close'], length=20, std=2)
    df = pd.concat([df, bb], axis=1) # Append BB columns directly
    
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_logic'] = df['atr'] # Alias
    
    # ADX is critical for Regime Switching
    adx_val = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx_val.iloc[:, 0]
    df['adx_logic'] = df['adx'] # Alias

    # --- Target Labeling ---
    df['target'] = 0 # Hold
    future_change = df['close'].shift(-LOOK_FORWARD_CANDLES) / df['close'] - 1
    df.loc[future_change > (PROFIT_TARGET_PCT/100), 'target'] = 2 # Buy
    df.loc[future_change < -(STOP_LOSS_PCT/100), 'target'] = 0    # Sell
    df.loc[(future_change <= (PROFIT_TARGET_PCT/100)) & (future_change >= -(STOP_LOSS_PCT/100)), 'target'] = 1 # Hold
    
    return df.dropna()

# --- 🚀 MAIN TRAINING EXECUTION ---

if __name__ == "__main__":
    logger.info("☢️ NUCLEAR RESET: Starting Council Training...")
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    
    for symbol in TOP_SYMBOLS:
        for timeframe in TIMEFRAMES:
            logger.info(f"Processing {symbol} @ {timeframe}")
            df_raw = fetch_data(symbol, timeframe)
            if df_raw is None: continue
            
            df = engineer_features(df_raw)
            
            # 🟢 UPGRADE: Features List
            # This matches the new list in ModelFactory.load_model()
            # We explicitly include the Trend/Range indicators
            features = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'atr', 'adx', 'st_trend',
                'sma_50', 'sma_200',  # For Trend Logic
                'BBL_20_2.0', 'BBU_20_2.0' # For Range Logic
            ]
            
            # Double check columns exist
            final_features = [f for f in features if f in df.columns]
            
            X = df[final_features]
            y = df['target']
            
            # Split
            split = int(len(df) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            for config in MODEL_CONFIGS:
                name = config['name']
                asset_clean = symbol.split('/')[0].lower()
                
                if config['type'] == 'sklearn':
                    fname = f"{asset_clean}_{timeframe}_{name.lower()}_model.joblib"
                    save_path = os.path.join(MODEL_SAVE_DIR, fname)
                    
                    logger.info(f"Training Expert: {name}")
                    search = RandomizedSearchCV(config['estimator'], config['param_grid'], n_iter=10, cv=3)
                    search.fit(X_train_scaled, y_train)
                    
                    payload = {
                        'model': search.best_estimator_,
                        'scaler': scaler,
                        'feature_names': final_features,
                        'class_indices': {'sell': 0, 'hold': 1, 'buy': 2}
                    }
                    joblib.dump(payload, save_path)
                    logger.info(f"✅ Saved {fname}")

                elif config['type'] == 'keras':
                    fname = f"{asset_clean}_{timeframe}_{name.lower()}_model.keras"
                    save_path = os.path.join(MODEL_SAVE_DIR, fname)
                    
                    logger.info(f"Training Neural Expert: {name}")
                    X_seq, y_seq = create_sequences(pd.DataFrame(X_train_scaled), y_train, LOOK_BACK_WINDOW)
                    
                    if len(X_seq) > 0:
                        model = build_transformer((LOOK_BACK_WINDOW, X_seq.shape[2]))
                        model.fit(X_seq, y_seq, epochs=10, batch_size=32, verbose=0)
                        model.save(save_path)
                        logger.info(f"✅ Saved {fname}")
                    else:
                        logger.warning(f"⚠️ Not enough data for Transformer on {symbol}")

    logger.info("🏁 COUNCIL REBUILT. Audit ready.")
