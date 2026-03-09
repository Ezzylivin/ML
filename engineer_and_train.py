import os
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import sys
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Ensure root is in path
sys.path.append(os.getcwd())

from app.config2 import DATA_DIR, MODEL_STORAGE_DIR

# 🟢 SHARED FEATURE ENGINE
def apply_mega_features(df):
    df.columns = [c.lower() for c in df.columns]
    
    # 1. Moving Averages
    df['sma_50'] = ta.sma(df['close'], length=50)
    df['sma_200'] = ta.sma(df['close'], length=200)
    df['ema_9'] = ta.ema(df['close'], length=9)
    df['ema_21'] = ta.ema(df['close'], length=21)
    df['ema_20'] = ta.ema(df['close'], length=20)
    
    # 2. Oscillators
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx_df.iloc[:, 0] if adx_df is not None else 0
    
    # 3. Bollinger & Supertrend
    bb = ta.bbands(df['close'], length=20, std=2.0)
    df['BBL_20_2.0_2.0'] = bb.iloc[:, 0] if bb is not None else 0
    df['BBU_20_2.0_2.0'] = bb.iloc[:, 2] if bb is not None else 0
    st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
    df['st_trend'] = st.iloc[:, 1] if st is not None else 0

    # 4. Momentum
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    df['STOCHk_14_3_3'] = stoch.iloc[:, 0] if stoch is not None else 50
    macd = ta.macd(df['close'])
    df['MACD_12_26_9'] = macd.iloc[:, 0] if macd is not None else 0
    df['MACDs_12_26_9'] = macd.iloc[:, 2] if macd is not None else 0

    # 5. Price Action / Vol
    df['pa_high'] = df['high'].rolling(window=20).max()
    df['pa_low'] = df['low'].rolling(window=20).min()
    df['vol_ma'] = ta.sma(df['volume'], length=20)
    
    # 6. Model Logic Features
    df['adx_logic'] = np.where(df['adx'] > 25, 1, 0)
    df['atr_logic'] = (df['atr'] / df['close']) * 1000
    df['sma_logic'] = np.where(df['close'] > df['sma_200'], 1, -1)
    
    features = [
        'open', 'high', 'low', 'close', 'volume',
        'sma_50', 'sma_200', 'ema_9', 'ema_21', 'ema_20',
        'rsi', 'atr', 'adx', 'st_trend', 
        'BBL_20_2.0_2.0', 'BBU_20_2.0_2.0', 
        'STOCHk_14_3_3', 'MACD_12_26_9', 'MACDs_12_26_9',
        'pa_high', 'pa_low', 'vol_ma', 
        'adx_logic', 'atr_logic', 'sma_logic'
    ]
    return df.dropna(subset=features), features

def train_all_symbols():
    # symbols must match your CSV file names in DATA_DIR
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "SUI-USD", "PEPE-USD"]
    
    print("\n🚀 STARTING MEGA-TRAINING (25 Features)")
    
    for symbol in symbols:
        try:
            path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
            if not os.path.exists(path):
                print(f"  ⚪ Skipped {symbol}: File {path} not found")
                continue

            print(f"🧠 Processing {symbol}...")
            raw_df = pd.read_csv(path)
            df, feats = apply_mega_features(raw_df)
            
            if len(df) < 500:
                print(f"  ⚠️ {symbol} ignored: Only {len(df)} rows left after indicators.")
                continue

            X = df[feats].values
            y = (df['close'].shift(-1) > df['close']).astype(int).values[:-1]
            X = X[:-1]

            # 🎯 Align with your specific app/models naming convention
            ticker = symbol.split('-')[0].lower()
            save_path_xgb = f"app/models/{ticker}_1h_xgboost_model.joblib"
            save_path_rf = f"app/models/{ticker}_1h_randomforest_model.joblib"

            # Train XGBoost
            xgb = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, base_score=0.5)
            xgb.fit(X, y)
            joblib.dump({"model": xgb, "feature_names": feats}, save_path_xgb)
            
            # Train RandomForest
            rf = RandomForestClassifier(n_estimators=100, max_depth=10)
            rf.fit(X, y)
            joblib.dump({"model": rf, "feature_names": feats}, save_path_rf)

            print(f"  ✅ {symbol} SUCCESS: Saved to {save_path_xgb}")

        except Exception as e:
            print(f"  ❌ {symbol} CRASH: {e}")

if __name__ == "__main__":
    train_all_symbols()
