import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import pandas_ta as ta
from pymongo import MongoClient

# --- 1. CONFIGURATION ---
try:
    from app.config2 import MONGO_URI, MODEL_STORAGE_DIR
    import app.config2 as cfg
    DATABASE_NAME = getattr(cfg, 'DATABASE_NAME', getattr(cfg, 'DB_NAME', 'SovereignDB'))
except ImportError:
    MONGO_URI = "mongodb+srv://ericjdickerson93:Dadeadend1!!@cluster0.wzztvhe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0" 
    MODEL_STORAGE_DIR = "/root/Project/ML/app/models"
    DATABASE_NAME = "SovereignDB"

# --- 2. THE JURY: Bi-Directional Factory ---
class InternalFactory:
    @staticmethod
    def get_prediction(model_type, symbol, timeframe, df):
        """Returns (Short_Prob, Long_Prob)"""
        clean_symbol = symbol.split('-')[0].lower()
        base_name = model_type.lower()
        ext = "keras" if base_name == "transformer" else "joblib"
        path = os.path.join(MODEL_STORAGE_DIR, f"{clean_symbol}_{timeframe}_{base_name}_model.{ext}")

        if not os.path.exists(path): return None

        features = ['open', 'high', 'low', 'close', 'rsi', 'atr', 'adx', 'macd', 'macds', 'st_trend']
        try:
            if ext == "keras":
                model = tf.keras.models.load_model(path, compile=False)
                data = df[features].tail(50).values
                x = data.reshape(1, 50, -1).astype('float32')
                preds = model(x, training=False).numpy()[0]
                # Class 0 = Short, Class 2 = Long
                return float(preds[0]), float(preds[2])
            else:
                payload = joblib.load(path)
                model = payload['model'] if isinstance(payload, dict) else payload
                data = df[features].tail(1).values
                probs = model.predict_proba(data)[0]
                return float(probs[0]), float(probs[2])
        except Exception:
            return 0.33, 0.33

# --- 3. THE DATA ENGINE ---
class AuditEngine:
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DATABASE_NAME]

    def fetch_data(self, symbol, timeframe):
        possible_names = [f"{symbol}_{timeframe}", f"{symbol.replace('-', '/')}_{timeframe}"]
        df = pd.DataFrame()
        for name in possible_names:
            cursor = self.db[name].find().sort("timestamp", -1).limit(300)
            df = pd.DataFrame(list(cursor))
            if not df.empty and len(df) >= 50: break
        
        if df.empty or len(df) < 50: return None
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.columns = [c.lower() for c in df.columns]

        # Indicators
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['adx'] = ta.adx(df['high'], df['low'], df['close']).iloc[:, 0]
        macd = ta.macd(df['close'])
        df['macd'], df['macds'] = macd.iloc[:, 0], macd.iloc[:, 2]
        st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
        df['st_trend'] = st.iloc[:, 1]
        return df.dropna()

# --- 4. EXECUTION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
EXPERTS = ["xgboost", "randomforest", "lightgbm", "mlpclassifier", "transformer"]

def run_audit(symbol, timeframe):
    engine = AuditEngine()
    df = engine.fetch_data(symbol, timeframe)
    
    print(f"\n🎭 BI-DIRECTIONAL AUDIT: {symbol} | {timeframe}")
    print("=" * 100)
    print(f"{'EXPERT':<15} | {'SHORT %':<15} | {'LONG %':<15} | {'BIAS'}")
    print("-" * 100)

    if df is None:
        print("Error: DB data missing.")
        return

    all_short, all_long = [], []
    for ex in EXPERTS:
        s_prob, l_prob = InternalFactory.get_prediction(ex, symbol, timeframe, df)
        bias = "🔴 SELL" if s_prob > l_prob else "🟢 BUY " if l_prob > s_prob else "⚪ NEUT"
        print(f"{ex:<15} | {s_prob:<15.2f} | {l_prob:<15.2f} | {bias}")
        all_short.append(s_prob)
        all_long.append(l_prob)

    avg_s, avg_l = np.mean(all_short), np.mean(all_long)
    final_bias = "🔥🔥 STRONG SHORT" if avg_s > 0.6 else "🚀 STRONG LONG" if avg_l > 0.6 else "💤 NO TRADE"
    
    print("-" * 100)
    print(f"{'COUNCIL AVG':<15} | {avg_s:<15.2f} | {avg_l:<15.2f} | {final_bias}")
    print("=" * 100)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC-USD")
    p.add_argument("--timeframe", default="1h")
    args = p.parse_args()
    run_audit(args.symbol, args.timeframe)
