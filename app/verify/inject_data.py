import ccxt
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import time
import os

# --- 🟢 CONFIGURATION ---
# The script will try to pull from your config2 first, then fallback to these defaults
try:
    from app.config2 import MONGO_URI, DATABASE_NAME
except ImportError:
    MONGO_URI = "mongodb+srv://ericjdickerson93:Dadeadend1!!@cluster0.wzztvhe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0" # Fallback if config not found
    DATABASE_NAME = "SovereignDB"

# US-Compliant Symbols and Exchanges
SYMBOLS = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD']
TIMEFRAME = '1h'
EXCHANGE_ID = 'coinbase' # Highly reliable for US data

def inject_data():
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    
    # Initialize a US-compliant exchange
    print(f"🏛️ Initializing {EXCHANGE_ID} for US-compliant data...")
    exchange = getattr(ccxt, EXCHANGE_ID)()

    print(f"🚀 Starting Data Injection into {DATABASE_NAME}...")

    for symbol in SYMBOLS:
        try:
            print(f"📦 Fetching {symbol} ({TIMEFRAME}) from {EXCHANGE_ID}...")
            
            # Fetch last 250 candles to ensure we have a solid buffer for indicators
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=250)
            
            # Formatting collection name: BTC-USD_1h
            coll_name = symbol.replace("/", "-") + "_" + TIMEFRAME
            collection = db[coll_name]
            
            # Clear old data to ensure we are using the freshest 250 bars
            collection.delete_many({})
            
            data_to_insert = []
            for candle in ohlcv:
                data_to_insert.append({
                    "timestamp": candle[0],
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
                    "datetime": datetime.fromtimestamp(candle[0] / 1000.0)
                })
            
            if data_to_insert:
                collection.insert_many(data_to_insert)
                print(f"✅ Injected {len(data_to_insert)} candles into [{coll_name}]")
            else:
                print(f"⚠️ No data returned for {symbol}")
            
        except Exception as e:
            print(f"❌ Error for {symbol}: {e}")
        
        # Respect API rate limits
        time.sleep(2)

    print("\n🏁 Injection Complete. The Council now has 'fuel' to run the audit.")

if __name__ == "__main__":
    inject_data()
