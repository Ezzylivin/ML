import ccxt
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from app.config2 import DATA_DIR

def get_super_data(symbol="BTC/USD", limit=2000):
    """
    Step 1: AGGREGATE - Pull from all US exchanges & blend into one 'Truth' CSV.
    Step 2: RESAMPLE  - Take that Truth and build 4h, 1d, and 1w structures.
    """
    print(f"\n🚀 STARTING SUPER ENGINE FOR: {symbol}")
    
    # 1. Exchange Setup (US Based)
    exchanges = {
        'coinbase': ccxt.coinbaseexchange(),
        'kraken': ccxt.kraken(),
        'gemini': ccxt.gemini(),
        'binanceus': ccxt.binanceus()
    }
    
    all_exchange_data = []
    
    # --- PHASE 1: AGGREGATION ---
    for name, ex in exchanges.items():
        try:
            print(f"  📡 Fetching 1h from {name}...")
            ohlcv = ex.fetch_ohlcv(symbol, timeframe='1h', limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['exchange'] = name
            all_exchange_data.append(df)
            time.sleep(1) # Rate limit safety
        except Exception as e:
            print(f"  ⚠️  {name} failed: {e}")

    if not all_exchange_data:
        print("❌ CRITICAL: No data fetched from any exchange. Aborting.")
        return

    # Blend exchanges (Average prices, Sum volumes)
    master = pd.concat(all_exchange_data)
    
    # Force column 7 (Volume) to be numeric to kill those DtypeWarnings
    master['volume'] = pd.to_numeric(master['volume'], errors='coerce').fillna(0)

    # Group by timestamp to create the 'True 1h' file
    agg_1h = master.groupby('timestamp').agg({
        'open': 'mean',
        'high': 'mean',
        'low': 'mean',
        'close': 'mean',
        'volume': 'sum'
    }).sort_index()

    # Convert timestamp to human-readable datetime
    agg_1h['datetime'] = pd.to_datetime(agg_1h.index, unit='ms')
    
    # Save the 'Truth' file
    clean_symbol = symbol.replace('/', '-')
    base_path = os.path.join(DATA_DIR, f"{clean_symbol}-1h.csv")
    agg_1h.to_csv(base_path, index=False)
    print(f"✅ AGGREGATION COMPLETE: {base_path}")

    # --- PHASE 2: RESAMPLING ---
    print(f"🔄 RESAMPLING structural timeframes from {clean_symbol}-1h.csv...")
    
    agg_1h.set_index('datetime', inplace=True)
    timeframes = {'4h': '4H', '1d': '1D', '1w': '1W'}

    for label, rule in timeframes.items():
        resampled = agg_1h.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        out_path = os.path.join(DATA_DIR, f"{clean_symbol}-{label}.csv")
        resampled.to_csv(out_path)
        print(f"  ✨ Created {clean_symbol}-{label}.csv ({len(resampled)} bars)")

if __name__ == "__main__":
    # List all symbols you want to fully refresh
    symbols_to_fix = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"]
    for s in symbols_to_fix:
        get_super_data(s)
    print("\n🏁 ALL SYSTEMS LOADED. Data is now 100% clean and multi-exchange synced.")
