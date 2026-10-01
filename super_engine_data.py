import ccxt
import pandas as pd
import numpy as np
import time
import os
from app.config2 import DATA_DIR

def get_super_data(symbol="BTC/USD", limit=2000):
    print(f"\n🚀 STARTING US-AGGREGATOR FOR: {symbol}")
    
    # 1. Exchange Setup (US Based Only)
    # Note: 'ccxt.coinbaseexchange' is the correct ID for Advanced Trade
    exchanges = {
        'coinbase': ccxt.coinbaseexchange({'enableRateLimit': True}),
        'kraken': ccxt.kraken({'enableRateLimit': True}),
        'gemini': ccxt.gemini({'enableRateLimit': True}),
        'binanceus': ccxt.binanceus({'enableRateLimit': True})
    }
    
    all_dfs = []
    
    # --- PHASE 1: AGGREGATION ---
    for name, ex in exchanges.items():
        try:
            print(f"  📡 Syncing 1h from {name}...")
            # We fetch more than the limit to ensure enough overlap for blending
            ohlcv = ex.fetch_ohlcv(symbol, timeframe='1h', limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 🛑 CRITICAL FIX: Ensure numeric types before any math
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['exchange'] = name
            all_dfs.append(df)
            time.sleep(1.5) # Extra safety for US rate limits
        except Exception as e:
            print(f"  ⚠️  {name} skipped: {str(e)[:50]}...")

    if not all_dfs:
        print("❌ FAILED: No data sources responded.")
        return

    # 2. CREATE THE MASTER 'TRUTH' FILE
    # We blend by taking the MEDIAN price (to ignore outlier flash crashes) 
    # and the SUM of volume across all US venues.
    master = pd.concat(all_dfs)
    agg_1h = master.groupby('timestamp').agg({
        'open': 'median',
        'high': 'max',   # True High is the highest across all US venues
        'low': 'min',    # True Low is the lowest across all US venues
        'close': 'median',
        'volume': 'sum'
    }).sort_index()

    # Convert to Human Readable
    agg_1h['datetime'] = pd.to_datetime(agg_1h.index, unit='ms', utc=True)
    
    # --- PHASE 2: RESAMPLING (The Multi-Timeframe Fix) ---
    # We use 'closed=left' and 'label=left' to ensure parity with standard charting
    agg_1h.set_index('datetime', inplace=True)
    
    # Save 1h 'Truth'
    clean_name = symbol.replace('/', '-')
    agg_1h.to_csv(os.path.join(DATA_DIR, f"{clean_name}-1h.csv"))
    print(f"✅ 1h TRUTH SAVED: {len(agg_1h)} bars.")

    timeframes = {'4h': '4h', '1d': '1d', '1w': '1w'}
    for label, rule in timeframes.items():
        # Standardize resample rules for Pandas 2.0+
        resample_rule = rule.upper() if label != '1w' else 'W-SUN' 
        
        resampled = agg_1h.resample(resample_rule, closed='left', label='left').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        out_path = os.path.join(DATA_DIR, f"{clean_name}-{label}.csv")
        resampled.to_csv(out_path)
        print(f"  ✨ Resampled {label}: {len(resampled)} bars.")

if __name__ == "__main__":
    # The full list of your intended symbols
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP-USD", "ADA/USD", "DOGE/USD"]
    for s in symbols:
        # Standardize symbol format for CCXT
        ccxt_symbol = s.replace("-", "/") 
        get_super_data(ccxt_symbol)
