import ccxt
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from app.config2 import DATA_DIR

def fetch_paginated_ohlcv(exchange, symbol, timeframe='1h', target_limit=5000):
    """Walks backward in time to fetch deep history in batches."""
    all_ohlcv = []
    # Current time in milliseconds
    since = exchange.milliseconds() - (target_limit * 60 * 60 * 1000)
    
    while len(all_ohlcv) < target_limit:
        try:
            # Fetch batch
            limit = min(1000, target_limit - len(all_ohlcv))
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            # Update 'since' to the last bar's timestamp + 1ms to avoid overlap
            since = ohlcv[-1][0] + (60 * 60 * 1000) 
            
            # If we got fewer than 100 bars, we've likely hit the beginning of history
            if len(ohlcv) < 100:
                break
                
            time.sleep(exchange.rateLimit / 1000) # Respect US exchange limits
        except Exception as e:
            print(f"      ⚠️ Pagination error: {str(e)[:50]}")
            break
            
    return all_ohlcv

def get_super_data(symbol="BTC/USD", target_limit=5000):
    print(f"\n🚀 STARTING DEEP US-AGGREGATOR FOR: {symbol}")
    
    exchanges = {
        'coinbase': ccxt.coinbaseexchange({'enableRateLimit': True}),
        'kraken': ccxt.kraken({'enableRateLimit': True}),
        'gemini': ccxt.gemini({'enableRateLimit': True}),
        'binanceus': ccxt.binanceus({'enableRateLimit': True})
    }
    
    all_dfs = []
    
    for name, ex in exchanges.items():
        try:
            print(f"  📡 Syncing deep history from {name}...")
            ohlcv = fetch_paginated_ohlcv(ex, symbol, '1h', target_limit)
            
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['exchange'] = name
                all_dfs.append(df)
        except Exception as e:
            print(f"  ⚠️  {name} skipped: {str(e)[:50]}...")

    if not all_dfs:
        print(f"❌ FAILED: No data for {symbol}.")
        return

    # Blend and Clean
    master = pd.concat(all_dfs)
    agg_1h = master.groupby('timestamp').agg({
        'open': 'median', 'high': 'max', 'low': 'min', 'close': 'median', 'volume': 'sum'
    }).sort_index()

    agg_1h['datetime'] = pd.to_datetime(agg_1h.index, unit='ms', utc=True)
    agg_1h.set_index('datetime', inplace=True)
    
    clean_name = symbol.replace('/', '-')
    agg_1h.to_csv(os.path.join(DATA_DIR, f"{clean_name}-1h.csv"))
    print(f"✅ DEEP TRUTH SAVED: {len(agg_1h)} bars.")

    # Resampling... (same logic as before)
    timeframes = {'4h': '4h', '1d': '1d', '1w': '1w'}
    for label, rule in timeframes.items():
        resample_rule = rule.upper() if label != '1w' else 'W-SUN' 
        resampled = agg_1h.resample(resample_rule, closed='left', label='left').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        resampled.to_csv(os.path.join(DATA_DIR, f"{clean_name}-{label}.csv"))
        print(f"  ✨ Resampled {label}: {len(resampled)} bars.")

if __name__ == "__main__":
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD", "DOGE/USD"]
    for s in symbols:
        get_super_data(s)
