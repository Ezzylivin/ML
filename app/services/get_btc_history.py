import ccxt
import pandas as pd
import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BTC_Fetcher")

# 🟢 CONFIG
SYMBOL = "BTC/USD"
TIMEFRAME = "1h"
START_DATE = "2024-01-01T00:00:00Z" 

def fetch_history():
    ex = ccxt.binanceus()
    clean_sym = SYMBOL.replace("/", "-")
    filepath = f"data/{clean_sym}-{TIMEFRAME}.csv"
    
    logger.info(f"🕰️  Fetching BTC history from {START_DATE}...")
    
    since = ex.parse8601(START_DATE)
    all_candles = []
    
    while True:
        try:
            # Fetch batch
            candles = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
            if not candles: break
            
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            
            # Progress update
            last_date = pd.to_datetime(candles[-1][0], unit='ms')
            print(f"   Downloaded up to {last_date}...", end="\r")
            
            # Stop if we reach 2026 (we have enough)
            if last_date.year >= 2026: break
            
            time.sleep(0.1) # Respect rate limits
        except Exception as e:
            logger.error(f"Error: {e}")
            break
            
    if not all_candles:
        logger.error("No data downloaded.")
        return

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Save to disk
    df.to_csv(filepath, index=False)
    logger.info(f"\n✅ BTC HISTORY SAVED: {filepath}")
    logger.info(f"   Range: {df['datetime'].min()} <--> {df['datetime'].max()}")

if __name__ == "__main__":
    fetch_history()
