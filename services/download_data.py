import os
import ccxt
import pandas as pd
from datetime import datetime, timezone
import time

# --- Configuration ---
EXCHANGE = 'coinbase' 
SYMBOLS = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'ADA/USD', 'DOGE/USD', 'SUI/USD', 'PEPE/USD']
TIMEFRAME = '1h'
DEFAULT_START = '2023-01-01' 
OUTPUT_FOLDER = 'data'

def get_last_timestamp(filepath):
    """Checks the existing file to find where we left off."""
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        if df.empty: return None
        # Assumes 'timestamp' column exists from previous runs
        return int(df['timestamp'].max())
    except Exception:
        return None

def fetch_and_merge(symbol):
    filename = f"{symbol.replace('/', '-')}-{TIMEFRAME}.csv"
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    last_ts = get_last_timestamp(filepath)
    
    exchange = getattr(ccxt, EXCHANGE)()
    
    if last_ts:
        since = last_ts + 1  # Start 1ms after the last record
        print(f"🔄 Updating {symbol}: Resuming from {datetime.fromtimestamp(since/1000, tz=timezone.utc)}")
    else:
        since = int(datetime.strptime(DEFAULT_START, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
        print(f"🚀 Initializing {symbol}: Starting from {DEFAULT_START}")

    all_new_candles = []
    while True:
        try:
            new_candles = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since, limit=300)
            if not new_candles: break
            
            all_new_candles.extend(new_candles)
            since = new_candles[-1][0] + 1
            
            if new_candles[-1][0] >= int(time.time() * 1000) - 3600000:
                break # We are caught up to the current hour
                
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"   ❌ Error: {e}. Retrying...")
            time.sleep(5)
            break

    if all_new_candles:
        new_df = pd.DataFrame(all_new_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if last_ts:
            old_df = pd.read_csv(filepath)
            final_df = pd.concat([old_df, new_df]).drop_duplicates(subset=['timestamp'])
        else:
            final_df = new_df
            
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        final_df.to_csv(filepath, index=False)
        print(f"   ✅ Merged {len(all_new_candles)} new candles into {filename}")
    else:
        print(f"   ✨ {symbol} is already up to date.")

if __name__ == "__main__":
    for sym in SYMBOLS:
        fetch_and_merge(sym)
    print("\n🏁 ALL SYSTEMS SYNCHRONIZED.")
