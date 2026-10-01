import os
import ccxt
import pandas as pd
from datetime import datetime, timezone
import time

# --- Configuration ---
EXCHANGE = 'coinbase' 
# 🎯 ALL COUNCIL SYMBOLS
SYMBOLS = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'ADA/USD', 'DOGE/USD', 'SUI/USD', 'PEPE/USD']
TIMEFRAME = '1h'
START_DATE = '2023-01-01' # 💡 Pro-tip: 2023 is enough for 200 SMA/Indicators and much faster.
OUTPUT_FOLDER = 'data'

def fetch_historical_data(symbol, timeframe, start_date):
    print(f"\n🚀 Downloading {symbol}...")
    exchange = getattr(ccxt, EXCHANGE)()
    since = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
    all_candles = []
    
    while True:
        try:
            # Coinbase specific limit is often 300 per call
            new_candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=300)
            if not new_candles:
                break
            
            all_candles.extend(new_candles)
            since = new_candles[-1][0] + 1
            
            # Progress print
            last_dt = datetime.fromtimestamp(new_candles[-1][0] / 1000, tz=timezone.utc)
            print(f"   ∟ Fetched up to {last_dt.strftime('%Y-%m-%d %H:%M')}. Total: {len(all_candles)}", end='\r')
            
            time.sleep(exchange.rateLimit / 1000) 
            
            # Stop if we've reached "Now"
            if new_candles[-1][0] >= int(time.time() * 1000) - 3600000:
                break
                
        except Exception as e:
            print(f"\n   ❌ Error: {e}. Retrying...")
            time.sleep(10)
            
    return all_candles

def save_to_csv(candles, symbol):
    if not candles: return
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # Standardizing naming: BTC/USD -> BTC-USD-1h.csv
    filename = f"{symbol.replace('/', '-')}-{TIMEFRAME}.csv"
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"\n💾 Saved to {filepath}")

if __name__ == "__main__":
    for sym in SYMBOLS:
        data = fetch_historical_data(sym, TIMEFRAME, START_DATE)
        save_to_csv(data, sym)
    print("\n✅ ALL COUNCIL DATA UPDATED.")
