# File: download_data.py
import os
import ccxt
import pandas as pd
from datetime import datetime, timezone
import time

# --- Configuration ---
EXCHANGE = 'coinbase' 
SYMBOL = 'BTC/USD'
TIMEFRAME = '1h'
START_DATE = '2017-01-01'
OUTPUT_FOLDER = 'data'

def fetch_all_historical_data(symbol, timeframe, start_date):
    print(f"Starting data download for {symbol} on {timeframe}...")
    exchange = getattr(ccxt, EXCHANGE)()
    since_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
    all_candles = []
    limit = 1000 
    while True:
        try:
            new_candles = exchange.fetchOHLCV(symbol, timeframe, since=since_timestamp, limit=limit)
            if len(new_candles) > 0:
                all_candles.extend(new_candles)
                since_timestamp = new_candles[-1][0] + 1
                last_date = datetime.fromtimestamp(new_candles[-1][0] / 1000).strftime('%Y-%m-%d')
                print(f"Fetched {len(new_candles)} candles. Current date: {last_date}. Total: {len(all_candles)}")
                time.sleep(exchange.rateLimit / 1000) 
            else:
                break
        except Exception as e:
            print(f"An error occurred: {e}. Retrying in 30 seconds...")
            time.sleep(30)
    print(f"\n✅ Download complete. Total candles: {len(all_candles)}")
    return all_candles

def save_data_to_csv(candles, folder, filename):
    if not candles: return
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('datetime', inplace=True)
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    df.to_csv(filepath)
    print(f"💾 Data successfully saved to {filepath}")

if __name__ == "__main__":
    candles = fetch_all_historical_data(SYMBOL, TIMEFRAME, START_DATE)
    filename = f"{SYMBOL.replace('/', '-')}-{TIMEFRAME}.csv"
    save_data_to_csv(candles, OUTPUT_FOLDER, filename)
