import os
import sys
import pandas as pd
import ccxt
import time
from datetime import datetime

# Path Injector
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# 🟢 CONFIGURATION: Switched to Coinbase to bypass regional blocks
EXCHANGE_ID = 'coinbase' 
exchange = getattr(ccxt, EXCHANGE_ID)({'enableRateLimit': True})

def patch_file(filename):
    path = os.path.join(project_root, "data", filename)
    print(f"\n🩺  SURGERY ON: {filename} (Source: COINBASE)")
    
    # 1. Load Data
    try:
        df = pd.read_csv(path)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], utc=True)
        df = df.sort_values(df.columns[0]).reset_index(drop=True)
    except Exception as e:
        print(f" ❌ Skipping {filename}: {e}")
        return

    # 2. Detect Timeframe
    # NOTE: Coinbase natively supports 1m, 5m, 15m, 1h, 6h, 1d. 
    # CCXT will emulate 4h and 1w if needed, but results may vary.
    timeframe_map = {"30m": 30, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}
    tf = next((k for k in timeframe_map.keys() if f"-{k}.csv" in filename), "1h")
    interval_ms = timeframe_map[tf] * 60 * 1000
    
    # 3. Find Gaps
    diffs = df.iloc[:, 0].diff().dt.total_seconds() / 60
    gap_indices = df.index[diffs > (timeframe_map[tf] * 1.1)].tolist()
    
    if not gap_indices:
        print(" ✅  No gaps found. File is healthy.")
        return

    print(f" 🔍  Found {len(gap_indices)} gaps. Patching via Coinbase API...")
    
    # Normalize Symbol (BTC-USD -> BTC/USD)
    asset = filename.split("-USD")[0]
    symbol = f"{asset}/USD"
    
    new_rows = []
    for idx in gap_indices:
        # Start fetching from the missing timestamp
        start_ts = int(df.iloc[idx-1, 0].timestamp() * 1000) + interval_ms
        end_ts = int(df.iloc[idx, 0].timestamp() * 1000)
        
        while start_ts < end_ts:
            try:
                print(f"   -> Fetching {tf} batch from {datetime.fromtimestamp(start_ts/1000)}")
                # Coinbase limit is usually 300 candles per request
                ohlcv = exchange.fetch_ohlcv(symbol, tf, since=start_ts, limit=300)
                if not ohlcv: break
                
                for bar in ohlcv:
                    if bar[0] >= end_ts: break
                    new_rows.append(bar)
                
                start_ts = ohlcv[-1][0] + interval_ms
                time.sleep(exchange.rateLimit / 1000)
            except Exception as e:
                print(f" ❌ Coinbase Error: {str(e)[:100]}")
                break

    # 4. Merge and Heal
    if new_rows:
        patch_df = pd.DataFrame(new_rows, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        patch_df['time'] = pd.to_datetime(patch_df['time'], unit='ms', utc=True)
        
        # Combine old and new, ensuring OHLC columns match original CSV case
        final_df = pd.concat([df, patch_df]).drop_duplicates(subset=[df.columns[0]])
        final_df = final_df.sort_values(df.columns[0])
        
        final_df.to_csv(path, index=False)
        print(f" 🎉  SUCCESS: {len(new_rows)} bars stitched into {filename}")
    else:
        print(f" ⚠️  Patch failed. Some older coins or specific timeframes might lack Coinbase history.")

if __name__ == "__main__":
    data_dir = os.path.join(project_root, "data")
    for f in os.listdir(data_dir):
        if f.endswith(".csv"):
            patch_file(f)
