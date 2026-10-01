import pandas as pd
import pandas_ta as ta
import os
import traceback
import numpy as np

def debug_backtest_data(symbol="BTC-USD", timeframe="1h", start_date="2025-08-21"):
    print(f"🔍 STARTING ENGINE AUDIT v8.8 [HARDENED] | {symbol}")
    
    try:
        file_path = f"data/{symbol}-{timeframe}.csv"
        df = pd.read_csv(file_path)
        
        # --- 🟢 THE FIX: SCHEMA PURGE ---
        # Force columns to lowercase and remove any existing logic/indicator columns
        df.columns = [c.strip().lower() for c in df.columns]
        to_drop = ['rsi', 'atr', 'adx', 'sma_200', 'sma_logic', 'adx_logic', 'atr_logic']
        df = df.drop(columns=[c for c in to_drop if c in df.columns])
        print(f"🧹 Schema Purged. Raw columns: {list(df.columns)}")

        # --- 🟢 TIME ALIGNMENT (No inplace) ---
        time_options = ['datetime', 'time', 'timestamp']
        found_time = next((c for c in time_options if c in df.columns), df.columns[0])
        df['time'] = pd.to_datetime(df[found_time], utc=True)
        df = df.set_index('time', drop=False)
        df = df.sort_index()

        # --- 🟢 INDICATOR CALCULATION ---
        print("🛠️ Calculating Indicators via Pandas-TA...")
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.sma(length=200, append=True)
        
        # Consolidate memory to prevent fragmentation errors
        df = df.copy() 
        df.columns = [c.lower() for c in df.columns]

        # --- 🟢 THE "VALUE" FIX ---
        # Using .values ensures we are moving raw data arrays, not indexed Series
        print("⚖️ Mapping Logic Pointers via .values...")
        
        # We look for the specific names pandas-ta generates
        df['sma_logic'] = df['sma_200'].values if 'sma_200' in df.columns else np.zeros(len(df))
        df['adx_logic'] = df['adx_14'].values if 'adx_14' in df.columns else np.zeros(len(df))
        df['atr_logic'] = df['atr_14'].values if 'atr_14' in df.columns else np.zeros(len(df))
        
        df = df.fillna(0)
        
        print(f"✅ SUCCESS: Logic columns aligned. Length: {len(df)}")
        print(f"📊 Sample SMA Logic: {df['sma_logic'].tail(1).values[0]}")

    except Exception as e:
        print(f"🔥 CRASH DETECTED:")
        print(traceback.format_exc())

if __name__ == "__main__":
    debug_backtest_data()
