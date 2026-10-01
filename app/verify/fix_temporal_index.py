import pandas as pd
import os

def debug_and_fix_index(file_path, start_date_str):
    print(f"🔍 Analyzing: {file_path}")
    if not os.path.exists(file_path):
        print("❌ Error: File not found.")
        return

    df = pd.read_csv(file_path)
    print(f"📊 Initial Index Type: {type(df.index)}")
    print(f"📝 Columns found: {list(df.columns)}")

    # 1. Find the date column
    time_candidates = ['datetime', 'time', 'timestamp', 'date']
    time_col = next((c for c in time_candidates if c.lower() in [col.lower() for col in df.columns]), None)

    if not time_col:
        print("❌ Error: No temporal column found. Pandas is defaulting to Integers.")
        return

    # 2. Convert and Set Index
    try:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df.set_index(time_col, inplace=True)
        df.sort_index(inplace=True)
        print(f"✅ Index successfully converted to: {type(df.index)}")
    except Exception as e:
        print(f"❌ Conversion Failed: {e}")
        return

    # 3. Test the Slice
    try:
        start_ts = pd.to_datetime(start_date_str).tz_localize('UTC') if 'UTC' not in start_date_str else pd.to_datetime(start_date_str)
        print(f"🧪 Testing slice with Timestamp: {start_ts}")
        
        test_slice = df.loc[start_ts:]
        print(f"✨ Success! Slice returned {len(test_slice)} rows.")
    except TypeError as e:
        print(f"💀 SLICE FAILED: {e}")
        print("💡 Cause: The index is still containing integers or non-datetime objects.")

if __name__ == "__main__":
    # Update these paths to match your environment
    target_csv = "data/BTC-USD-1h.csv"
    test_start = "2025-08-21"
    debug_and_fix_index(target_csv, test_start)
