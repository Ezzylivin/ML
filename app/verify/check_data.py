import pandas as pd
import os

def audit_datasets():
    data_dir = "data/"
    print("🔍 AUDITING DATA INTEGRITY (TIMEFRAME-AWARE)...")
    
    # Map filenames to expected hour gaps
    timeframe_map = {
        "30m": 0.5,
        "1h": 1.0,
        "4h": 4.0,
        "1d": 24.0,
        "1w": 168.0
    }

    for file in os.listdir(data_dir):
        if not file.endswith(".csv"): continue
        path = os.path.join(data_dir, file)
        
        # Detect timeframe from filename (e.g., BTC-USD-4h.csv -> 4h)
        detected_tf = "1h" # Default
        for tf in timeframe_map.keys():
            if f"-{tf}.csv" in file:
                detected_tf = tf
                break
        
        expected_gap = timeframe_map[detected_tf]
        
        try:
            df = pd.read_csv(path)
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], utc=True)
            
            # Check for missing columns
            required = {'open', 'high', 'low', 'close'}
            missing = required - {c.lower().strip() for c in df.columns}
            
            # Calculate gaps based on the SPECIFIC timeframe
            time_diffs = df.iloc[:, 0].diff().dt.total_seconds() / 3600
            # We allow a small 10% buffer for network jitter
            gaps = time_diffs[time_diffs > (expected_gap * 1.1)]
            
            status = "✅ PASS" if not missing and gaps.empty else "❌ FAIL"
            print(f"[{status}] {file:20} | TF: {detected_tf:4} | Gaps: {len(gaps)}")
            
        except Exception as e:
            print(f"❌ ERROR reading {file}: {e}")

if __name__ == "__main__":
    audit_datasets()
