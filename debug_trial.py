import requests
import json
import os

# Configuration
ML_SERVER_URL = "http://74.208.28.77:8000"

# This is the exact config from your failed Trial 2557
debug_config = {
    "symbol": "BTC-USD",
    "timeframe": "1d",
    "startDate": "2023-01-01", # Adjusted to a safe recent date
    "endDate": "2023-12-31",
    "initialBalance": 1000,
    "fee": 0.006,
    "mlMode": "predictions",
    "mlModel": "btc_1d_randomforest_model", # The model from the log
    "mlThreshold": 0.65,
    "strategies": [
        {"code": "psar_signal", "params": {}},
        {"code": "rsi_divergence", "params": {"rsi_length": 20}}
    ],
    "params": {
        "hybridMode": "REGIME",
        "regime_threshold": 30,
        "minAdxLevel": 0,
        "tslAtrMult": 2.5,
        "trendFilterPeriod": 200
    }
}

def debug_run():
    print(f"🚀 Sending Debug Request to {ML_SERVER_URL}...")
    try:
        url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
        res = requests.post(url, json=debug_config, timeout=30)
        
        print(f"\n📡 Status Code: {res.status_code}")
        
        if res.status_code == 200:
            data = res.json()
            metrics = data.get("combinedResult", {}).get("metrics", {})
            print(f"✅ Success! Trades: {metrics.get('totalTrades')}")
            print(f"   Return: {metrics.get('totalReturn')}%")
        else:
            print(f"❌ Server Error:")
            print(res.text) # This will print the Python Stack Trace
            
    except Exception as e:
        print(f"🔥 Connection Failed: {e}")

if __name__ == "__main__":
    debug_run()
