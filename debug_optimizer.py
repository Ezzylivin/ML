import requests
import json
import random
import pandas as pd
from datetime import datetime

# --- CONFIG ---
ML_SERVER_URL = "http://74.208.28.77:8000"

# This payload mimics a complex Optimizer Trial
# Random parameters, ML turned ON, WFO date range
payload = {
    "symbol": "BTC-USD",
    "timeframe": "1h",
    "startDate": "2023-01-01",
    "endDate": "2023-12-31",
    "initialBalance": 300,
    "fee": 0.006,
    "mlMode": "predictions", 
    "mlModel": "btc_1h_mock_model", # We will update this dynamically below if possible
    "mlThreshold": 0.55,
    
    # Complex Strategy Combo (Trend + Range)
    "strategies": [
        {
            "code": "sma_crossover", 
            "params": {"sma_fast_period": 12, "sma_slow_period": 55}
        },
        {
            "code": "rsi_divergence", 
            "params": {"rsi_length": 21, "oversold_level": 35, "overbought_level": 65}
        }
    ],
    
    # Regime Filters
    "params": {
        "hybridMode": "REGIME",
        "regime_threshold": 25,
        "minAdxLevel": 15,
        "tslAtrMult": 3.0,
        "trendFilterPeriod": 200
    }
}

# 1. Fetch a real model name first
try:
    print("🔍 Fetching available models...")
    r = requests.get(f"{ML_SERVER_URL}/api/ml/available-models", timeout=5)
    models = r.json()
    if models:
        # Find a BTC model
        btc_models = [m['id'] for m in models if 'btc' in m['id'].lower()]
        if btc_models:
            payload['mlModel'] = btc_models[0]
            print(f"✅ Selected Model: {payload['mlModel']}")
        else:
            print("⚠️ No BTC model found, trying generic.")
except Exception as e:
    print(f"❌ Could not fetch models: {e}")

# 2. Run the X-Ray Request
print(f"\n🚀 Sending Optimization-Style Request to {ML_SERVER_URL}...")
print(f"   Params: {json.dumps(payload['strategies'], indent=2)}")

try:
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
    start_time = datetime.now()
    response = requests.post(url, json=payload, timeout=60)
    end_time = datetime.now()
    
    print(f"\n⏱️ Time taken: {(end_time - start_time).total_seconds()}s")
    print(f"📡 HTTP Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        
        # Check if the server actually returned a result structure
        if "combinedResult" in data:
            metrics = data["combinedResult"].get("metrics", {})
            trades = metrics.get("totalTrades", 0)
            ret = metrics.get("totalReturn", 0)
            
            print("\n✅ SERVER PROCESSED REQUEST")
            print(f"   Trades: {trades}")
            print(f"   Return: {ret}%")
            
            if trades == 0:
                print("\n⚠️ DIAGNOSIS: 0 TRADES.")
                print("   This means the ML Model or Strategy Logic vetoed everything.")
                print("   The Optimizer sees this as 'Failed' and moves on.")
            else:
                print("\n✅ DIAGNOSIS: SUCCESS.")
                print("   The system works. The -999s might be due to Timeouts or Bad Params.")
        else:
             print(f"⚠️ UNEXPECTED JSON STRUCTURE: {data.keys()}")
    else:
        print(f"\n❌ SERVER CRASHED OR REJECTED REQUEST")
        print("-" * 50)
        print(response.text)
        print("-" * 50)

except Exception as e:
    print(f"\n🔥 CLIENT CONNECTION FAILED: {e}")
