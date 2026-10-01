import requests
import json

# Configuration
ML_SERVER_URL = "http://74.208.28.77:8000"

# 🚀 TEST CONFIGURATION: ML IS ON
payload = {
    "symbol": "BTC-USD",
    "timeframe": "1d",
    "startDate": "2023-01-01",
    "endDate": "2023-12-31",
    "initialBalance": 1000,
    "fee": 0.006,
    
    # ⚠️ ACTIVATING THE SUSPECT
    "mlMode": "predictions", 
    # We use a model name seen in your logs previously. 
    # If this specific file doesn't exist, the server *should* handle it gracefully.
    "mlModel": "btc_1d_lightgbm_model", 
    "mlThreshold": 0.60,
    
    "strategies": [
        {"code": "sma_crossover", "params": {"sma_fast_period": 10, "sma_slow_period": 50}},
        {"code": "rsi_divergence", "params": {"rsi_length": 14, "oversold_level": 30, "overbought_level": 70}}
    ],
    "params": {
        "hybridMode": "REGIME",
        "regime_threshold": 20,
        "minAdxLevel": 0
    }
}

print(f"🚀 Sending ML STRESS TEST to {ML_SERVER_URL}...")
try:
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
    response = requests.post(url, json=payload, timeout=30)
    
    print(f"\n📡 HTTP Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        result = data.get("combinedResult", {})
        metrics = result.get("metrics", {})
        print("\n✅ ML INTEGRATION SUCCESS!")
        print(f"   Total Trades: {metrics.get('totalTrades')}")
        print(f"   Total Return: {metrics.get('totalReturn')}%")
        print("   (If you see this, the Optimizer should work now)")
    else:
        print(f"❌ ML CRASHED THE SERVER!")
        print("-" * 40)
        print(response.text)
        print("-" * 40)

except Exception as e:
    print(f"🔥 Connection Failed: {e}")
