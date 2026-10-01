import requests
import json

# Configuration
ML_SERVER_URL = "http://74.208.28.77:8000"

# A simple, standard configuration that should produce trades
payload = {
    "symbol": "BTC-USD",
    "timeframe": "1d",
    "startDate": "2023-01-01",
    "endDate": "2023-12-31",
    "initialBalance": 1000,
    "fee": 0.006,
    "mlMode": "off", # Turn ML OFF first to test core logic
    "strategies": [
        {"code": "sma_crossover", "params": {"sma_fast_period": 10, "sma_slow_period": 50}},
        {"code": "rsi_divergence", "params": {"rsi_length": 14, "oversold_level": 30, "overbought_level": 70}}
    ],
    "params": {
        "hybridMode": "REGIME",
        "regime_threshold": 20,
        "minAdxLevel": 0 # Disable ADX filter
    }
}

print(f"🚀 Sending Diagnostic Request to {ML_SERVER_URL}...")
try:
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
    response = requests.post(url, json=payload, timeout=30)

    print(f"\n📡 HTTP Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        result = data.get("combinedResult", {})
        metrics = result.get("metrics", {})
        print("\n✅ SUCCESS! Response received.")
        print(f"   Total Trades: {metrics.get('totalTrades')}")
        print(f"   Total Return: {metrics.get('totalReturn')}%")
        print(f"   Final Balance: {metrics.get('finalBalance')}")

        if metrics.get('totalTrades') == 0:
            print("\n⚠️ WARNING: 0 Trades executed. Logic issue likely.")
    else:
        print(f"❌ SERVER ERROR!")
        print("-" * 40)
        print(response.text)
        print("-" * 40)

except Exception as e:
    print(f"🔥 Connection Failed: {e}")
