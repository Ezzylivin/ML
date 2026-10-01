import requests

URL = "https://neov6backend.onrender.com/api/bot/start"
# 🟢 Replace this with your actual token from the dashboard
TOKEN = "4b752a90afa23849c6b9fff4bc0666d9dc0f00b6ce57fe3626f0ba47b229c116" 

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "userId": "0x1589DB9ef013Bb3394089191E5b76E49af1AacB4",
    "config": {
        "symbol": "BTC-USD",
        "capitalAllocation": 125,
        "strategies": [{"code": "rsi_threshold", "params": {"rsi_length": 14}}]
    }
}

print(f"🚀 Sending authenticated test request...")
try:
    response = requests.post(URL, json=payload, headers=headers, timeout=10)
    print(f"📡 Status Code: {response.status_code}")
    print(f"📦 Body: {response.json()}")
except Exception as e:
    print(f"❌ Failed: {e}")
