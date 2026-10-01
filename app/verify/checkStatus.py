import requests
import json

# Configuration
BASE_URL = "https://neov6backend.onrender.com/api/backtest/status"
TOKEN = "YOUR_JWT_TOKEN_HERE" # Get this from your browser's LocalStorage/Network tab
USER_ID = "68b33a9a00093db11e60295f"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

def test_endpoint(params):
    print(f"🔍 Testing with params: {params}")
    try:
        response = requests.get(BASE_URL, headers=headers, params=params)
        print(f"📡 Status: {response.status_code}")
        if response.status_code == 200:
            print(f"✅ SUCCESS! Response: {response.json()}")
        else:
            print(f"❌ FAILED: {response.text}")
    except Exception as e:
        print(f"💥 Error: {e}")
    print("-" * 30)

# 🧪 Test Case 1: Just userId (Most likely requirement)
test_endpoint({"userId": USER_ID})

# 🧪 Test Case 2: Just user_id (Snake case check)
test_endpoint({"user_id": USER_ID})

# 🧪 Test Case 3: Empty (What your React app is doing now)
test_endpoint({})

# 🧪 Test Case 4: symbol and timeframe (Some engines use this as a key)
test_endpoint({"userId": USER_ID, "symbol": "SOL-USD", "timeframe": "1h"})
