import requests
import json

URL = "http://74.208.28.77:8000"
USER_ID = "0x1589DB9ef013Bb3394089191E5b76E49af1AacB4"

def test_start():
    print(f"🚀 Attempting to START bot on {URL}...")
    
    payload = {
        "userId": USER_ID,
        "config": {
            "symbol": "BTC-USD",
            "timeframe": "1h",
            "capitalAllocation": 1000
        }
    }
    
    try:
        # 1. SEND START COMMAND
        res = requests.post(f"{URL}/api/bot/start", json=payload, timeout=10)
        print(f"   📡 Start Response: {res.status_code}")
        print(f"   📝 Message: {res.json()}")

        # 2. CHECK STATUS IMMEDIATELY
        print("\n🔍 Checking resulting status...")
        status_res = requests.get(f"{URL}/api/bot/status", params={"userId": USER_ID})
        data = status_res.json()
        
        candles = data.get("candles", [])
        print(f"   🕯️ Candles Loaded: {len(candles)}")
        
        if len(candles) > 0:
            print("   ✅ SUCCESS: First candle:", candles[0])
        else:
            print("   ❌ FAILURE: Bot started but has NO candles.")

    except Exception as e:
        print(f"   ❌ ERROR: {e}")

if __name__ == "__main__":
    test_start()
