import requests
import json

# 🟢 CONFIG
PYTHON_URL = "http://74.208.28.77:8000"  # Your VPS IP
USER_ID = "0x1589DB9ef013Bb3394089191E5b76E49af1AacB4"  # Your Wallet

def run_debug():
    print(f"\n🔍 --- DIAGNOSTIC START: Checking Python Engine at {PYTHON_URL} ---\n")

    try:
        # 1. Check Status
        print("1️⃣  Pinging Bot Status...")
        url = f"{PYTHON_URL}/api/bot/status"
        params = {"userId": USER_ID}
        
        response = requests.get(url, params=params, timeout=5)
        
        print(f"   ✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   🤖 Bot Status: {data.get('status')}")
            print(f"   💰 Balance: {data.get('currentBalance')}")
            
            candles = data.get('candles', [])
            equity = data.get('equityCurve', [])
            
            print(f"   🕯️  Candles Count: {len(candles)}")
            print(f"   📈 Equity Points: {len(equity)}")

            if len(candles) > 0:
                print("\n   ✅ SAMPLE CANDLE DATA (First Item):")
                print(json.dumps(candles[0], indent=2))
            else:
                print("\n   ❌ CRITICAL FAILURE: Python returned 0 candles.")
                print("   👉 Check 'main.py' data loading logic or CCXT keys.")
        else:
            print(f"   ❌ Server Error: {response.text}")

    except Exception as e:
        print(f"\n   ❌ CONNECTION FAILED: {str(e)}")

    print("\n🔍 --- DIAGNOSTIC COMPLETE ---")

if __name__ == "__main__":
    run_debug()
