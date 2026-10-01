import requests
import time
import sys
import json

# 🟢 CONFIGURATION
# Ensure this matches your running Uvicorn port (default 8000)
BASE_URL = "http://localhost:8000"
USER_ID = "debug_admin_01"

def start_bot():
    """Manually starts a bot with a known valid configuration."""
    print(f"🚀 Initializing Debug Bot for user: {USER_ID}...")
    
    # 🟢 PAYLOAD STRUCTURE (Matches your Pydantic Model)
    payload = {
        "userId": USER_ID,
        "config": {
            "symbol": "BTC-USD",
            "timeframe": "1h",
            "capitalAllocation": 500.0, # Using $500 to test balance logic
            "strategies": [
                {
                    "code": "rsi_threshold",
                    "params": {"rsi_length": 14, "oversold": 30}
                }
            ],
            "comboConfig": {
                "strategyCodes": ["rsi_threshold"],
                "combinationRule": "AND"
            }
        }
    }

    try:
        response = requests.post(f"{BASE_URL}/api/bot/start", json=payload)
        if response.status_code == 200:
            print("✅ Bot Start Command Sent (200 OK)")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Error Starting Bot: {response.status_code}")
            print(f"   Reason: {response.text}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        print("   Is uvicorn running? (Try: uvicorn main:app --reload)")
        sys.exit(1)

def monitor_stream():
    """Polls the status endpoint every 5 seconds to visualize the stream."""
    print("\n📡 Connected to Neural Stream. Waiting for logic updates...")
    print("   (Press Ctrl+C to stop debugging)\n")
    
    seen_logs = set()
    
    try:
        while True:
            try:
                # Poll Status
                res = requests.get(f"{BASE_URL}/api/bot/status", params={"userId": USER_ID})
                data = res.json()
                
                # Check Balance
                balance = data.get("currentBalance", 0)
                status = data.get("status", "unknown")
                logs = data.get("logs", [])

                # Print System Heartbeat
                sys.stdout.write(f"\r💓 Status: {status.upper()} | Balance: ${balance:.2f} | Log Count: {len(logs)}   ")
                sys.stdout.flush()

                # Print NEW Logs
                if logs:
                    # Logs are usually newest-first. Let's check the top 3.
                    for log in reversed(logs[:5]): 
                        if log not in seen_logs:
                            print(f"\n   > {log}")
                            seen_logs.add(log)
                            
                # Check for "Stuck" state
                if status == "stopped":
                    print("\n\n⚠️ Bot has stopped unexpectedly!")
                    break
                    
            except Exception as e:
                print(f"\n❌ Stream Error: {e}")
            
            time.sleep(5) # Poll every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\n🛑 Debugging stopped.")

if __name__ == "__main__":
    start_bot()
    # Give the background task a moment to fetch initial data
    print("⏳ Waiting for data fetch (3 seconds)...")
    time.sleep(3)
    monitor_stream()
