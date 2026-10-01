import requests
import json

USER_ID = "0x1589DB9ef013Bb3394089191E5b76E49af1AacB4"
BASE_URL = "http://127.0.0.1:8000/api/bot"

def test_halt_flow():
    print(f"--- Starting Halt Flow Debug ---")
    
    # 1. Check current status
    print(f"Checking status for {USER_ID}...")
    res = requests.get(f"{BASE_URL}/status?userId={USER_ID}")
    print(f"Current Status: {res.json().get('status')}")
    
    # 2. Send Stop Command
    print("\nSending STOP command...")
    stop_res = requests.post(f"{BASE_URL}/stop", json={"userId": USER_ID})
    print(f"Stop Response: {stop_res.status_code} - {stop_res.json()}")
    
    # 3. Verification Poll
    print("\nVerifying if stop persisted...")
    verify = requests.get(f"{BASE_URL}/status?userId={USER_ID}")
    print(f"Post-Stop Status: {verify.json().get('status')}")
    
    if verify.json().get('status') == 'stopped':
        print("\n✅ SUCCESS: Bot successfully killed and reset.")
    else:
        print("\n❌ FAILURE: Bot is still reporting as running.")

if __name__ == "__main__":
    test_halt_flow()
