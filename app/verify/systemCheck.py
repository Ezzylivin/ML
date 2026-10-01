# File: neo_diag.py (Standalone Diagnostic Tool)
import math
import requests
import pymongo
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure

# --- 🟢 CONFIGURATION (Update with your actual values) ---
MONGO_URI = "mongodb+srv://ericjdickerson93:Dadeadend1!!@cluster0.wzztvhe.mongodb.net/?retryWrites=true&w=majority"
WALLET_ID = "0x1589DB9ef013Bb3394089191E5b76E49af1AacB4"
# Point to your local FastAPI engine running on this server
FASTAPI_URL = "http://localhost:8000"

def run_diagnostics():
    print("🚀 STARTING STANDALONE SYSTEM DIAGNOSTIC (ML SIDE)...\n")

    # 1. TEST MONGODB CONNECTION
    print("Step 1: Testing MongoDB Atlas Connection...")
    try:
        # Use a short timeout to prevent hanging
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # The ismaster/hello command is a cheap way to verify connectivity
        client.admin.command('hello') 
        print("✅ SUCCESS: Connected to MongoDB Atlas.\n")
    except (ServerSelectionTimeoutError, ConnectionFailure) as e:
        print(f"❌ FAIL: Could not connect to MongoDB. Error: {e}")
        print("👉 Check: Is your IP whitelisted in Atlas? Is the URI string correct?\n")

    # 2. TEST FASTAPI ENGINE STATUS
    print(f"Step 2: Testing FastAPI Engine on {FASTAPI_URL}...")
    try:
        # Pinging the status endpoint of your running bot
        status_endpoint = f"{FASTAPI_URL}/api/bot/status?userId={WALLET_ID}"
        response = requests.get(status_endpoint, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS: FastAPI is reachable. Bot Status: {data.get('status')}")
            
            # 3. SCAN FOR SERIALIZATION ERRORS (NaNs)
            print("Step 3: Scanning for NaN values (MongoDB's 'Offline' Killer)...")
            illegal_values = scan_for_nans(data)
            if illegal_values:
                print(f"🚨 FOUND {len(illegal_values)} NaN VALUES:")
                for err in illegal_values[:5]: print(f"   - {err}")
                print("👉 FIX: Add '.fillna(0)' to indicators in main3.py.\n")
            else:
                print("✅ SUCCESS: No NaN values found in data stream.\n")
        else:
            print(f"❌ FAIL: FastAPI returned status {response.status_code}\n")
    except Exception as e:
        print(f"❌ FAIL: Could not reach FastAPI engine. Is main3.py running? Error: {e}\n")

    print("🏁 DIAGNOSTIC COMPLETE.")

def scan_for_nans(obj, path="root"):
    """Recursively checks for NaN values which MongoDB cannot serialize."""
    errors = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            errors.extend(scan_for_nans(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            errors.extend(scan_for_nans(v, f"{path}[{i}]"))
    elif isinstance(obj, float):
        if math.isnan(obj):
            errors.append(f"NaN at {path}")
    return errors

if __name__ == "__main__":
    run_diagnostics()
