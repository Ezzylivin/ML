import requests
import math

# 🟢 CONFIG
WALLET_ID = "0x1589DB9ef013Bb3394089191E5b76E49af1AacB4"
PYTHON_URL = f"http://localhost:8000/api/bot/status?userId={WALLET_ID}"

def check_for_nan(obj, path=""):
    """Finds NaN values that crash MongoDB and cause the 'Offline' glitch."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.extend(check_for_nan(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(check_for_nan(v, f"{path}[{i}]"))
    elif isinstance(obj, float):
        if math.isnan(obj):
            found.append(f"❌ NaN found at: {path}")
    return found

try:
    print(f"📡 Checking main3.py state...")
    data = requests.get(PYTHON_URL).json()
    
    issues = check_for_nan(data)
    if issues:
        print(f"\n🚨 FOUND {len(issues)} SERIALIZATION ERRORS:")
        for issue in issues[:5]: print(issue)
        print("\n👉 FIX: Add '.fillna(0)' to your indicators in main3.py.")
    else:
        print("✅ Data is clean of NaNs. The issue is likely the MongoDB URI string in your Backend.")
except Exception as e:
    print(f"💥 Failed to connect to main3.py: {e}")
