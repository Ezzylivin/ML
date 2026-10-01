import sys
import os

# Set path so we simulate running from root
sys.path.append(os.getcwd())

print("1. Testing Config import...")
from app.config import bots_collection
print("   ✅ Config Loaded.")

print("2. Testing Utils import...")
from app.utils import clean_float
print("   ✅ Utils Loaded.")

print("3. Testing Logging import...")
from app.logging_handlers import MongoLogHandler
print("   ✅ Logging Loaded.")

print("4. Testing Bot Class import...")
try:
    from app.live_bot import LiveExecutiveBot
    print("   ✅ LiveExecutiveBot Imported Successfully!")
except ImportError as e:
    print(f"   ❌ IMPORT FAILED: {e}")
except Exception as e:
    print(f"   ❌ CRASHED: {e}")

print("5. Testing API import...")
try:
    from app.api2 import app
    print("   ✅ API Loaded Successfully!")
except Exception as e:
    print(f"   ❌ API FAILED: {e}")
