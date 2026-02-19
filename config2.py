import os
import warnings
import threading
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load .env variables (if you have them)
load_dotenv()

# Filter noise from libraries
warnings.filterwarnings("ignore")

# --- 1. MODULAR DIRECTORY ARCHITECTURE ---

# Get absolute pathing for the root and app directories
CURRENT_FILE = os.path.abspath(__file__)
APP_DIR = os.path.dirname(CURRENT_FILE)  # /root/Project/ML/app
PROJECT_ROOT = os.path.dirname(APP_DIR)   # /root/Project/ML/

# Core Data & Script Directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PREDICTORS_DIR = os.path.join(PROJECT_ROOT, "predictors")
AGENTS_DIR = os.path.join(PROJECT_ROOT, "agents")
VALIDATION_DIR = os.path.join(APP_DIR, "discovery") # Pointing to discovery for now

# Output & Result Paths
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
OPTIMIZER_RESULTS_DIR = os.path.join(DATA_DIR, "optimizer_results")

# --- 🟢 CRITICAL FIX: Variable Aliasing for API3 Compatibility ---
# This ensures every script finds the same "Brain Storage"
MODEL_DIR = os.path.join(DATA_DIR, "saved_models")
MODEL_STORAGE_DIR = "/root/Project/ML/app/models"
RESULTS_DIR = OPTIMIZER_RESULTS_DIR # Alias for API3
OPTIMIZER_DIR = OPTIMIZER_RESULTS_DIR # Alias for Main Optimizer

# Auto-Create structure if missing
REQUIRED_PATHS = [
    DATA_DIR, MODEL_DIR, LOG_DIR, 
    OPTIMIZER_RESULTS_DIR, PREDICTORS_DIR, AGENTS_DIR
]
for path in REQUIRED_PATHS:
    os.makedirs(path, exist_ok=True)


# --- 2. ADVANCED ML SETTINGS (NEO-V7) ---

ML_CONFIG = {
    "DEFAULT_LONG_THRESHOLD": 0.65,   # Confidence required to go Long
    "DEFAULT_SHORT_THRESHOLD": 0.35,  # Confidence required to go Short
    "DEFAULT_EXIT_THRESHOLD": 0.50,   # Point where bot flips neutral
    "DEFAULT_LOOKBACK_WINDOW": 50,    # Number of candles fed to XGBoost
    "FEATURE_SET": "technical_v7",    # Version of feature engineering logic
    "USE_ONNX_INFERENCE": False       # Set to True if using ONNX speed boosts
}


# --- 3. DATABASE CONNECTION (Hardened) ---

# Replace the URI below with your actual production string if it changes
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://ericjdickerson93:Dadeadend1!!@cluster0.wzztvhe.mongodb.net/?retryWrites=true&w=majority")

class MemoryDB:
    """Fallback database if Atlas is unreachable"""
    def __init__(self):
        self.data = {}
        print("⚠️  Warning: Using In-Memory Mock Database")
    def find_one(self, query): return self.data.get(query.get("botId"))
    def find(self, query): return []
    def update_one(self, query, update, upsert=False): return None
    def insert_one(self, doc): return None
    def delete_many(self, query): return None 
    def update_many(self, query, update): return None

try:
    from pymongo import MongoClient
    # Use TLS for 2026 security standards
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tls=True)
    client.server_info() # Trigger connection check
    db = client["test"]
    bots_collection = db["bots"]
    print("✅ Connected to MongoDB Atlas")
    
    # Startup Cleanup: Clear stale positions
    try:
        bots_collection.update_many({}, {"$unset": {"currentPosition": ""}})
    except: pass
except Exception as e:
    print(f"❌ DB Connection Failed: {e}")
    bots_collection = MemoryDB()


# --- 4. PERFORMANCE & RISK SETTINGS ---

bots_lock = threading.Lock()
active_live_bots = {}
shutdown_event = threading.Event()

# Refined Fees for Institutional Coinbase Access
DEFAULT_TAKER_FEE = 0.0006  # 0.06% 
SLIPPAGE_BPS = 1.0          # 0.1% expected slippage

print(f"🚀 NEO-V7 Config Active | Fee: {DEFAULT_TAKER_FEE*100}% | ML Lookback: {ML_CONFIG['DEFAULT_LOOKBACK_WINDOW']}")
