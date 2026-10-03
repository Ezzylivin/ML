import os
import warnings
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Filter noise from libraries
warnings.filterwarnings("ignore")

# --- 1. MODULAR DIRECTORY ARCHITECTURE ---
CURRENT_FILE = os.path.abspath(__file__)
APP_DIR = os.path.dirname(CURRENT_FILE)       # /root/Project/ML/app
PROJECT_ROOT = os.path.dirname(APP_DIR)        # /root/Project/ML/

# Core Data & Script Directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PREDICTORS_DIR = os.path.join(PROJECT_ROOT, "predictors")
AGENTS_DIR = os.path.join(PROJECT_ROOT, "agents")
VALIDATION_DIR = os.path.join(APP_DIR, "discovery")

# Output & Result Paths
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
OPTIMIZER_RESULTS_DIR = os.path.join(DATA_DIR, "optimizer_results")

# ============================================================
# 🔧 FIX #1: SINGLE MODEL DIRECTORY (Was 4 different paths)
# ============================================================
# Every script that saves or loads models MUST use this one path.
# Old mess:
#   - config2: MODEL_DIR -> data/saved_models
#   - config2: MODEL_STORAGE_DIR -> /root/Project/ML/app/models (hardcoded)
#   - main4.py: MODEL_DIR = "models" (shadowed the import)
#   - ModelFactory: searched "models/" (relative)
#   - Training: saved to "app/models/"
#
# NEW: One canonical path used everywhere.
MODEL_DIR = os.path.join(APP_DIR, "models")       # /root/Project/ML/app/models
MODEL_STORAGE_DIR = MODEL_DIR                       # Alias kept for backward compat

# Aliases for other scripts
RESULTS_DIR = OPTIMIZER_RESULTS_DIR
OPTIMIZER_DIR = OPTIMIZER_RESULTS_DIR

# Auto-Create structure if missing
REQUIRED_PATHS = [
    DATA_DIR, MODEL_DIR, LOG_DIR,
    OPTIMIZER_RESULTS_DIR, PREDICTORS_DIR, AGENTS_DIR
]
for path in REQUIRED_PATHS:
    os.makedirs(path, exist_ok=True)

# --- 2. ADVANCED ML SETTINGS (NEO-V7) ---
ML_CONFIG = {
    "DEFAULT_LONG_THRESHOLD": 0.65,
    "DEFAULT_SHORT_THRESHOLD": 0.35,
    "DEFAULT_EXIT_THRESHOLD": 0.50,
    "DEFAULT_LOOKBACK_WINDOW": 50,
    "FEATURE_SET": "technical_v7",
    "USE_ONNX_INFERENCE": False
}

# ============================================================
# 🔧 FIX #2: CENTRALIZED FEE CONSTANTS
# ============================================================
# Old mess:
#   - config2: DEFAULT_TAKER_FEE = 0.0006 (0.06%)
#   - backtest2.py: hardcoded 0.0006 (0.06%) ✓ matched config
#   - main4.py live engine: hardcoded 0.006 (0.6%) ✗ 10x higher!
#
# NEW: Single source of truth. Import these in main4 and backtest2.
# REALISTIC RETAIL FEES so paper-trading matches what users actually pay LIVE.
# 0.0006 (0.06%) was an institutional/high-volume rate — ~10x too low for a new
# retail account, which made every backtest wildly over-optimistic. Entry-tier
# 2026 spot TAKER is ~0.6-0.9% (Coinbase Advanced) / ~0.8% (Kraken Pro); MAKER
# (limit orders) is lower (~0.4%). Override per YOUR exchange + 30-day-volume
# tier via env so paper == live:  TAKER_FEE_PCT, MAKER_FEE_PCT, SLIPPAGE_BPS.
DEFAULT_TAKER_FEE = float(os.getenv("TAKER_FEE_PCT", "0.006"))        # 0.6% — realistic retail taker (market orders)
DEFAULT_MAKER_FEE = float(os.getenv("MAKER_FEE_PCT", "0.004"))        # 0.4% — maker (limit orders): the cheaper path
KRAKEN_TAKER_FEE  = float(os.getenv("KRAKEN_TAKER_FEE_PCT", "0.008")) # 0.8% — Kraken Pro base taker (2026)
SLIPPAGE_BPS      = float(os.getenv("SLIPPAGE_BPS", "1.0"))           # 0.1% expected slippage (market orders)

# Coinbase One (flat ~$30/mo subscription) = 0% Coinbase trading fees up to a
# monthly volume cap. DEFAULT ON ("use 0% venues until profits can absorb fees"):
# the Coinbase (spot / LONG) maker+taker fees are 0, so paper / validation / live
# == a Coinbase One or Binance.US (0% maker) account. Set COINBASE_ONE=false to
# trade at real retail fees. Kraken (shorts) is NOT covered and keeps its fee.
COINBASE_ONE = os.getenv("COINBASE_ONE", "true").lower() == "true"
if COINBASE_ONE:
    DEFAULT_TAKER_FEE = 0.0
    DEFAULT_MAKER_FEE = 0.0

# ============================================================
# 🔧 FIX #3: CREDENTIALS REMOVED FROM SOURCE CODE
# ============================================================
# Old: MONGO_URI had the real password as the default fallback,
# meaning the "env var" protection was useless.
#
# NEW: No fallback. If .env is missing, we fail loudly.
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    print("⚠️  MONGO_URI not found in .env — database features disabled.")
    print("   Add MONGO_URI=mongodb+srv://... to your .env file.")

# --- 3. DATABASE CONNECTION (Hardened) ---
class MemoryDB:
    """Fallback database if Atlas is unreachable"""
    def __init__(self):
        self.data = {}
        print("⚠️  Using In-Memory Mock Database")
    def find_one(self, query): return self.data.get(query.get("botId"))
    def find(self, query): return []
    def update_one(self, query, update, upsert=False): return None
    def insert_one(self, doc): return None
    def delete_many(self, query): return None
    def update_many(self, query, update): return None

bots_collection = None

if MONGO_URI:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tls=True)
        client.server_info()  # Trigger connection check
        db = client["test"]
        bots_collection = db["bots"]
        print("✅ Connected to MongoDB Atlas")

        # Startup Cleanup: Clear stale positions
        try:
            bots_collection.update_many({}, {"$unset": {"currentPosition": ""}})
        except:
            pass
    except Exception as e:
        print(f"❌ DB Connection Failed: {e}")
        bots_collection = MemoryDB()
else:
    bots_collection = MemoryDB()

# --- 4. SHARED FEATURE LIST (Single Source of Truth) ---
# ============================================================
# 🔧 FIX #1b: CANONICAL FEATURE LIST
# ============================================================
# Was hardcoded in 4 places: engineer_and_train.py, RawModelAdapter,
# backtest2.py calculate_indicators, and train_judge.py
#
# NEW: Import FEATURE_COLUMNS from config2 everywhere.
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'sma_50', 'sma_200', 'ema_9', 'ema_21', 'ema_20',
    'rsi', 'atr', 'adx', 'st_trend',
    'BBL_20_2.0_2.0', 'BBU_20_2.0_2.0',
    'STOCHk_14_3_3', 'MACD_12_26_9', 'MACDs_12_26_9',
    'pa_high', 'pa_low', 'vol_ma',
    'adx_logic', 'atr_logic', 'sma_logic'
]

print(f"🚀 NEO-V25 Config Active | Fee: {DEFAULT_TAKER_FEE*100}% | Features: {len(FEATURE_COLUMNS)} | ML Lookback: {ML_CONFIG['DEFAULT_LOOKBACK_WINDOW']}")
