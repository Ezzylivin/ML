import os
import warnings
import threading
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Filter warnings
warnings.filterwarnings("ignore")

# --- 1. Project Directory Setup (THE SIBLING FIX) ---

# Step 1: Get the absolute path of this file (app/config.py)
CURRENT_FILE = os.path.abspath(__file__)

# Step 2: Get the directory containing this file (ML/app)
APP_DIR = os.path.dirname(CURRENT_FILE)

# Step 3: Go UP one level to the Project Root (ML/)
PROJECT_ROOT = os.path.dirname(APP_DIR)

# Step 4: Define Paths relative to Root
# This creates: /root/Project/ML/data/optimizer_results
OPTIMIZER_DIR = os.path.join(PROJECT_ROOT, "data", "optimizer_results")
RESULTS_DIR = OPTIMIZER_DIR  # Alias for compatibility with other modules
MODEL_DIR = os.path.join(APP_DIR, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs") # 🟢 Added Missing LOG_DIR

# --- DEBUG PRINTS (Check your logs for these!) ---
print(f"📍 Config Location: {APP_DIR}")
print(f"🏠 Project Root:    {PROJECT_ROOT}")
print(f"📂 Target Data Dir: {OPTIMIZER_DIR}")

if os.path.exists(OPTIMIZER_DIR):
    print(f"✅ FOUND: {len(os.listdir(OPTIMIZER_DIR))} files in optimizer_results")
else:
    print(f"❌ ERROR: Directory does not exist at {OPTIMIZER_DIR}")

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OPTIMIZER_DIR, exist_ok=True)

# --- 2. Database Connection ---
MONGO_URI = "mongodb+srv://ericjdickerson93:Dadeadend1!!@cluster0.wzztvhe.mongodb.net/?retryWrites=true&w=majority"

if os.getenv("MONGO_URI"):
    MONGO_URI = os.getenv("MONGO_URI")

class MemoryDB:
    def __init__(self):
        self.data = {}
        print("⚠️  Using In-Memory Mock Database")
    def find_one(self, query): return self.data.get(query.get("botId"))
    def find(self, query): return [] # Return empty list iterator
    def update_one(self, query, update, upsert=False): return None
    def insert_one(self, doc): return None
    def delete_one(self, query): return None
    def delete_many(self, query): return None 
    def update_many(self, query, update): return None
    def aggregate(self, pipeline): return []

try:
    from pymongo import MongoClient
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tls=True)
    # Trigger connection check
    client.server_info()
    db = client["test"]
    bots_collection = db["bots"]
    print("✅ Connected to MongoDB Atlas")
    
    # 🧹 Automatic Cleanup on Startup
    try:
        bots_collection.update_many({}, {"$unset": {"currentPosition": ""}})
    except: pass

except Exception as e:
    print(f"❌ Connection Failed: {e}")
    bots_collection = MemoryDB()

# --- 3. Global Signals & Locks (🟢 ADDED THESE) ---
bots_lock = threading.Lock()
active_live_bots = {}
shutdown_event = threading.Event()

# Default Settings
DEFAULT_TAKER_FEE = 0.0006
SLIPPAGE_BPS = 2.0
