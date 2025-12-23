import os
import warnings
import threading
import logging

# Filter warnings to keep console clean
warnings.filterwarnings("ignore")

# --- 1. Project Directory Setup ---
# Automatically set root to current directory for local execution
PROJECT_ROOT = os.getcwd()

# Define sub-directories
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- 2. Database Connection (With Mock Fallback) ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority")

class MemoryDB:
    """
    A simple in-memory database shim that mimics PyMongo.
    Used when a real MongoDB connection is not available.
    """
    def __init__(self):
        self.data = {}
        print("⚠️  Using In-Memory Mock Database (No Persistence)")

    def find_one(self, query):
        # Very basic mock: returns the last saved state for a botId if exact match
        bot_id = query.get("botId")
        if bot_id and bot_id in self.data:
            return self.data[bot_id]
        return None

    def update_one(self, query, update, upsert=False):
        # Simulates an update/upsert
        bot_id = query.get("botId")
        if bot_id:
            if "$set" in update:
                if bot_id not in self.data:
                    self.data[bot_id] = {}
                self.data[bot_id].update(update["$set"])
        return None

    def insert_one(self, doc):
        bot_id = doc.get("botId")
        if bot_id:
            self.data[bot_id] = doc
        return None
    
    def delete_one(self, query):
        bot_id = query.get("botId")
        if bot_id and bot_id in self.data:
            del self.data[bot_id]
        return None

# Try to connect to MongoDB, fall back to MemoryDB on failure
try:
    from pymongo import MongoClient
    from pymongo.errors import ServerSelectionTimeoutError
    
    # Attempt connection with a short timeout to fail fast if local
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    client.server_info() # Trigger connection check
    
    db = client["sovereign_executive"]
    bots_collection = db["bots"]
    print("✅ Connected to MongoDB")
    
except Exception as e:
    # Fallback if pymongo not installed or connection fails
    bots_collection = MemoryDB()

# --- 3. Global Signals ---
# Event used to signal all running bot threads to stop
shutdown_event = threading.Event()
