# File: app/verify/check_db_inventory.py
from pymongo import MongoClient
try:
    from app.config2 import MONGO_URI
    import app.config2 as cfg
    DATABASE_NAME = getattr(cfg, 'DATABASE_NAME', getattr(cfg, 'DB_NAME', 'SovereignDB'))
except:
    MONGO_URI = "mongodb://localhost:27017"
    DATABASE_NAME = "SovereignDB"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

print(f"\n📊 DATABASE INVENTORY: {DATABASE_NAME}")
print("=" * 50)
collections = db.list_collection_names()

if not collections:
    print("❌ No collections found. Your database is EMPTY.")
else:
    for coll in sorted(collections):
        count = db[coll].count_documents({})
        print(f"[{coll:<20}] -> {count} documents")
print("=" * 50)
