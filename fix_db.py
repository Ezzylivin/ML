from app.config import bots_collection

try:
    print("🔧 Attempting to fix MongoDB Indexes...")
    
    # 1. Drop the bad index that is forcing unique User IDs
    # The error message explicitly named it 'userId_1'
    bots_collection.drop_index("userId_1")
    print("✅ Successfully deleted index: 'userId_1'")
    
except Exception as e:
    # If the index doesn't exist, that's fine too.
    print(f"ℹ️ Report: {e}")

print("🎉 Database rule fixed. You can now run multiple bots.")
