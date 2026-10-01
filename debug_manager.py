import os
import sys
import inspect
from app.manager2 import PrecisionPyramidManager

def verify_manager():
    print("🔍 DIAGNOSTIC: PrecisionPyramidManager")
    
    # 1. Find where Python is loading this class from
    file_path = inspect.getfile(PrecisionPyramidManager)
    print(f"📍 Loaded from: {file_path}")
    
    # 2. List all available methods
    methods = [m[0] for m in inspect.getmembers(PrecisionPyramidManager, predicate=inspect.isfunction)]
    print(f"📜 Methods found: {methods}")
    
    if 'check_exit' in methods:
        print("✅ SUCCESS: 'check_exit' IS PRESENT in this version.")
    else:
        print("❌ FAILURE: 'check_exit' IS MISSING. You are running an old version.")
        
    # 3. Print the source code of the class (to be 100% sure)
    print("\n--- SOURCE CODE PREVIEW ---")
    print(inspect.getsource(PrecisionPyramidManager)[:500] + "...")

if __name__ == "__main__":
    verify_manager()

