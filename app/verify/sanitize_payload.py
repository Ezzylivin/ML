import json

def sanitize_ui_payload(payload):
    """
    Strips redundant model_type and symbol keys from the payload
    to prevent Python argument collisions.
    """
    # 1. Define the 'Protected' keys we pass explicitly to functions
    protected_keys = {'model_type', 'symbol', 'startDate', 'endDate', 'initial_balance'}
    
    print(f"🧹 Sanitizing payload. Protecting: {protected_keys}")

    # 2. Clean the top-level
    clean_payload = {k: v for k, v in payload.items() if k not in protected_keys}
    
    # 3. Clean the nested params object (where most collisions live)
    if 'params' in clean_payload and isinstance(clean_payload['params'], dict):
        clean_payload['params'] = {
            k: v for k, v in clean_payload['params'].items() if k not in protected_keys
        }
        
    return clean_payload

if __name__ == "__main__":
    # --- STANDALONE TEST CASE ---
    dirty_payload = {
        "symbol": "BTC-USD",
        "model_type": "stacking",
        "startDate": "2025-08-21",
        "params": {
            "model_type": "stacking",
            "symbol": "BTC-USD",
            "commission": 0.006,
            "lookback": 50
        }
    }
    
    clean = sanitize_ui_payload(dirty_payload)
    print("\n✨ Cleaned Params Keys:", list(clean['params'].keys()))
    
    if 'symbol' in clean['params'] or 'model_type' in clean['params']:
        print("❌ FAILURE: Duplicates still exist.")
    else:
        print("✅ SUCCESS: Collisions removed.")
