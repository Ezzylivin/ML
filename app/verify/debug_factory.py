import traceback
# Dummy ModelFactory to simulate the collision
class MockModelFactory:
    @staticmethod
    def load_model(model_type, symbol, **kwargs):
        print(f"✅ Success! Loaded {model_type} for {symbol}")
        print(f"📦 Remaining Kwargs: {list(kwargs.keys())}")

def test_sanitization():
    # Simulate the messy data coming from the UI
    kwargs = {
        "symbol": "BTC-USD",
        "model_type": "stacking",
        "params": {
            "model_type": "stacking",
            "commission": 0.006,
            "long_threshold": 0.65
        }
    }

    print("🔍 Testing Argument Sanitization...")
    try:
        # 1. Extract the primary value
        params = kwargs.get('params', {})
        model_type = kwargs.get('model_type') or params.get('model_type', 'stacking')
        symbol = kwargs.get('symbol', 'BTC-USD')

        # 2. SANITIZE: Deep purge of the duplicate key
        safe_kwargs = kwargs.copy()
        safe_kwargs.pop('model_type', None) # Remove from top level
        
        if 'params' in safe_kwargs:
            # Remove from nested params to stop the 'multiple values' crash
            safe_params = safe_kwargs['params'].copy()
            safe_params.pop('model_type', None)
            safe_kwargs['params'] = safe_params

        # 3. Call the factory
        MockModelFactory.load_model(
            model_type=model_type,
            symbol=symbol,
            **safe_kwargs
        )
    except TypeError as e:
        print(f"❌ FAILED: {e}")
    except Exception:
        print(traceback.format_exc())

if __name__ == "__main__":
    test_sanitization()
