import sys
import traceback
# Import the actual logic from your client file
# Make sure client_v47_final.py is in the same folder
try:
    import client_v35 as client
except ImportError:
    print("❌ Could not import client_v47_final.py. Make sure the filename matches!")
    sys.exit(1)

from copy import deepcopy

print("🚀 DEBUGGING CLIENT PAYLOAD GENERATION...")

# 1. Generate a Mock Trial (Sample Params)
# We mimic what Optuna does inside the loop
try:
    print("1️⃣  Generating Mock Configuration...")
    
    test_symbol = "BTC-USD"
    test_timeframe = "1h"
    
    # Get one fold (2023) to test
    folds = client.generate_wfo_folds()
    test_fold = [f for f in folds if "2023" in f['startDate']][0]
    
    # Base Config
    config = deepcopy(client.BASE_CONFIG)
    config['symbol'] = test_symbol
    config['timeframe'] = test_timeframe
    config['startDate'] = test_fold['startDate']
    config['endDate'] = test_fold['endDate']
    config['mlMode'] = "predictions"
    config['mlModel'] = "btc_1d_lightgbm_model" # Hardcoded for test
    config['mlThreshold'] = 0.60
    
    # Inject Strategies (like the optimizer does)
    config['strategies'] = [
        {"code": "sma_crossover", "params": {"sma_fast_period": 10, "sma_slow_period": 50}},
        {"code": "rsi_divergence", "params": {"rsi_length": 14, "oversold_level": 30, "overbought_level": 70}}
    ]
    
    # Inject Params
    client.set_nested_value(config, "params.hybridMode", "REGIME")
    client.set_nested_value(config, "params.regime_threshold", 25)
    client.set_nested_value(config, "params.minAdxLevel", 10)
    client.set_nested_value(config, "params.tslAtrMult", 3.0)
    
    print("✅ Configuration Built Successfully.")
    print(f"   URL: {client.ML_SERVER_URL}/api/ml/run-combo-backtest")
    # print(f"   Payload: {json.dumps(config, indent=2)}") 

except Exception as e:
    print(f"❌ FAILED TO BUILD CONFIG: {e}")
    traceback.print_exc()
    sys.exit(1)

# 2. Attempt Execution (Capturing the error)
print("\n2️⃣  Attempting Execution (Running 'execute_simulation')...")
try:
    # We call the function that was swallowing the error
    result = client.execute_simulation(config, dry_run=False)
    
    metrics = result.get('metrics', {})
    trades = metrics.get('totalTrades', 0)
    
    print("\n✅ EXECUTION FINISHED.")
    print(f"   Trades: {trades}")
    print(f"   Return: {metrics.get('totalReturn', 0)}%")
    
    if trades == 0:
        print("⚠️  RESULT: 0 Trades (This explains the -999 if persistent)")
    else:
        print("🎉 RESULT: SUCCESS! (The client logic works manually)")

except Exception as e:
    print("\n❌ EXECUTION CRASHED!")
    print("   This is the error the Optimizer was hiding:")
    print("=" * 50)
    print(e)
    print("=" * 50)
    traceback.print_exc()
