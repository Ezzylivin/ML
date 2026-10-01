import logging
from datetime import datetime, timezone
import pandas as pd

# Import your manager class
from app.manager2 import PrecisionPyramidManager

def run_manager_stress_test():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("StressTester")

    # 1. Initialize with standard UI parameters
    test_params = {
        "commission": 0.001,      # 0.1% Binance Tier
        "slippage": 0.0005,       # 0.05%
        "long_threshold": 0.65,
        "short_threshold": 0.35,
        "minAdxLevel": 20,
        "tslAtrMult": 3.0
    }
    
    manager = PrecisionPyramidManager(
        capital=10000.0, 
        base_risk=0.02, # 2% risk per trade
        **test_params
    )

    print("🚀 STARTING MANAGER STRESS TEST...")
    print(f"Initial Equity: ${manager.current_equity}")

    # 2. Simulate High Conviction Long Entry
    # Logic: Prob 0.85 (High), ATR 150 (Low Vol), Price 40000
    print("\n--- TEST 1: High Conviction Entry ---")
    manager.handle(
        signal=1, 
        price=40000.0, 
        time=datetime.now(timezone.utc).isoformat(), 
        signal_prob=0.85, 
        atr=150.0, 
        adx=30.0
    )

    # 3. Simulate Price Movement and Equity Update
    print("\n--- TEST 2: Step Equity ---")
    current_price = 40500.0 # Price went up
    manager.step_equity(current_price, datetime.now(timezone.utc).isoformat())
    print(f"Mark-to-Market Equity: ${manager.current_equity}")

    # 4. Simulate Panic Volatility Exit
    print("\n--- TEST 3: High Volatility Exit Check ---")
    # Simulate a sudden spike in ATR (1600 / 40000 = 4% Volatility)
    # This should trigger the "Panic Vol" halving logic if we tried to enter now
    manager._close_position(price=41000.0, time=datetime.now(timezone.utc).isoformat(), reason="Panic Volatility Test")

    # 5. Review Results
    results = manager.get_results()
    print("\n--- FINAL METRICS ---")
    print(f"Final Balance: ${manager.current_equity:.2f}")
    print(f"Total Trades: {results['metrics']['totalTrades']}")
    print(f"ROI: {results['metrics']['roi']}%")
    
    if results['trades']:
        last_trade = results['trades'][0]
        print(f"Last Trade Profit (Net of Fees/Slip): ${last_trade['profit']}")

if __name__ == "__main__":
    run_manager_stress_test()
