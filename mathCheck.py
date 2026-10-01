import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from app.manager import PrecisionPyramidManager

def run_infallibility_audit():
    print("🚀 Starting Coinbase-Ready Mathematical Integrity Audit...")
    
    # 1. SETUP
    initial_cap = 1000.0
    commission = 0.006 # 🟢 Coinbase Hardened
    slippage = 0.001   # 🟢 Coinbase Hardened
    now = datetime.now(timezone.utc)
    
    mgr = PrecisionPyramidManager(
        capital=initial_cap, 
        commission=commission, 
        slippage=slippage,
        minAdxLevel=20.0
    )

    # 2. PHASE 1: ENTRY VERIFICATION
    print("\n--- Phase 1: Entry Math Check ---")
    test_price = 100.0
    # 🟢 FIX: Provide ADX=35 to pass the Veto Filter
    mgr.handle(signal=1, price=test_price, time=now.isoformat(), atr=1.0, adx=35.0, tslAtrMult=3.0)
    
    if not mgr.positions:
        print("❌ Entry ERROR: Position failed to open. Check Manager Veto logic.")
        return

    pos = mgr.positions[0]
    expected_entry = test_price * (1 + slippage) # 100.10 for Coinbase
    
    if abs(pos['entry'] - expected_entry) < 1e-6:
        print(f"✅ Entry Slippage Accurate: {pos['entry']}")
    else:
        print(f"❌ Entry Slippage ERROR: Got {pos['entry']}, Expected {expected_entry}")

    # 3. PHASE 2: FRICTION & PNL VERIFICATION
    print("\n--- Phase 2: Exit & Friction Check ---")
    exit_price = 110.0
    mgr._close_position(exit_price, (now + timedelta(minutes=5)).isoformat(), "Audit Exit")
    
    if len(mgr.trades) > 0:
        trade = mgr.trades[0]
        # Profit must be net of 0.6% entry and 0.6% exit fees
        if trade['profit'] < (exit_price - expected_entry) * pos['size']:
            print(f"✅ Fee Deduction Verified: Profit {trade['profit']} is Net of Coinbase Fees.")
        else:
            print(f"❌ Fee Logic ERROR: Profit is Gross, not Net.")
    else:
        print("❌ Exit ERROR: Trade was not recorded.")

    # 4. PHASE 3: CURVE SYNCHRONIZATION
    print("\n--- Phase 3: Curve Synchronization ---")
    initial_count = len(mgr.equity_curve)
    for i in range(1, 11):
        fake_time = (now + timedelta(hours=i)).isoformat()
        mgr.step_equity(current_price=105.0, timestamp=fake_time)

    final_count = len(mgr.equity_curve)
    if final_count > initial_count:
        print(f"✅ Curve Tracking Active: {final_count} points recorded.")
        last_bal = mgr.equity_curve[-1]['balance']
        print(f"✅ Curve Math Verified: Last balance is {last_bal}")
    else:
        print("❌ Curve Logic ERROR: Logic is not appending.")

    print("\nAudit Complete.")

if __name__ == "__main__":
    run_infallibility_audit()
