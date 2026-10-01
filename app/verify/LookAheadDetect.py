import numpy as np
import json
import os
from colorama import Fore, Style, init

init(autoreset=True)

def detect_look_ahead(trades, candles):
    print(f"{Fore.CYAN}--- 🔍 LOOK-AHEAD BIAS AUDIT ---")
    
    if not trades or not candles:
        print(f"{Fore.RED}❌ Error: No data to analyze. Ensure trades and candles are provided.")
        return

    # Create lookup map
    candle_map = {c['time']: c for c in candles}
    violations = 0
    suspicious_trades = []

    for i, trade in enumerate(trades):
        # Handle different potential key names from backend
        t_time = trade.get('entry_time') or trade.get('time')
        entry_price = float(trade['entry_price'])
        
        candle = candle_map.get(t_time)
        if candle:
            low = float(candle['low'])
            high = float(candle['high'])
            
            # If entry price is EXACTLY the low or high, it's a peek-ahead red flag
            if entry_price == low or entry_price == high:
                suspicious_trades.append({"index": i, "time": t_time, "price": entry_price})
                violations += 1

    if violations > 0:
        print(f"{Fore.RED}❌ FAILED: {violations} suspicious entries found.")
        print(f"{Fore.YELLOW}Look-ahead bias detected. You are likely entering at a candle's extreme price.")
    else:
        print(f"{Fore.GREEN}✅ PASSED: No look-ahead bias detected in {len(trades)} trades.")

# --- 🚀 AUTO-RUNNER LOGIC ---
if __name__ == "__main__":
    # 1. Try to find your latest backtest results file
    # Replace 'latest_results.json' with whatever your backend saves as
    results_file = 'latest_results.json' 
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            data = json.load(f)
            # Backend might return data nested in 'combinedResult' or 'results'
            backtest = data.get('combinedResult', data)
            
            detect_look_ahead(
                backtest.get('trades', []), 
                backtest.get('candleData', [])
            )
    else:
        print(f"{Fore.YELLOW}⚠️ Audit Idle: Could not find '{results_file}'.")
        print("To run this, ensure your backend saves the backtest JSON to this folder.")
