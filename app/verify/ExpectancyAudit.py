import numpy as np
import json
import os
from colorama import Fore, Style, init

init(autoreset=True)

def calculate_expectancy(trades):
    if not trades:
        print(f"{Fore.RED}❌ ERROR: No trades found to analyze.")
        return

    # trades should be a list of profit/loss percentages
    wins = [t for t in trades if t > 0]
    losses = [abs(t) for t in trades if t <= 0]
    
    wr = len(wins) / len(trades)
    avg_w = np.mean(wins) if wins else 0
    avg_l = np.mean(losses) if losses else 0
    
    # Mathematical Edge Formula: E = (WR * AvgW) - (LR * AvgL)
    expectancy = (wr * avg_w) - ((1 - wr) * avg_l)
    
    print(f"{Fore.CYAN}--- 📊 EXPECTANCY AUDIT ---")
    print(f"Win Rate: {wr*100:.2f}% | Avg Win: {avg_w*100:.2f}% | Avg Loss: {avg_l*100:.2f}%")
    
    if expectancy <= 0:
        print(f"{Fore.RED}Mathematical Expectancy: {expectancy:.4f}")
        print(f"{Fore.RED}❌ WARNING: This strategy has NO mathematical edge. It will lose money over time.")
    else:
        print(f"{Fore.GREEN}Mathematical Expectancy: {expectancy:.4f}")
        print(f"{Fore.GREEN}✅ SUCCESS: Every trade is worth {expectancy*100:.2f}% on average.")

# --- 🚀 AUTO-LOADER BLOCK ---
if __name__ == "__main__":
    results_file = 'latest_results.json'
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            data = json.load(f)
            # Support both direct results and combined result objects
            backtest = data.get('combinedResult', data)
            
            # Extract profits from the trades list
            # We assume your backend returns a list of trade dicts with a 'profit' key
            trade_list = backtest.get('trades', [])
            profits = [float(t.get('profit', 0)) for t in trade_list]
            
            calculate_expectancy(profits)
    else:
        print(f"{Fore.YELLOW}⚠️ Audit Idle: '{results_file}' not found. Save your backtest JSON first.")
