import numpy as np
import json
import os
from colorama import Fore, Style, init

init(autoreset=True)

def run_kelly_audit(trades):
    if not trades:
        print(f"{Fore.RED}❌ ERROR: No trades found for Kelly Audit.")
        return

    # Split trades into wins and losses
    wins = [t for t in trades if t > 0]
    losses = [abs(t) for t in trades if t <= 0]
    
    if not wins or not losses:
        print(f"{Fore.YELLOW}⚠️ Not enough diverse data (need wins and losses) for Kelly math.")
        return

    # Inputs for Kelly
    p = len(wins) / len(trades) # Win Probability
    q = 1 - p                  # Loss Probability
    b = np.mean(wins)          # Avg Win %
    a = np.mean(losses)        # Avg Loss %
    
    # 🧮 Leverage Factor (How many "accounts" of size to buy)
    # This formula is optimized for trading percentage outcomes
    kelly_leverage = (p / a) - (q / b)
    
    # 🛡️ Recommended Risk per trade (Percentage of account to lose on SL)
    # Standard practice is Quarter Kelly (0.25)
    recommended_risk = (kelly_leverage * a) * 0.25

    print(f"{Fore.CYAN}--- ⚖️ KELLY RISK REPORT ---")
    print(f"Win Rate: {p*100:.2f}% | R:R Ratio: {b/a:.2f}")
    
    if kelly_leverage <= 0:
        print(f"Mathematical Edge: {Fore.RED}{kelly_leverage:.4f}")
        print(f"{Fore.RED}🛑 STATUS: UNTRADEABLE. Strategy has a negative expectancy.")
    else:
        print(f"Mathematical Edge: {Fore.GREEN}{kelly_leverage:.4f}")
        print(f"Safe Risk Per Trade: {Fore.GREEN}{recommended_risk*100:.2f}%")
        print(f"{Fore.WHITE}Advice: Set your dashboard 'Risk %' to the value above.")

if __name__ == "__main__":
    results_file = 'latest_results.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            data = json.load(f)
            backtest = data.get('combinedResult', data)
            profits = [float(t.get('profit', 0)) for t in backtest.get('trades', [])]
            run_kelly_audit(profits)
    else:
        print(f"{Fore.YELLOW}⚠️ Kelly Idle: No '{results_file}' found.")
