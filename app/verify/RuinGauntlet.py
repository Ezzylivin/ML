import numpy as np
import json
import os
from colorama import Fore, Style, init

init(autoreset=True)

class RuinGauntlet:
    def __init__(self, trades, initial_balance=1000):
        self.trades = np.array(trades)
        self.initial_balance = initial_balance
        self.ruin_threshold = 0.30  # 30% drawdown = Failure
        self.iterations = 5000

    def run(self):
        print(f"{Fore.CYAN}--- 🛡️ RUNNING THE RUIN GAUNTLET ---")
        
        if len(self.trades) < 5:
            print(f"{Fore.RED}❌ ERROR: Not enough trades to simulate a multiverse.")
            return

        # 🚀 THE MATRIX METHOD
        rng = np.random.default_rng()
        # Shuffling the trade order 5000 different ways
        shuffled_indices = rng.random((self.iterations, len(self.trades))).argsort(axis=1)
        shuffled_profits = self.trades[shuffled_indices]

        # Calculate compounded equity across all 5000 paths
        equity_curves = self.initial_balance * np.cumprod(1 + shuffled_profits, axis=1)
        
        # Calculate Peak Equity and Drawdown
        peaks = np.maximum.accumulate(equity_curves, axis=1)
        drawdowns = (peaks - equity_curves) / peaks
        max_dds = np.max(drawdowns, axis=1)
        
        # Risk of Ruin calculation
        ruin_count = np.sum(np.min(equity_curves, axis=1) < (self.initial_balance * (1 - self.ruin_threshold)))
        prob_of_ruin = (ruin_count / self.iterations) * 100

        # Stats
        avg_dd = np.mean(max_dds) * 100
        worst_dd = np.max(max_dds) * 100
        median_return = (np.median(equity_curves[:, -1]) / self.initial_balance - 1) * 100

        print(f"Total Trades: {len(self.trades)} | Paths Simulated: {self.iterations}")
        
        status_color = Fore.GREEN if prob_of_ruin < 5 else (Fore.YELLOW if prob_of_ruin < 15 else Fore.RED)
        print(f"\nRisk of Ruin: {status_color}{prob_of_ruin:.2f}%")
        print(f"Median ROI: {Fore.WHITE}{median_return:.2f}%")
        print(f"Worst Case DD: {Fore.RED}{worst_dd:.2f}%")

        if prob_of_ruin > 10:
            print(f"\n{Fore.RED}🚨 WARNING: Strategy is fragile. You likely got lucky in the backtest order.")
        else:
            print(f"\n{Fore.GREEN}💎 CERTIFIED: Strategy survives random sequence risk.")

if __name__ == "__main__":
    results_file = 'latest_results.json'
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            data = json.load(f)
            backtest = data.get('combinedResult', data)
            trade_list = backtest.get('trades', [])
            # Extract 'profit' field from each trade dict
            profits = [float(t.get('profit', 0)) for t in trade_list]
            
            gauntlet = RuinGauntlet(profits, initial_balance=backtest.get('initialBalance', 1000))
            gauntlet.run()
    else:
        print(f"{Fore.YELLOW}⚠️ Gauntlet Idle: '{results_file}' not found. Save backtest JSON first.")
