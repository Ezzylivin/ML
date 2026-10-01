import os
import json
import numpy as np
from colorama import Fore, Style, init

# Import your five pillars
try:
    from LookAheadDetect import detect_look_ahead
    from ExpectancyAudit import calculate_expectancy
    from KellyCalibrator import run_kelly_audit
    from RuinGauntlet import RuinGauntlet
    from WalkForwardAudit import run_walk_forward_audit
except ImportError as e:
    print(f"{Fore.RED}❌ INTEGRATION ERROR: {e}")
    print("Ensure all 5 audit files are in the same folder as Certify.py.")

init(autoreset=True)

def calculate_alpha_grade(metrics):
    """Calculates a letter grade based on strategy robustness."""
    score = 0
    # 1. Expectancy (Max 30 pts)
    if metrics['expectancy'] > 0.005: score += 30
    elif metrics['expectancy'] > 0: score += 15
    
    # 2. Risk of Ruin (Max 30 pts)
    if metrics['ruin'] < 1: score += 30
    elif metrics['ruin'] < 5: score += 20
    elif metrics['ruin'] < 15: score += 10
    
    # 3. Walk-Forward Efficiency (Max 40 pts)
    if metrics['wfe'] > 0.7: score += 40
    elif metrics['wfe'] > 0.5: score += 25
    elif metrics['wfe'] > 0.3: score += 10

    if score >= 90: return "A+ (SOVEREIGN)"
    if score >= 80: return "A (INSTITUTIONAL)"
    if score >= 70: return "B (ROBUST)"
    if score >= 50: return "C (FRAGILE)"
    return "F (REJECTED)"

def run_full_certification():
    print(Fore.CYAN + "="*60)
    print(Fore.WHITE + " 🛡️  SOVEREIGN QUANT STRATEGY CERTIFICATION ".center(60))
    print(Fore.CYAN + "="*60)

    # 1. LOAD DATA
    results_file = 'latest_results.json'
    holdout_file = 'holdout_results.json'

    if not os.path.exists(results_file):
        print(Fore.RED + f"FATAL: {results_file} not found. Run an optimization first.")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)
        backtest = data.get('combinedResult', data)
        trades = backtest.get('trades', [])
        candles = backtest.get('candleData', [])
        profits = [float(t.get('profit', 0)) for t in trades]

    # 2. EXECUTE FIVE PILLAR AUDIT
    print(f"\n{Fore.YELLOW}[PILLAR 1] INTEGRITY")
    detect_look_ahead(trades, candles)

    print(f"\n{Fore.YELLOW}[PILLAR 2] EXPECTANCY")
    exp_val = calculate_expectancy(profits) # Modified to return value

    print(f"\n{Fore.YELLOW}[PILLAR 3] RISK SIZING")
    run_kelly_audit(profits)

    print(f"\n{Fore.YELLOW}[PILLAR 4] STABILITY")
    gauntlet = RuinGauntlet(profits)
    ruin_prob = gauntlet.run() # Modified to return prob

    print(f"\n{Fore.YELLOW}[PILLAR 5] ROBUSTNESS")
    wfe_val = 0
    if os.path.exists(holdout_file):
        # run_walk_forward_audit returns efficiency ratio
        wfe_val = run_walk_forward_audit() 
    else:
        print(Fore.MAGENTA + "⚠️ Holdout data missing. Cannot calculate Alpha Grade.")

    # 3. FINAL GRADE
    final_metrics = {
        'expectancy': exp_val if exp_val else 0,
        'ruin': ruin_prob if ruin_prob else 100,
        'wfe': wfe_val if wfe_val else 0
    }
    
    grade = calculate_alpha_grade(final_metrics)

    print(Fore.CYAN + "\n" + "="*60)
    print(Fore.WHITE + f" FINAL STRATEGY GRADE: {grade} ".center(60))
    print(Fore.CYAN + "="*60)

if __name__ == "__main__":
    run_full_certification()
