import json
import os
import sys
from colorama import Fore, Style, init

init(autoreset=True)

def run_walk_forward_audit():
    print(f"{Fore.CYAN}--- 🧪 WALK-FORWARD ROBUSTNESS AUDIT ---")

    # 1. FILE PATHS
    # Optimization results (In-Sample)
    train_file = 'latest_results.json'
    # Test results on unseen data (Out-of-Sample)
    holdout_file = 'holdout_results.json'

    # 2. FILE VALIDATION
    if not os.path.exists(train_file) or not os.path.exists(holdout_file):
        print(f"{Fore.RED}❌ ERROR: Missing required data files.")
        print(f"{Fore.YELLOW}Expected: '{train_file}' AND '{holdout_file}'")
        print("💡 Step: Run your optimization, save it, then run the same config on a different date range and save as 'holdout_results.json'.")
        return

    # 3. LOAD DATA
    try:
        with open(train_file, 'r') as f:
            train_data = json.load(f)
            train_results = train_data.get('combinedResult', train_data)
        
        with open(holdout_file, 'r') as f:
            holdout_data = json.load(f)
            holdout_results = holdout_data.get('combinedResult', holdout_data)
    except Exception as e:
        print(f"{Fore.RED}❌ ERROR: Failed to parse JSON data: {e}")
        return

    # 4. EXTRACT METRICS
    train_metrics = train_results.get('metrics', {})
    hold_metrics = holdout_results.get('metrics', {})

    t_roi = float(train_metrics.get('roi', 0))
    h_roi = float(hold_metrics.get('roi', 0))
    
    t_dd = abs(float(train_metrics.get('maxDrawdown', 1)))
    h_dd = abs(float(hold_metrics.get('maxDrawdown', 1)))

    # 5. ROBUSTNESS CALCULATIONS
    # WFE: Walk Forward Efficiency (Target > 0.5)
    wfe = (h_roi / t_roi) if t_roi > 0 else 0
    
    # Consistency Ratio: Compare Calmar Ratios (Return/Drawdown)
    t_calmar = t_roi / t_dd if t_dd != 0 else 0
    h_calmar = h_roi / h_dd if h_dd != 0 else 0
    consistency = (h_calmar / t_calmar) if t_calmar > 0 else 0

    # 6. REPORTING
    print(f"\n{Fore.WHITE}>>> PERFORMANCE COMPARISON:")
    print(f"In-Sample (Train) ROI:   {Fore.YELLOW}{t_roi:.2f}%")
    print(f"Out-of-Sample (Holdout) ROI: {Fore.YELLOW}{h_roi:.2f}%")
    
    print(f"\n{Fore.WHITE}>>> STABILITY METRICS:")
    status_color = Fore.GREEN if wfe >= 0.5 else (Fore.YELLOW if wfe >= 0.3 else Fore.RED)
    print(f"Walk-Forward Efficiency: {status_color}{wfe:.2f}")
    
    calmar_status = Fore.GREEN if consistency >= 0.5 else Fore.RED
    print(f"Risk Consistency Ratio:  {calmar_status}{consistency:.2f}")

    print(f"\n{Fore.CYAN}--- 📜 FINAL VERDICT ---")
    if wfe >= 0.5 and consistency >= 0.5:
        print(f"{Fore.GREEN}✅ ROBUST: Strategy translates well to unseen data.")
    elif wfe < 0.3:
        print(f"{Fore.RED}❌ OVERFIT: Performance collapsed on holdout data. Parameters are too specific.")
    elif wfe > 1.5:
        print(f"{Fore.MAGENTA}⚠️ OUTLIER: Holdout outperformed Training. Check for news-driven luck.")
    else:
        print(f"{Fore.YELLOW}⚠️ FRAGILE: Strategy is borderline. Proceed with reduced risk.")

if __name__ == "__main__":
    run_walk_forward_audit()
