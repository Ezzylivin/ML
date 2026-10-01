# 🚀 UPGRADE: v22.1 - "Optimizer Supreme: Interactive Edition (Deprecation Fix)"
# Features:
#   1. Interactive CLI Menu
#   2. Dynamic User Configuration
#   3. Real API Connection
#   4. Multi-Objective Logic
#   5. Timezone-aware Datetimes (Fixes DeprecationWarning)

import requests
import json
import time
import optuna
from datetime import datetime, timedelta, timezone
import os
import random
import argparse
import logging
import traceback
from colorama import Fore, Style, init
import uuid
from typing import Dict, List, Optional

# Initialize Colorama for nice UI
init(autoreset=True)

# ===================================================
# CONFIGURATIONS
# ===================================================
ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://127.0.0.1:8000")
RESULTS_DIR = os.path.join(os.getcwd(), "data", "optimizer_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Setup Logging
logging.basicConfig(
    filename="optimizer_errors.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("Optimizer")
logger.setLevel(logging.INFO)

# Global Settings (Populated by Menu)
USER_SETTINGS = {
    "symbol": "BTC-USD",
    "timeframe": "optimize", # 'optimize' lets Optuna pick, or specific like '1h'
    "days_back": 60,
    "risk_level": "medium"   # limits ranges
}

# ===================================================
# STRATEGY POOLS & PARAMETERS
# ===================================================
TREND_POOL = ["sma_crossover", "macd_crossover", "atr_breakout"]
RANGE_POOL = ["rsi_divergence", "stochastic_crossover", "bollinger_bands"]

def get_strategy_params(trial, strategy_code):
    """
    Defines the search space dynamically based on strategy type.
    """
    if strategy_code == "sma_crossover":
        return {
            "sma_fast_period": trial.suggest_int("sma_fast", 5, 60, step=5),
            "sma_slow_period": trial.suggest_int("sma_slow", 40, 200, step=10)
        }
    elif strategy_code == "macd_crossover":
        return {
            "macd_fast_period": trial.suggest_int("macd_fast", 8, 20),
            "macd_slow_period": trial.suggest_int("macd_slow", 21, 60),
            "macd_signal_period": trial.suggest_int("macd_sig", 5, 15)
        }
    elif strategy_code == "rsi_divergence":
        return {
            "rsi_length": trial.suggest_int("rsi_len", 10, 28),
            "oversold_level": trial.suggest_int("rsi_os", 20, 45),
            "overbought_level": trial.suggest_int("rsi_ob", 55, 80)
        }
    elif strategy_code == "atr_breakout":
        return {
            "atr_period": trial.suggest_int("atr_per", 10, 30),
            "atr_multiplier": trial.suggest_float("atr_mult", 1.0, 4.0, step=0.1)
        }
    return {}

# ===================================================
# API COMMUNICATOR
# ===================================================
def run_remote_backtest(config: Dict) -> Dict:
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
    if "userId" not in config: config["userId"] = "optimizer_agent"

    try:
        response = requests.post(url, json=config, timeout=900)
        if response.status_code != 200: return {}
        
        data = response.json()
        result = data.get("combinedResult") or data
        if not result.get("metrics"): return {}
        return result
    except:
        return {}

# ===================================================
# OPTUNA OBJECTIVE
# ===================================================
def objective(trial):
    # 1. Use User Settings
    symbol = USER_SETTINGS["symbol"]
    
    # Timeframe: Fixed by user OR Optimized by AI
    if USER_SETTINGS["timeframe"] == "optimize":
        timeframe = trial.suggest_categorical("timeframe", ["1h", "4h"])
    else:
        timeframe = USER_SETTINGS["timeframe"]

    # 2. Select Strategy Mix
    strat_trend = trial.suggest_categorical("strat_trend", TREND_POOL)
    strat_range = trial.suggest_categorical("strat_range", RANGE_POOL)
    
    strategies = [
        {"code": strat_trend, "params": get_strategy_params(trial, strat_trend)},
        {"code": strat_range, "params": get_strategy_params(trial, strat_range)}
    ]

    # 3. Risk Params
    risk_pct = trial.suggest_float("risk_pct", 0.5, 3.0, step=0.5)
    max_pyramiding = trial.suggest_int("max_pyramid", 1, 3)
    
    # 4. Payload (🟢 FIXED: Use datetime.now(timezone.utc))
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "initialBalance": 1000.0,
        "startDate": (datetime.now(timezone.utc) - timedelta(days=USER_SETTINGS["days_back"])).strftime("%Y-%m-%d"),
        "endDate": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "strategies": strategies,
        "comboConfig": {
            "combinationRule": "OR", 
            "strategyCodes": [s["code"] for s in strategies]
        },
        "params": {
            "riskPercentage": risk_pct,
            "maxPyramiding": max_pyramiding,
            "tradeDirection": "both"
        },
        "mlMode": "off"
    }

    # 5. Execute
    result = run_remote_backtest(payload)
    metrics = result.get("metrics", {})
    
    # 6. Scoring
    roi = metrics.get("totalReturn", -100.0)
    trades = metrics.get("totalTrades", 0)
    dd = metrics.get("maxDrawdown", 100.0)

    # Filters
    if trades < 3: return -100.0   # Not enough data
    if dd > 25: return -100.0      # Too risky
    
    # Save Winners
    if roi > 5.0: # Only save strictly positive results
        save_result(payload, result, roi)

    return roi

# ===================================================
# HELPER: SAVE RESULT
# ===================================================
def save_result(config, result, roi):
    filename = f"{config['symbol']}_{roi:.2f}pct_{uuid.uuid4().hex[:6]}.json"
    path = os.path.join(RESULTS_DIR, filename)
    data = {"roi": roi, "metrics": result.get("metrics"), "config": config}
    with open(path, "w") as f: json.dump(data, f, indent=2)
    
    # Nice print output
    t_frame = config['timeframe']
    strats = [s['code'] for s in config['strategies']]
    print(f"   {Fore.GREEN}✨ WINNER FOUND: {roi:.2f}% ROI | {t_frame} | {strats}{Style.RESET_ALL}")

# ===================================================
# 🟢 INTERACTIVE MENU
# ===================================================
def run_interactive_menu():
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"   🚀 OPTIMIZER SUPREME - INTERACTIVE MODE")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    # 1. Symbol
    sym = input(f"   📝 Target Symbol [BTC-USD]: ").strip().upper()
    if not sym: sym = "BTC-USD"
    USER_SETTINGS["symbol"] = sym

    # 2. Timeframe
    tf = input(f"   ⏰ Timeframe (1h, 4h, 15m or 'opt' for auto) [opt]: ").strip().lower()
    if not tf or tf == 'opt': USER_SETTINGS["timeframe"] = "optimize"
    else: USER_SETTINGS["timeframe"] = tf

    # 3. Days Back
    days = input(f"   📅 Days to Look Back [60]: ").strip()
    USER_SETTINGS["days_back"] = int(days) if days.isdigit() else 60

    # 4. Trials
    trials = input(f"   🎲 Number of Trials [50]: ").strip()
    n_trials = int(trials) if trials.isdigit() else 50

    # 5. Parallel Jobs
    jobs = input(f"   ⚡ Parallel Jobs (1-4) [1]: ").strip()
    n_jobs = int(jobs) if jobs.isdigit() else 1

    print(f"\n{Fore.YELLOW}   ⚡ STARTING OPTIMIZATION ENGINE...{Style.RESET_ALL}")
    print(f"   Target: {sym} | Timeframe: {USER_SETTINGS['timeframe']} | Trials: {n_trials}\n")
    
    return n_trials, n_jobs

# ===================================================
# MAIN
# ===================================================
if __name__ == "__main__":
    # Check for CLI args first (for automated cron jobs)
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, help="Number of trials")
    parser.add_argument("--symbol", type=str, help="Target Symbol")
    args = parser.parse_args()

    if args.trials:
        # Non-Interactive Mode (Cron/Script)
        n_trials = args.trials
        n_jobs = 1
        if args.symbol: USER_SETTINGS["symbol"] = args.symbol
        print(f"🚀 Running Auto-Mode: {USER_SETTINGS['symbol']} for {n_trials} trials")
    else:
        # Interactive Mode
        n_trials, n_jobs = run_interactive_menu()

    # Run Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Reduce noise
    study = optuna.create_study(direction="maximize")
    
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}🛑 Optimization paused by user.{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"   🏆 OPTIMIZATION COMPLETE")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    if len(study.trials) > 0:
        best = study.best_trial
        print(f"   🥇 Best ROI: {Fore.GREEN}{best.value:.2f}%{Style.RESET_ALL}")
        print(f"   ⚙️  Best Params: {best.params}")
        print(f"\n   📂 Results saved to: {RESULTS_DIR}")
    else:
        print("   ❌ No successful trials completed.")
