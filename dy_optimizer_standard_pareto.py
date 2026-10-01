# File: dy_optimizer_standard_pareto.py
# 🚀 UPGRADE: v28.0 - "The Unified Specialist Gauntlet (Coinbase Edition)"

import requests, json, numpy as np, optuna, os, logging, sys
from colorama import Fore, init

init(autoreset=True)

# 🟢 CONFIG
ML_SERVER_URL = "http://74.208.28.77:8000"
RESULTS_DIR = "data/optimizer_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Logging to file and console
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s', 
    handlers=[logging.FileHandler("opt_stnd.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("StandardOptimizer")

# Strategy pool for specialized deep-dives
STRAT_POOL = ["sma_crossover", "macd_crossover", "rsi_threshold", "bollinger_bands", "ichimoku_cloud"]

def run_certification(trades, initial_balance=1000):
    """🛡️ Monte Carlo Certification (Luck Test)"""
    if not trades or len(trades) < 10: return False
    profits = np.array([t.get('profit', 0) for t in trades])
    ruin_count = 0
    for _ in range(5000):
        # Resample trades with replacement to simulate alternate market paths
        shuffled_path = np.random.choice(profits, size=len(profits), replace=True)
        if np.min(np.cumsum(shuffled_path) + initial_balance) < (initial_balance * 0.7):
            ruin_count += 1
    # Reject if > 5% risk of hitting 30% drawdown
    return (ruin_count / 5000) <= 0.05

def objective(trial):
    # 🟢 1. SPECIALIST SEARCH (Single deep layer optimization)
    code = trial.suggest_categorical("strat", STRAT_POOL)
    p = {}
    
    # Strict key alignment for the Strategy Registry
    if "sma" in code:
        p = {
            "fast_sma": trial.suggest_int("f", 5, 80), 
            "slow_sma": trial.suggest_int("s", 100, 500)
        }
    elif "rsi" in code:
        p = {
            "rsi_length": trial.suggest_int("r", 7, 30),
            "oversold_level": 30,
            "overbought_level": 70
        }
    elif "macd" in code:
        p = {"fast": 12, "slow": 26, "signal": 9}
    
    base_cfg = {
        "symbol": "BTC-USD", 
        "timeframe": "1h", 
        "initialBalance": 1000, 
        "risk_percentage": 2,
        "strategies": [{"code": code, "params": p}],
        "params": {
            "tslAtrMult": trial.suggest_float("tsl", 1.5, 4.5, step=0.5), 
            "minAdxLevel": trial.suggest_int("adx", 5, 20, step=5),
            "commission": 0.006,  # 🟢 COINBASE TAKER FEE
            "slippage": 0.001     # 🟢 COINBASE SPREAD
        }
    }

    # 🟢 2. RIGOROUS 2-YEAR GAUNTLET
    try:
        response = requests.post(
            f"{ML_SERVER_URL}/api/backtest/combo", 
            json={**base_cfg, "startDate": "2024-01-01", "endDate": "2026-01-01"}, 
            timeout=300
        )
        full_res = response.json().get("combinedResult", {})
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return -100, 100

    roi = full_res.get('metrics', {}).get('roi', -100)
    dd = abs(full_res.get('metrics', {}).get('maxDrawdown', 0))
    trades_count = full_res.get('metrics', {}).get('totalTrades', 0)

    # 🟢 3. COINBASE DRAG PENALTY
    # Every trade on Coinbase costs money. We penalize excessive trading 
    # to find high-conviction, low-frequency winners.
    drag_penalty = trades_count * 0.05 # Deduct 0.05% ROI per trade over 2 years
    adjusted_roi = roi - drag_penalty

    # 🟢 4. FINAL CERTIFICATION
    if roi > 20 and run_certification(full_res.get('trades', [])):
        logger.info(f"{Fore.CYAN}🎖️ CERTIFIED SPECIALIST: {code} | ROI {roi}% | DD {dd}%{Style.RESET_ALL}")
        
        # Save unique winner config
        save_path = os.path.join(RESULTS_DIR, f"CERT_STND_{code}_{int(roi)}pct.json")
        with open(save_path, 'w') as f:
            json.dump(base_cfg, f, indent=2)

    return adjusted_roi, dd

if __name__ == "__main__":
    logger.info("🚀 Starting Unified Standard Optimizer (Coinbase Mode)...")
    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=200)
