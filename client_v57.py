# File: client_v57.py (run_optimizer.py)
# 🚀 UPGRADE: v3.1 - Fixed NameError
# - Restored missing PARALLEL_JOBS constant.

import requests
import json
import pandas as pd
import numpy as np
import time
import optuna
from datetime import datetime, timedelta, timezone
import os
import sys
import random
import argparse
import logging
import traceback
from copy import deepcopy
from colorama import Fore, Style, init
import warnings

# ----------------------- INITIAL SETUP -----------------------
init(autoreset=True)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configuration
ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://127.0.0.1:8000")
RESULTS_DIR = os.path.join(os.getcwd(), "data", "optimizer_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 12345
set_random_seed = lambda: (random.seed(SEED), np.random.seed(SEED))
set_random_seed()

# --- SETTINGS (Fixed) ---
PARALLEL_JOBS = 1           # 🚀 RESTORED: Set to 1 for stability with SQLite/File locks
TIMEOUT_SECONDS = 600 

# 🚀 GHOST BUSTER SETTINGS
MIN_TRADES_TOTAL = 20       # Must trade >20 times to be real
PRUNE_DRAWDOWN_LIMIT = 50.0 # Allow some drawdown to find aggressive growth
MAX_CALMAR_CAP = 30.0       # Reject "Infinite" fake scores

# Strategy Pools
TREND_STRATEGIES = ["sma_crossover", "macd_crossover", "ichimoku_cloud", "psar_signal", "obv_signal", "atr_breakout"]
RANGE_STRATEGIES = ["rsi_divergence", "stochastic_crossover", "bollinger_bands", "cci_oversold"]
ALL_AVAILABLE_MODELS = []

# Base Config
BASE_CONFIG = {
    "symbol": "BTC-USD",
    "timeframe": "1h",
    "initialBalance": 1000,
    "fee": 0.006,
    "riskManagementMode": "standard",
    "riskPercentage": 1, 
    "optimizer_mode": True
}

# Sanity Defaults (For Audit)
SANITY_PARAMS = {
    "sma_crossover": {"sma_fast_period": 10, "sma_slow_period": 50},
    "macd_crossover": {"macd_fast_period": 12, "macd_slow_period": 26, "macd_signal_period": 9},
    "ichimoku_cloud": {"tenkan_period": 9, "kijun_period": 26, "senkou_period": 52},
    "psar_signal": {"psar_step": 0.02, "psar_max": 0.2},
    "obv_signal": {"obv_ma_period": 20},
    "atr_breakout": {"atr_period": 14, "atr_multiplier": 2.0},
    "rsi_divergence": {"rsi_length": 14, "oversold_level": 30, "overbought_level": 70},
    "stochastic_crossover": {"k_period": 14, "d_period": 3},
    "bollinger_bands": {"bb_length": 20, "bb_std": 2.0},
    "cci_oversold": {"cci_length": 20, "cci_oversold": -100, "cci_overbought": 100}
}

# ==============================================================================
# 1. UI UTILITIES (BOXES & MENUS)
# ==============================================================================
def print_header(text):
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f" {text.center(58)} ")
    print(f"{'='*60}{Style.RESET_ALL}")

def print_box(title, content, color=Fore.GREEN):
    print(f"\n{color}┌{'─'*58}┐")
    print(f"│ {title:<56} │")
    print(f"├{'─'*58}┤")
    for line in content:
        print(f"│ {line:<56} │")
    print(f"└{'─'*58}┘{Style.RESET_ALL}")

def select_from_list(title, options):
    print(f"\n{Fore.YELLOW}>>> Select {title}:{Style.RESET_ALL}")
    for i, opt in enumerate(options):
        print(f"  {Fore.CYAN}{i+1}.{Style.RESET_ALL} {opt}")
    
    while True:
        choice = input(f"\n{Fore.GREEN}Enter number (default 1): {Style.RESET_ALL}").strip()
        if not choice: return options[0]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options): return options[idx]
        except ValueError: pass
        print(f"{Fore.RED}Invalid selection.{Style.RESET_ALL}")

# ==============================================================================
# 2. NETWORK & DATA
# ==============================================================================
def robust_request(method, url, json_data=None, retries=3):
    for i in range(retries):
        try:
            if method == 'GET': response = requests.get(url, timeout=10)
            else: response = requests.post(url, json=json_data, timeout=300)
            
            if response.status_code == 200: return response
            elif response.status_code >= 500: 
                print(f"{Fore.YELLOW}⚠️ Server Error {response.status_code}. Retrying...{Style.RESET_ALL}")
                time.sleep(1)
            else: return response
        except requests.exceptions.RequestException:
            time.sleep(1)
    raise Exception(f"Failed to connect to {url}")

def fetch_all_models(dry_run=False):
    global ALL_AVAILABLE_MODELS
    if dry_run:
        ALL_AVAILABLE_MODELS = [{"id": "btc_1h_mock_model", "name": "Mock Model"}]
        return True
    try:
        print_header("CONNECTING TO ML SERVER")
        response = robust_request('GET', f"{ML_SERVER_URL}/api/ml/available-models")
        if response.status_code == 200:
            ALL_AVAILABLE_MODELS = response.json()
            print(f"✅ Connected. Found {len(ALL_AVAILABLE_MODELS)} models.")
            return True
        return False
    except: 
        print(f"{Fore.RED}❌ Failed to connect to ML Server at {ML_SERVER_URL}{Style.RESET_ALL}")
        return False

def execute_simulation(config, dry_run=False):
    if dry_run: 
        return {"metrics": {"totalReturn": 10.0, "totalTrades": 25, "maxDrawdown": 5.0}, "equityCurve": []}
    
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
    response = robust_request('POST', url, json_data=config)
    if response.status_code != 200: raise Exception(f"Server Error: {response.text}")
    
    json_res = response.json()
    if "combinedResult" in json_res: return json_res["combinedResult"]
    return json_res

# ==============================================================================
# 3. OPTIMIZER CORE
# ==============================================================================
def set_nested(d, keys, value):
    keys = keys.split('.')
    for key in keys[:-1]: d = d.setdefault(key, {})
    d[keys[-1]] = value

def create_objective(symbol, timeframe, dry_run=False):
    
    # Prepare Data Range (Rolling 2 Years)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    def objective(trial: optuna.Trial):
        # 1. Strategy Selection
        trend_strat = trial.suggest_categorical("trend", TREND_STRATEGIES)
        range_strat = trial.suggest_categorical("range", RANGE_STRATEGIES)
        trial.set_user_attr("combo", f"{trend_strat}+{range_strat}")

        # 2. Parameter Generation
        strategies = []
        def gen_params(code):
            p = {}
            if code == "sma_crossover":
                s1 = trial.suggest_int(f"{code}_s1", 5, 60, step=5)
                s2 = trial.suggest_int(f"{code}_s2", 20, 200, step=10)
                p = {"sma_fast_period": min(s1,s2), "sma_slow_period": max(s1,s2)}
            elif code == "macd_crossover":
                p = {
                    "macd_fast_period": trial.suggest_int(f"{code}_f", 8, 20),
                    "macd_slow_period": trial.suggest_int(f"{code}_s", 21, 50),
                    "macd_signal_period": trial.suggest_int(f"{code}_sig", 5, 15)
                }
            elif code == "rsi_divergence":
                p = {
                    "rsi_length": trial.suggest_int(f"{code}_len", 10, 30),
                    "oversold_level": trial.suggest_int(f"{code}_os", 20, 40),
                    "overbought_level": trial.suggest_int(f"{code}_ob", 60, 80)
                }
            elif code == "bollinger_bands":
                p = {
                    "bb_length": trial.suggest_int(f"{code}_len", 15, 30),
                    "bb_std": trial.suggest_float(f"{code}_std", 1.5, 3.0, step=0.1)
                }
            elif code == "atr_breakout":
                 p = {
                     "atr_period": trial.suggest_int(f"{code}_p", 10, 30),
                     "atr_multiplier": trial.suggest_float(f"{code}_m", 1.5, 5.0, step=0.1)
                 }
            # Generic fallback for others
            elif code == "stochastic_crossover":
                p = {"k_period": 14, "d_period": 3}
            elif code == "cci_oversold":
                p = {"cci_length": 20, "cci_oversold": -100, "cci_overbought": 100}
            elif code == "psar_signal":
                p = {"psar_step": 0.02, "psar_max": 0.2}
            elif code == "ichimoku_cloud":
                 p = {"tenkan_period": 9, "kijun_period": 26}
            elif code == "obv_signal":
                 p = {"obv_ma_period": 20}
                 
            return p

        strategies.append({"code": trend_strat, "params": gen_params(trend_strat)})
        strategies.append({"code": range_strat, "params": gen_params(range_strat)})

        # 3. Config Construction
        cfg = deepcopy(BASE_CONFIG)
        cfg['symbol'] = symbol
        cfg['timeframe'] = timeframe
        cfg['strategies'] = strategies
        cfg['startDate'] = start_date
        cfg['endDate'] = end_date

        # 4. Hybrid Logic & Risk
        set_nested(cfg, "params.hybridMode", "REGIME")
        set_nested(cfg, "params.regime_threshold", trial.suggest_int("regime_thresh", 15, 35, step=5))
        
        # 🚀 CRITICAL: Allow AI to tune Stop Loss (1.0 = Tight, 5.0 = Loose)
        set_nested(cfg, "params.tslAtrMult", trial.suggest_float("tsl_mult", 1.5, 5.0, step=0.5))

        # 5. Execution
        try:
            res = execute_simulation(cfg, dry_run)
            m = res.get('metrics', {})
            
            trades = m.get('totalTrades', 0)
            dd = abs(m.get('maxDrawdown', 0))
            ret = m.get('totalReturn', 0)
            
            # 6. Ghost Buster Filtering
            if trades < MIN_TRADES_TOTAL:
                # print(f"{Fore.BLACK}[Trial {trial.number}] Pruned: Low Trades ({trades}){Style.RESET_ALL}")
                raise optuna.exceptions.TrialPruned()
            
            if dd > PRUNE_DRAWDOWN_LIMIT:
                # print(f"{Fore.BLACK}[Trial {trial.number}] Pruned: High DD ({dd:.2f}%){Style.RESET_ALL}")
                raise optuna.exceptions.TrialPruned()

            # 7. Scoring (Calmar Ratio)
            calmar = ret / (dd if dd > 0 else 1.0)
            if calmar > MAX_CALMAR_CAP: calmar = 0.1 # Reject fake perfection
            
            # Save metadata for best trial retrieval
            trial.set_user_attr("calmar", calmar)
            trial.set_user_attr("return", ret)
            trial.set_user_attr("dd", dd)
            trial.set_user_attr("trades", trades)
            trial.set_user_attr("config", cfg)

            color = Fore.GREEN if calmar > 2.0 else Fore.YELLOW
            print(f"{color}[Trial {trial.number}] {trend_strat[:10]}/{range_strat[:10]} | Calmar: {calmar:.2f} | Ret: {ret:.1f}% | DD: {dd:.1f}% | Trades: {trades}{Style.RESET_ALL}")
            
            return calmar

        except optuna.exceptions.TrialPruned: raise
        except Exception: return -9999

    return objective

# ==============================================================================
# 4. AUDIT LOGIC
# ==============================================================================
def run_comprehensive_audit(symbol, timeframe, dry_run):
    print_header(f"STRATEGY AUDIT: {symbol} {timeframe}")
    
    base = deepcopy(BASE_CONFIG)
    base['symbol'] = symbol
    base['timeframe'] = timeframe
    base['endDate'] = datetime.now().strftime("%Y-%m-%d")
    base['startDate'] = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    print(f"{'Strategy':<25} | {'Trades':<8} | {'Return':<10} | {'Status'}")
    print("-" * 60)
    
    all_passed = True
    for strat in TREND_STRATEGIES + RANGE_STRATEGIES:
        cfg = deepcopy(base)
        params = SANITY_PARAMS.get(strat, {})
        cfg['strategies'] = [{"code": strat, "params": params}]
        
        try:
            res = execute_simulation(cfg, dry_run)
            m = res.get('metrics', {})
            trades = m.get('totalTrades', 0)
            ret = m.get('totalReturn', 0)
            
            status = f"{Fore.GREEN}PASS{Style.RESET_ALL}" if trades > 0 else f"{Fore.RED}FAIL (0 Trades){Style.RESET_ALL}"
            print(f"{strat:<25} | {trades:<8} | {ret:>8.2f}% | {status}")
            if trades == 0: all_passed = False
        except:
            print(f"{strat:<25} | ERROR    | 0.00%      | {Fore.RED}CRASH{Style.RESET_ALL}")
    
    return all_passed

# ==============================================================================
# 5. MAIN ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not fetch_all_models(args.dry_run): return

    # --- 1. INTERACTIVE MENU ---
    print_header("SETUP OPTIMIZER")
    
    # Get Symbols
    symbols = list(set([m['id'].split('_')[0].upper() + "-USD" for m in ALL_AVAILABLE_MODELS]))
    if not symbols: symbols = ["BTC-USD", "ETH-USD"]
    selected_symbol = select_from_list("Symbol", sorted(symbols))

    # Get Timeframes
    selected_timeframe = select_from_list("Timeframe", ["1h", "4h", "15m"])
    
    # Get Trials
    try:
        n_trials = int(input(f"\n{Fore.YELLOW}Enter Number of Trials (Default 50): {Style.RESET_ALL}").strip() or 50)
    except: n_trials = 50

    # --- 2. AUDIT ---
    run_comprehensive_audit(selected_symbol, selected_timeframe, args.dry_run)

    # --- 3. OPTIMIZATION ---
    print_header(f"STARTING OPTIMIZATION ({n_trials} Trials)")
    print(f"Target: {selected_symbol} {selected_timeframe} | Filters: >{MIN_TRADES_TOTAL} Trades, <{PRUNE_DRAWDOWN_LIMIT}% DD")
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.NSGAIISampler(seed=SEED))
    
    try:
        study.optimize(create_objective(selected_symbol, selected_timeframe, args.dry_run), n_trials=n_trials, n_jobs=PARALLEL_JOBS)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Optimization paused by user.{Style.RESET_ALL}")

    # --- 4. RESULTS ---
    print_header("🏆 TOP 3 GOLDEN STRATEGIES")
    
    best_trials = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value, reverse=True)
    
    for i, t in enumerate(best_trials[:3]):
        cal = t.value
        cfg = t.user_attrs.get('config')
        trades = t.user_attrs.get('trades')
        ret = t.user_attrs.get('return')
        dd = t.user_attrs.get('dd')
        combo = t.user_attrs.get('combo')

        if not cfg: continue
        
        print_box(f"RANK #{i+1}: {combo}", [
            f"Calmar Ratio : {cal:.2f}",
            f"Total Return : {ret:.2f}%",
            f"Max Drawdown : {dd:.2f}%",
            f"Total Trades : {trades}"
        ], color=Fore.MAGENTA if i==0 else Fore.CYAN)
        
        # Save Winner
        fname = f"winner_{selected_symbol}_{selected_timeframe}_RANK{i+1}_{int(time.time())}.json"
        
        # Add Metadata for UI
        final_data = {
            "strategies": cfg['strategies'],
            "params": cfg['params'],
            "mlMode": "predictions", # Suggest enabling ML for final
            "mlModel": f"{selected_symbol.split('-')[0].lower()}_1d_xgboost_model", # Guess model name or fetch logic
            "mlThreshold": 0.6,
            "metrics": {"calmar": cal, "totalReturn": ret, "maxDrawdown": dd, "totalTrades": trades},
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(RESULTS_DIR, fname), 'w') as f:
            json.dump(final_data, f, indent=2)
        print(f"{Fore.GREEN}   💾 Saved to: {fname}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
