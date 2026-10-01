#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import pandas as pd
import numpy as np
import time
import optuna
from datetime import datetime, timedelta
import os
import sys
import random
from copy import deepcopy
from colorama import Fore, Style, init
import warnings

# ----------------------- INITIAL SETUP -----------------------
init(autoreset=True)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://127.0.0.1:8000")
RESULTS_DIR = os.path.join(os.getcwd(), "data", "optimizer_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 12345
PARALLEL_JOBS = 6
TIMEOUT_SECONDS = 600
REPRODUCIBILITY_ATTEMPTS = 3
MIN_TRADES_FOR_VALIDITY = 2
PRUNE_DRAWDOWN_LIMIT = 35.0
PRUNE_RETURN_LIMIT = -20.0

TREND_STRATEGIES = ["sma_crossover", "macd_crossover", "ichimoku_cloud", "psar_signal", "obv_signal", "atr_breakout"]
RANGE_STRATEGIES = ["rsi_divergence", "stochastic_crossover", "bollinger_bands", "cci_oversold"]

BASE_CONFIG = {
    "symbol": "BTC-USD",
    "timeframe": "1h",
    "initialBalance": 1000,
    "fee": 0.006,
    "riskManagementMode": "standard",
    "riskPercentage": 1,
    "optimizer_mode": True
}

SANITY_PARAMS = {
    "sma_crossover": {"sma_fast_period": 5, "sma_slow_period": 10},
    "macd_crossover": {"macd_fast_period": 12, "macd_slow_period": 26, "macd_signal_period": 9},
    "ichimoku_cloud": {"tenkan_period": 9, "kijun_period": 26, "senkou_period": 52},
    "psar_signal": {"psar_step": 0.02, "psar_max": 0.2},
    "obv_signal": {"obv_ma_period": 10},
    "atr_breakout": {"atr_period": 14, "atr_multiplier": 0.5},
    "rsi_divergence": {"rsi_length": 14, "oversold_level": 45, "overbought_level": 55},
    "stochastic_crossover": {"k_period": 14, "d_period": 3, "smooth_k": 3},
    "bollinger_bands": {"bb_length": 20, "bb_std": 1.5},
    "cci_oversold": {"cci_length": 14, "cci_oversold": -10, "cci_overbought": 10}
}

ALL_AVAILABLE_MODELS = []

# ----------------------- UTILITIES -----------------------
def set_global_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

def set_nested_value(d, keys, value):
    keys = keys.split('.')
    for key in keys[:-1]: d = d.setdefault(key, {})
    d[keys[-1]] = value

def robust_request(method, url, json_data=None, retries=3):
    for i in range(retries):
        try:
            if method == 'GET': response = requests.get(url, timeout=10)
            else: response = requests.post(url, json=json_data, timeout=600)
            if response.status_code == 200: return response
            elif response.status_code >= 500: 
                print(f"{Fore.YELLOW}⚠️ Server Error {response.status_code}. Retrying...{Style.RESET_ALL}")
                time.sleep(1)
            else: return response
        except requests.exceptions.RequestException as e:
            print(f"{Fore.RED}⚠️ Connection Refused ({i+1}/{retries}). {e}{Style.RESET_ALL}")
            time.sleep(1)
    raise Exception(f"Failed to connect to {url}")

def execute_simulation(config, dry_run=False):
    if dry_run:
        s_date = config.get('startDate', '2020-01-01')
        dates = pd.date_range(start=s_date, periods=50, freq='h')
        fake_curve = [{"timestamp": d.isoformat(), "balance": 1000 * (1 + (0.01 * i))} for i, d in enumerate(dates)]
        return {"metrics": {"totalReturn": 50.0, "totalTrades": 10, "finalBalance": 1050.0}, "equityCurve": fake_curve, "tradeBreakdown": []}
    
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
    response = robust_request('POST', url, json_data=config)
    if response.status_code != 200: raise Exception(f"Server Error: {response.text}")
    return response.json().get("combinedResult", response.json())

def fetch_all_models(dry_run=False):
    global ALL_AVAILABLE_MODELS
    if dry_run:
        ALL_AVAILABLE_MODELS = [{"id": "btc_1h_mock_model", "name": "Mock Model"}]
        return True
    try:
        response = robust_request('GET', f"{ML_SERVER_URL}/api/ml/available-models")
        if response.status_code == 200:
            ALL_AVAILABLE_MODELS = response.json()
            return True
        return False
    except Exception as e:
        print(f"{Fore.RED}❌ Could not connect to ML Server: {e}{Style.RESET_ALL}")
        return False

def extract_unique_symbols():
    symbols = set()
    for model in ALL_AVAILABLE_MODELS:
        parts = model['id'].split('_')
        if len(parts) >= 2:
            base = parts[0].upper()
            if "-" not in base: base += "-USD"
            symbols.add(base)
    return sorted(list(symbols))

def extract_timeframes_for_symbol(symbol):
    timeframes = set()
    base_symbol = symbol.split('-')[0].lower()
    for model in ALL_AVAILABLE_MODELS:
        parts = model['id'].split('_')
        if len(parts) >= 2 and parts[0] == base_symbol: timeframes.add(parts[1])
    return sorted(list(timeframes))

def select_from_list(title, options):
    print(f"\n{Fore.CYAN}--- Select {title} ---{Style.RESET_ALL}")
    for i, opt in enumerate(options): print(f"{i+1}. {opt}")
    while True:
        choice = input(f"{Fore.YELLOW}Enter number (default 1): {Style.RESET_ALL}").strip()
        if not choice: return options[0]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options): return options[idx]
        except ValueError: pass

# ----------------------- AUDIT -----------------------
def run_comprehensive_audit(symbol, timeframe, dry_run=False):
    print(f"\n{Fore.CYAN}🕵️  Running Comprehensive Strategy Audit on {symbol} {timeframe} (ML: {'OFF' if dry_run else 'ON'})...{Style.RESET_ALL}")
    strategies = TREND_STRATEGIES + RANGE_STRATEGIES
    audit_results = []

    for strat in strategies:
        config = deepcopy(BASE_CONFIG)
        config['symbol'] = symbol
        config['timeframe'] = timeframe
        config['strategies'] = [{"code": strat, "params": SANITY_PARAMS.get(strat, {})}]
        config['startDate'] = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        config['endDate'] = datetime.now().strftime("%Y-%m-%d")
        config['mlMode'] = "off"

        try:
            result = execute_simulation(config, dry_run=dry_run)
            metrics = result.get('metrics', {})
            total_trades = metrics.get('totalTrades', 0)
            total_return = metrics.get('totalReturn', 0)
            final_balance = metrics.get('finalBalance', 0)
            status = f"{Fore.GREEN}PASS (Profit){Style.RESET_ALL}" if total_return > 0 else f"{Fore.YELLOW}PASS (Loss){Style.RESET_ALL}"
            if total_trades == 0: status = f"{Fore.RED}FAIL (No Trades){Style.RESET_ALL}"
            audit_results.append({
                "Strategy": strat,
                "Trades": total_trades,
                "Return": f"{total_return:.2f}%",
                "Balance": f"${final_balance:,.2f}",
                "Status": status
            })
        except Exception as e:
            audit_results.append({
                "Strategy": strat,
                "Trades": 0,
                "Return": "N/A",
                "Balance": "N/A",
                "Status": f"{Fore.RED}ERROR{Style.RESET_ALL}: {e}"
            })

    header = f"{'Strategy':<25} | {'Trades':<7} | {'Return':<10} | {'Balance':<15} | Status"
    print("\n" + header)
    print("-" * len(header))
    for row in audit_results:
        print(f"{row['Strategy']:<25} | {row['Trades']:<7} | {row['Return']:<10} | {row['Balance']:<15} | {row['Status']}")

# ----------------------- OPTIMIZATION -----------------------
def optimize_strategy(strategy, symbol, timeframe, dry_run=False, n_trials=20):
    print(f"\n{Fore.MAGENTA}⚡ Starting optimizer for {strategy} ({symbol} {timeframe}){Style.RESET_ALL}")

    def objective(trial):
        config = deepcopy(BASE_CONFIG)
        config['symbol'] = symbol
        config['timeframe'] = timeframe
        params = {}
        for key, val in SANITY_PARAMS[strategy].items():
            if isinstance(val, int):
                params[key] = trial.suggest_int(key, max(1, val//2), val*2)
            elif isinstance(val, float):
                params[key] = trial.suggest_float(key, val*0.5, val*2)
        config['strategies'] = [{"code": strategy, "params": params}]
        config['startDate'] = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        config['endDate'] = datetime.now().strftime("%Y-%m-%d")
        config['mlMode'] = "off"

        result = execute_simulation(config, dry_run=dry_run)
        metrics = result.get('metrics', {})
        total_trades = metrics.get('totalTrades', 0)
        total_return = metrics.get('totalReturn', 0)
        max_drawdown = metrics.get('maxDrawdown', 0)

        # Pruning
        if total_trades < MIN_TRADES_FOR_VALIDITY or total_return < PRUNE_RETURN_LIMIT or max_drawdown > PRUNE_DRAWDOWN_LIMIT:
            raise optuna.exceptions.TrialPruned()

        return total_return

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    print(f"{Fore.GREEN}✅ Best params: {study.best_params} | Best Return: {study.best_value:.2f}%{Style.RESET_ALL}")
    return study.best_params

# ----------------------- MAIN -----------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    set_global_seed(SEED)
    print(f"{Fore.CYAN}=== 🚀 Crypto Strategy Optimizer v2.0 ==={Style.RESET_ALL}")

    if not fetch_all_models(args.dry_run) and not args.dry_run:
        print(f"{Fore.RED}Could not fetch models from server!{Style.RESET_ALL}")

    symbols = extract_unique_symbols() if ALL_AVAILABLE_MODELS else ["BTC-USD"]
    selected_symbol = select_from_list("symbol", symbols)
    timeframes = extract_timeframes_for_symbol(selected_symbol) if ALL_AVAILABLE_MODELS else ["1h"]
    selected_timeframe = select_from_list("timeframe", timeframes)

    run_audit = True
    if run_audit:
        run_comprehensive_audit(selected_symbol, selected_timeframe, dry_run=args.dry_run)

    # Run optimizer for each strategy
    strategies = TREND_STRATEGIES + RANGE_STRATEGIES
    for strat in strategies:
        optimize_strategy(strat, selected_symbol, selected_timeframe, dry_run=args.dry_run, n_trials=10)

if __name__ == "__main__":
    main()
