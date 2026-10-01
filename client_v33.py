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
import hashlib
import logging
import traceback  # 🚀 ADDED FOR DEBUGGING
from copy import deepcopy
from colorama import Fore, Style, init

# Initialize Colorama
init(autoreset=True)

# --- CONFIGURATION ---
ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://74.208.28.77:8000")
RESULTS_DIR = "/root/Project/ML/data/optimizer_results"
TRIAL_LOG_FILE = os.path.join(RESULTS_DIR, "trial_history.jsonl")

os.makedirs(RESULTS_DIR, exist_ok=True)

# --- GLOBAL SETTINGS ---
SEED = 12345
PARALLEL_JOBS = 1  # 🚀 SET TO 1 TO ISOLATE THE ERROR
MAX_CONCURRENT_FOLDS = 10 
TIMEOUT_SECONDS = 600 
REPRODUCIBILITY_ATTEMPTS = 3 

# --- STRATEGY POOLS ---
TREND_STRATEGIES = ["sma_crossover", "macd_crossover", "ichimoku_cloud", "psar_signal", "obv_signal", "atr_breakout"]
RANGE_STRATEGIES = ["rsi_divergence", "stochastic_crossover", "bollinger_bands", "cci_oversold"]
ALL_AVAILABLE_MODELS = []

# --- BASE CONFIG ---
BASE_CONFIG = {
    "symbol": "btc-USD",
    "timeframe": "1h",
    "initialBalance": 300,
    "fee": 0.006,
    "riskManagementMode": "standard",
    "riskPercentage": 1,
    "optimizer_mode": True
}

# --- SANITY PARAMS ---
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

# ==============================================================================
# UTILITIES
# ==============================================================================

def set_global_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

def log_trial_result(trial_number, params, metrics):
    entry = {"timestamp": datetime.now().isoformat(), "trial": trial_number, "params": params, "metrics": metrics}
    try:
        with open(TRIAL_LOG_FILE, "a") as f: f.write(json.dumps(entry) + "\n")
    except: pass

def set_nested_value(d, keys, value):
    keys = keys.split('.')
    for key in keys[:-1]: d = d.setdefault(key, {})
    d[keys[-1]] = value

# ==============================================================================
# NETWORK
# ==============================================================================

def execute_simulation(config, dry_run=False):
    if dry_run: return {"metrics": {"totalReturn": 0}, "equityCurve": []}
    
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
    try:
        # 🚀 LOUD REQUEST: Print if non-200
        response = requests.post(url, json=config, timeout=TIMEOUT_SECONDS)
        if response.status_code != 200:
            print(f"\n{Fore.RED}❌ SERVER ERROR {response.status_code}:{Style.RESET_ALL}")
            print(f"{Fore.RED}{response.text[:500]}{Style.RESET_ALL}") # Print first 500 chars of error
            raise Exception(f"Server returned {response.status_code}")
            
        json_res = response.json()
        return json_res.get("combinedResult", json_res)
    except Exception as e:
        print(f"\n{Fore.RED}❌ NETWORK ERROR: {e}{Style.RESET_ALL}")
        raise e

def fetch_all_models(dry_run=False):
    if dry_run:
        global ALL_AVAILABLE_MODELS
        ALL_AVAILABLE_MODELS = [{"id": "btc_1h_mock_model", "name": "Mock Model"}]
        return True
    try:
        response = requests.get(f"{ML_SERVER_URL}/api/ml/available-models", timeout=10)
        if response.status_code == 200:
            global ALL_AVAILABLE_MODELS_REAL
            ALL_AVAILABLE_MODELS_REAL = response.json()
            globals()['ALL_AVAILABLE_MODELS'] = ALL_AVAILABLE_MODELS_REAL
            print(f"{Fore.GREEN}[API] Loaded {len(ALL_AVAILABLE_MODELS)} models.")
            return True
        return False
    except: return False

# ==============================================================================
# OPTIMIZER LOGIC
# ==============================================================================

def generate_wfo_folds():
    now = datetime.now(timezone.utc)
    start_year = 2018
    folds = []
    for year in range(start_year, now.year):
        folds.append({"startDate": f"{year}-01-01", "endDate": f"{year}-12-31", "id": year})
    folds.append({"startDate": f"{now.year}-01-01", "endDate": now.strftime("%Y-%m-%d"), "id": now.year})
    return folds

def calculate_stitched_metrics(equity_curves):
    if not equity_curves: return {"StitchedCalmarRatio": -999, "StitchedMaxDrawdown": 100}
    all_dfs = []
    last_balance = 0
    
    for i, fold_curve in enumerate(equity_curves):
        if not fold_curve: continue
        df = pd.DataFrame(fold_curve)
        if df.empty: continue
        if i == 0:
            df['balance_continuous'] = df['balance']
            last_balance = df['balance'].iloc[-1]
        else:
            initial_fold = fold_curve[0]['balance']
            df['balance_continuous'] = df['balance'] - initial_fold + last_balance
            last_balance = df['balance_continuous'].iloc[-1]
        all_dfs.append(df)
    
    if not all_dfs: return {"StitchedCalmarRatio": -999, "StitchedMaxDrawdown": 100}
    
    stitched_df = pd.concat(all_dfs, ignore_index=True)
    initial = stitched_df['balance_continuous'].iloc[0]
    final = stitched_df['balance_continuous'].iloc[-1]
    
    if initial <= 0: return {"StitchedCalmarRatio": -999, "StitchedMaxDrawdown": 100}
    
    peak = stitched_df['balance_continuous'].cummax()
    drawdown = (stitched_df['balance_continuous'] - peak) / (peak + 1e-9)
    max_dd = abs(drawdown.min() * 100)
    
    days = (pd.to_datetime(stitched_df['timestamp'].iloc[-1]) - pd.to_datetime(stitched_df['timestamp'].iloc[0])).days
    if days <= 0: days = 1
    annual_ret = ((final / initial) ** (365.25 / days)) - 1
    calmar = (annual_ret * 100) / (max_dd + 1e-9)
    
    return {"StitchedCalmarRatio": calmar, "StitchedMaxDrawdown": max_dd}

def create_objective(symbol, timeframe, wfo_folds, market_models, dry_run=False):
    def objective(trial: optuna.Trial):
        trial.set_user_attr("symbol", symbol)
        test_config = deepcopy(BASE_CONFIG)
        test_config['symbol'] = symbol.upper()
        test_config['timeframe'] = timeframe
        
        trend_strat = trial.suggest_categorical("trend_strategy", TREND_STRATEGIES)
        range_strat = trial.suggest_categorical("range_strategy", RANGE_STRATEGIES)
        trial.set_user_attr("combo_strategies", f"{trend_strat},{range_strat}")
        
        test_config['strategies'] = [{"code": trend_strat, "params": {}}, {"code": range_strat, "params": {}}]
        
        set_nested_value(test_config, "params.hybridMode", "REGIME")
        set_nested_value(test_config, "params.regime_threshold", trial.suggest_int("params.regime_threshold", 15, 40, step=5))
        set_nested_value(test_config, "params.minAdxLevel", trial.suggest_int("params.minAdxLevel", 0, 20, step=5))
        set_nested_value(test_config, "params.tslAtrMult", trial.suggest_float("params.tslAtrMult", 1.5, 6.0, step=0.5))

        # Dynamic Params
        if "bollinger_bands" in [trend_strat, range_strat]:
             set_nested_value(test_config, "params.bb_length", trial.suggest_int("params.bb_length", 15, 30))
             set_nested_value(test_config, "params.bb_std", trial.suggest_float("params.bb_std", 1.5, 2.5, step=0.5))
        if "rsi_divergence" in [trend_strat, range_strat]:
             set_nested_value(test_config, "params.rsi_length", trial.suggest_int("params.rsi_length", 10, 25))
        if "atr_breakout" in [trend_strat, range_strat]:
             set_nested_value(test_config, "params.atr_period", trial.suggest_int("params.atr_period", 10, 30))
             set_nested_value(test_config, "params.atr_multiplier", trial.suggest_float("params.atr_multiplier", 1.0, 4.0, step=0.5))
        if "cci_oversold" in [trend_strat, range_strat]:
             set_nested_value(test_config, "params.cci_length", trial.suggest_int("params.cci_length", 14, 40))
             set_nested_value(test_config, "params.cci_oversold", trial.suggest_int("params.cci_oversold", -150, -50))
             set_nested_value(test_config, "params.cci_overbought", trial.suggest_int("params.cci_overbought", 50, 150))

        test_config['mlMode'] = "predictions"
        test_config['mlModel'] = trial.suggest_categorical("mlModel", market_models)
        test_config['mlThreshold'] = trial.suggest_float("mlThreshold", 0.51, 0.65, step=0.02)

        all_curves = [None] * len(wfo_folds)
        cumulative_return = 0.0
        
        # 🚀 LOUD EXECUTION: NO THREADS, PRINT ERRORS
        for idx, fold in enumerate(wfo_folds):
            try:
                fold_config = deepcopy(test_config)
                fold_config['startDate'] = fold['startDate']
                fold_config['endDate'] = fold['endDate']
                
                print(f"{Fore.CYAN}   Running Fold {fold['id']} ({fold['startDate']})...{Style.RESET_ALL}")
                
                # 🚀 THIS IS WHERE IT USUALLY FAILS SILENTLY
                res = execute_simulation(fold_config, dry_run)
                
                metrics = res.get('metrics', {})
                trades = metrics.get('totalTrades', 0)
                ret = metrics.get('totalReturn', 0.0)
                
                print(f"     -> Trades: {trades} | Return: {ret:.2f}%")
                
                cumulative_return += ret
                trial.report(cumulative_return, idx)
                if trial.should_prune(): raise optuna.TrialPruned()
                all_curves[idx] = res.get('equityCurve', [])
                
            except optuna.TrialPruned: raise
            except Exception as e:
                # 🚀 THE TRUTH WILL BE REVEALED HERE
                print(f"\n{Fore.RED}🔥 CRASH IN FOLD {fold['id']}!{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Error: {e}{Style.RESET_ALL}")
                traceback.print_exc() # PRINT THE FULL ERROR
                return -999, 100 

        metrics = calculate_stitched_metrics(all_curves)
        calmar = metrics.get('StitchedCalmarRatio', -999)
        dd = metrics.get('StitchedMaxDrawdown', 100)
        
        log_trial_result(trial.number, trial.params, metrics)
        print(f"{Fore.WHITE}Trial {trial.number}: {trend_strat}/{range_strat} | Calmar: {calmar:.2f} | DD: {dd:.2f}%")
        return calmar, dd

    return objective

def save_best_params(study, trial):
    try:
        best_trials = study.best_trials
        if not best_trials: return
        sorted_best = sorted(best_trials, key=lambda t: t.values[0], reverse=True)
        calmar_king = sorted_best[0]

        if calmar_king.values[0] <= 0.0: return 

        if calmar_king.number == trial.number:
            symbol = trial.user_attrs.get("symbol", "UNKNOWN")
            filename = f"winner_{symbol}_LOUD_DEBUG.json"
            filepath = os.path.join(RESULTS_DIR, filename)
            
            output = {
                "timestamp": datetime.now().isoformat(),
                "calmar": trial.values[0],
                "drawdown": trial.values[1],
                "params": trial.params,
            }
            with open(filepath, 'w') as f: json.dump(output, f, indent=4)
            print(f"{Fore.MAGENTA}💾 WINNER SAVED! (Calmar: {trial.values[0]:.2f}){Style.RESET_ALL}")
    except Exception as e: pass

def run_optimizer(args):
    symbol = "BTC-USD"; timeframe = "1h"
    
    if not fetch_all_models(args.dry_run):
        print(f"{Fore.RED}No models found. Check server.{Style.RESET_ALL}")
        return

    m_check = f"{symbol.split('-')[0]}_{timeframe}"
    ml_models = [m['id'] for m in ALL_AVAILABLE_MODELS if m['id'].lower().startswith(m_check.lower())]
    
    if not ml_models:
        print(f"{Fore.RED}No matching ML models for {m_check}.{Style.RESET_ALL}")
        return

    print(f"{Fore.YELLOW}⚠️ LOUD DEBUG MODE: 1 Job, Sequential Folds.{Style.RESET_ALL}")
    
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=optuna.samplers.TPESampler(seed=SEED))

    print(f"\n{Fore.CYAN}🚀 STARTING OPTIMIZATION (LOUD MODE){Style.RESET_ALL}")
    
    wfo_folds = generate_wfo_folds()
    study.optimize(
        create_objective(symbol, timeframe, wfo_folds, ml_models, args.dry_run), 
        n_trials=10, # Just run 10 to find the error
        n_jobs=1, 
        callbacks=[save_best_params] 
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    set_global_seed(SEED)
    run_optimizer(args)

if __name__ == "__main__":
    main()
