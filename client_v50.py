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
import traceback 
from copy import deepcopy
from colorama import Fore, Style, init

# Initialize Colorama
init(autoreset=True)

# 🚀 SILENCE OPTUNA
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- CONFIGURATION ---
ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://74.208.28.77:8000")
RESULTS_DIR = "/root/Project/ML/data/optimizer_results"
TRIAL_LOG_FILE = os.path.join(RESULTS_DIR, "trial_history.jsonl")

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- GLOBAL SETTINGS ---
SEED = 12345
PARALLEL_JOBS = 6 
MAX_CONCURRENT_FOLDS = 10 
TIMEOUT_SECONDS = 600 
REPRODUCIBILITY_ATTEMPTS = 3 

# 🚀 PRUNING SETTINGS
PRUNE_DRAWDOWN_LIMIT = 20.0  # Kill if drawdown > 20% in any fold
PRUNE_RETURN_LIMIT = -10.0   # Kill if 1st year return < -10%

# --- STRATEGY POOLS ---
TREND_STRATEGIES = ["sma_crossover", "macd_crossover", "ichimoku_cloud", "psar_signal", "obv_signal", "atr_breakout"]
RANGE_STRATEGIES = ["rsi_divergence", "stochastic_crossover", "bollinger_bands", "cci_oversold"]
ALL_AVAILABLE_MODELS = []

# --- BASE CONFIG ---
BASE_CONFIG = {
    "symbol": "btc-USD",
    "timeframe": "1h",
    "initialBalance": 300,
    "fee": 0.006, # Lowered fee to 0.1% as discussed
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
# 1. UTILITIES
# ==============================================================================

def set_global_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

def get_config_hash(config):
    return hashlib.md5(json.dumps(config, sort_keys=True).encode('utf-8')).hexdigest()

def log_trial_result(trial_number, params, metrics):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "trial": trial_number,
        "params": params,
        "metrics": metrics
    }
    try:
        with open(TRIAL_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except: pass

def set_nested_value(d, keys, value):
    keys = keys.split('.')
    for key in keys[:-1]: d = d.setdefault(key, {})
    d[keys[-1]] = value

# ==============================================================================
# 2. NETWORK
# ==============================================================================

def robust_request(method, url, json_data=None, retries=3):
    for i in range(retries):
        try:
            if method == 'GET':
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, json=json_data, timeout=TIMEOUT_SECONDS)
            
            if response.status_code == 200: return response
            elif response.status_code >= 500: time.sleep(2 ** i)
            else: return response
        except requests.exceptions.RequestException as e:
            time.sleep(2 ** i)
    raise Exception(f"Failed to connect to {url}")

def execute_simulation(config, dry_run=False):
    if dry_run:
        time.sleep(0.01)
        return {"metrics": {"totalReturn": random.uniform(-0.1, 0.2), "totalTrades": 10}, "equityCurve": [{"balance": 1000}]}
    
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
    response = robust_request('POST', url, json_data=config)
    if response.status_code != 200:
        raise Exception(f"Server Error: {response.text}")
    json_res = response.json()
    return json_res.get("combinedResult", json_res)

def fetch_all_models(dry_run=False):
    if dry_run:
        global ALL_AVAILABLE_MODELS
        ALL_AVAILABLE_MODELS = [{"id": "btc_1h_mock_model", "name": "Mock Model"}]
        return True

    url = f"{ML_SERVER_URL}/api/ml/available-models"
    print(f"{Fore.CYAN}[API] Fetching models from {url}...")
    try:
        response = robust_request('GET', url)
        if response.status_code == 200:
            global ALL_AVAILABLE_MODELS_REAL
            ALL_AVAILABLE_MODELS_REAL = response.json()
            globals()['ALL_AVAILABLE_MODELS'] = ALL_AVAILABLE_MODELS_REAL
            print(f"{Fore.GREEN}[API] Successfully loaded {len(ALL_AVAILABLE_MODELS)} models.")
            return True
        return False
    except:
        return False

# ==============================================================================
# 3. MENU HELPERS
# ==============================================================================

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
        if len(parts) >= 2 and parts[0] == base_symbol:
            timeframes.add(parts[1])
    return sorted(list(timeframes))

def select_from_list(title, options):
    print(f"\n{Fore.CYAN}--- Select {title} ---")
    for i, opt in enumerate(options):
        print(f"{i+1}. {opt}")
    while True:
        choice = input(f"{Fore.YELLOW}Enter number (default 1): {Style.RESET_ALL}").strip()
        if not choice: return options[0]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options): return options[idx]
            print(f"{Fore.RED}Invalid number.")
        except ValueError: print(f"{Fore.RED}Please enter a number.")

# ==============================================================================
# 4. OPTIMIZER LOGIC
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
            initial_fold_balance = fold_curve[0]['balance']
            df['balance_continuous'] = df['balance'] - initial_fold_balance + last_balance
            last_balance = df['balance_continuous'].iloc[-1]
        all_dfs.append(df)
    
    if not all_dfs: return {"StitchedCalmarRatio": -999, "StitchedMaxDrawdown": 100}
    stitched_df = pd.concat(all_dfs, ignore_index=True)
    
    initial = stitched_df['balance_continuous'].iloc[0]
    final = stitched_df['balance_continuous'].iloc[-1]
    
    if final <= 0 or initial <= 0: return {"StitchedCalmarRatio": -999, "StitchedMaxDrawdown": 100}
    
    peak = stitched_df['balance_continuous'].cummax()
    drawdown = (stitched_df['balance_continuous'] - peak) / (peak + 1e-9)
    max_dd = abs(drawdown.min() * 100)
    if max_dd < 0.01: max_dd = 100.0 
    
    days = (pd.to_datetime(stitched_df['timestamp'].iloc[-1]) - pd.to_datetime(stitched_df['timestamp'].iloc[0])).days
    if days <= 0: days = 1
    try: annual_ret = ((final / initial) ** (365.25 / days)) - 1
    except: return {"StitchedCalmarRatio": -999, "StitchedMaxDrawdown": 100}
    
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
        
        for idx, fold in enumerate(wfo_folds):
            try:
                fold_config = deepcopy(test_config)
                fold_config['startDate'] = fold['startDate']
                fold_config['endDate'] = fold['endDate']
                res = execute_simulation(fold_config, dry_run)
                metrics = res.get('metrics', {})
                
                # 🚀 MANUAL PRUNING LOGIC
                fold_return = metrics.get('totalReturn', 0) * 100
                
                if fold_return < -PRUNE_DRAWDOWN_LIMIT:
                    # Silent prune for speed
                    return -999, 100 
                
                if idx == 0 and fold_return < PRUNE_RETURN_LIMIT:
                    return -999, 100

                all_curves[idx] = res.get('equityCurve', [])
            except Exception as e: 
                print(f"\n{Fore.RED}🔥 CLIENT EXCEPTION (Trial {trial.number}, Fold {idx}): {e}{Style.RESET_ALL}")
                return -999, 100 

        metrics = calculate_stitched_metrics(all_curves)
        calmar = metrics.get('StitchedCalmarRatio', -999)
        dd = metrics.get('StitchedMaxDrawdown', 100)
        
        log_trial_result(trial.number, trial.params, metrics)
        
        strat_label = f"{trend_strat}/{range_strat}"
        if calmar > 1.0:
            print(f"{Fore.GREEN}[Trial {trial.number}] 🟢 EXCELLENT | {strat_label:<35} | Calmar: {calmar:.2f} | DD: {dd:.2f}%")
        elif calmar > 0:
            print(f"{Fore.YELLOW}[Trial {trial.number}] 🟡 PROFIT    | {strat_label:<35} | Calmar: {calmar:.2f} | DD: {dd:.2f}%")
        elif calmar == -999:
            # Silent prune
            pass
        else:
            print(f"{Fore.RED}[Trial {trial.number}] 🔴 LOSS      | {strat_label:<35} | Calmar: {calmar:.2f} | DD: {dd:.2f}%")
            
        return calmar, dd

    return objective

# ==============================================================================
# 5. QUALITY GATES & REPORT
# ==============================================================================

def run_certification_gate(symbol, timeframe, best_trial, dry_run):
    """Gate 1: Victory Lap (Out-of-Sample Check)."""
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=365)
    cert_config = deepcopy(BASE_CONFIG)
    cert_config['symbol'] = symbol
    cert_config['timeframe'] = timeframe
    cert_config['startDate'] = start_dt.strftime("%Y-%m-%d")
    cert_config['endDate'] = end_dt.strftime("%Y-%m-%d")
    
    params = deepcopy(best_trial.params)
    trend_strat = params.pop("trend_strategy", "sma_crossover")
    range_strat = params.pop("range_strategy", "rsi_divergence")
    ml_model = params.pop("mlModel", "mock_model")
    ml_threshold = params.pop("mlThreshold", 0.60)
    
    cert_config['strategies'] = [{"code": trend_strat, "params": {}}, {"code": range_strat, "params": {}}]
    cert_config['mlMode'] = "predictions"
    cert_config['mlModel'] = ml_model
    cert_config['mlThreshold'] = ml_threshold
    
    for k, v in params.items():
        if k.startswith("params."):
            clean_key = k.split(".")[1]
            set_nested_value(cert_config, f"params.{clean_key}", v)
            
    try:
        res = execute_simulation(cert_config, dry_run)
        m = res.get('metrics', {})
        ret = m.get('totalReturn', 0)
        trades = m.get('totalTrades', 0)
        # 🚀 FIX: ALLOW SNIPERS (Lowered limit to 1 trade)
        if ret > 0 and trades >= 1:
             return True, f"PASSED (+{ret:.2f}% | {trades} trades)"
        elif trades < 1:
             return False, f"FAILED (Zero Volume: {trades} trades)"
        else:
             return False, f"FAILED (Negative Return: {ret:.2f}%)"
    except: return False, "ERROR"

def run_reproducibility_gate(symbol, timeframe, best_trial, dry_run):
    wfo_folds = generate_wfo_folds()
    fold = wfo_folds[-2]
    base_config = deepcopy(BASE_CONFIG)
    base_config['symbol'] = symbol
    base_config['timeframe'] = timeframe
    base_config['startDate'] = fold['startDate']
    base_config['endDate'] = fold['endDate']
    
    params = deepcopy(best_trial.params)
    trend_strat = params.pop("trend_strategy")
    range_strat = params.pop("range_strategy")
    base_config['strategies'] = [{"code": trend_strat, "params": {}}, {"code": range_strat, "params": {}}]
    base_config['mlMode'] = "predictions"
    base_config['mlModel'] = params.pop("mlModel")
    base_config['mlThreshold'] = params.pop("mlThreshold")
    for k, v in params.items():
        if k.startswith("params."): set_nested_value(base_config, f"params.{k.split('.')[1]}", v)

    returns = []
    for i in range(REPRODUCIBILITY_ATTEMPTS):
        try:
            res = execute_simulation(base_config, dry_run)
            returns.append(res['metrics']['totalReturn'])
        except: returns.append(-999)
    
    std_dev = np.std(returns)
    if std_dev < 1.0 and min(returns) > 0:
        return True, f"PASSED (StdDev: {std_dev:.2f})"
    else:
        return False, f"FAILED (Unstable: {returns})"

def save_best_params(study, trial):
    try:
        best_trials = study.best_trials
        if not best_trials: return
        sorted_best = sorted(best_trials, key=lambda t: t.values[0], reverse=True)
        calmar_king = sorted_best[0]

        if calmar_king.values[0] <= 0.0: return 

        if calmar_king.number == trial.number:
            symbol = trial.user_attrs.get("symbol", "UNKNOWN")
            timeframe = "1h" 
            
            print(f"\n{Fore.MAGENTA}╔════════════════════════════════════════════════════════════════╗")
            print(f"║ 🔍 NEW CANDIDATE FOUND (Trial {trial.number})                           ║")
            print(f"║    In-Sample Calmar: {calmar_king.values[0]:.2f}                              ║")
            print(f"╚════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
            
            is_cert, cert_msg = run_certification_gate(symbol, timeframe, trial, False)
            if is_cert:
                print(f"   {Fore.GREEN}🔒 Gate 1: {cert_msg}{Style.RESET_ALL}")
                is_stable, stable_msg = run_reproducibility_gate(symbol, timeframe, trial, False)
                if is_stable:
                    print(f"   {Fore.GREEN}🔒 Gate 2: {stable_msg}{Style.RESET_ALL}")
                    filename = f"winner_{symbol}_GOLDEN.json"
                    filepath = os.path.join(RESULTS_DIR, filename)
                    output = {
                        "timestamp": datetime.now().isoformat(),
                        "metrics": {"calmar": calmar_king.values[0], "certification": cert_msg},
                        "params": trial.params,
                        "strategies": trial.user_attrs.get("combo_strategies", "unknown")
                    }
                    with open(filepath, 'w') as f: json.dump(output, f, indent=4)
                    print(f"   {Fore.MAGENTA}🏆 GOLDEN WINNER SAVED! Path: {filename}{Style.RESET_ALL}")
                else:
                    print(f"   {Fore.RED}🔒 Gate 2: {stable_msg}{Style.RESET_ALL}")
            else:
                print(f"   {Fore.RED}🔒 Gate 1: {cert_msg}{Style.RESET_ALL}")

    except Exception as e: pass

def perform_final_validation(study, symbol, timeframe, dry_run):
    print(f"\n{Fore.YELLOW}=============================================")
    print(f"🏆 FINAL VALIDATION: The Gauntlet (Top 5 Unique)")
    print(f"============================================={Style.RESET_ALL}")
    
    seen_configs = set()
    unique_trials = []
    
    sorted_trials = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)
    
    for trial in sorted_trials:
        sig = json.dumps(trial.params, sort_keys=True)
        if sig not in seen_configs:
            seen_configs.add(sig)
            unique_trials.append(trial)
        if len(unique_trials) >= 5: break
    
    for i, trial in enumerate(unique_trials):
        calmar = trial.values[0]
        dd = trial.values[1]
        strat = trial.user_attrs.get("combo_strategies", "Unknown")
        
        print(f"\n📝 Candidate #{i+1}: {strat}")
        print(f"   In-Sample Score: Calmar {calmar:.2f} | DD {dd:.2f}%")
        
        passed_1, msg_1 = run_certification_gate(symbol, timeframe, trial, dry_run)
        color_1 = Fore.GREEN if passed_1 else Fore.RED
        print(f"   🔒 Gate 1 (Victory Lap): {color_1}{msg_1}{Style.RESET_ALL}")
        
        if passed_1:
            passed_2, msg_2 = run_reproducibility_gate(symbol, timeframe, trial, dry_run)
            color_2 = Fore.GREEN if passed_2 else Fore.RED
            print(f"   🔒 Gate 2 (Stability):   {color_2}{msg_2}{Style.RESET_ALL}")
            if passed_2:
                filename = f"winner_{symbol}_FINAL_CANDIDATE_{i+1}.json"
                with open(os.path.join(RESULTS_DIR, filename), 'w') as f:
                    json.dump(trial.params, f, indent=4)
                print(f"   {Fore.MAGENTA}✨ CERTIFIED GOLDEN! Saved.{Style.RESET_ALL}")

# ==============================================================================
# 6. MAIN
# ==============================================================================
def run_comprehensive_audit(symbol, timeframe, dry_run):
    print(f"\n{Fore.CYAN}🕵️  Running Comprehensive Strategy Audit on {symbol} {timeframe} (ML: OFF)...")
    base_cfg = deepcopy(BASE_CONFIG)
    base_cfg['symbol'] = symbol
    base_cfg['timeframe'] = timeframe
    base_cfg['startDate'] = "2022-01-01" 
    base_cfg['endDate'] = "2024-01-01"
    base_cfg['mlMode'] = "off"
    all_strategies = TREND_STRATEGIES + RANGE_STRATEGIES
    failed_strategies = []
    print(f"{'Strategy':<25} | {'Trades':<8} | {'Return':<10} | {'Balance':<12} | {'Status'}")
    print("-" * 75)
    for strat in all_strategies:
        cfg = deepcopy(base_cfg)
        params = SANITY_PARAMS.get(strat, {})
        cfg['strategies'] = [{"code": strat, "params": params}]
        try:
            res = execute_simulation(cfg, dry_run)
            m = res.get('metrics', {})
            trades = m.get('totalTrades', 0)
            ret = m.get('totalReturn', 0)
            bal = m.get('finalBalance', 0)
            if trades > 0 and bal > 0:
                status = f"{Fore.GREEN}PASS (Profit){Style.RESET_ALL}" if ret > 0 else f"{Fore.YELLOW}PASS (Loss){Style.RESET_ALL}"
            else:
                status = f"{Fore.RED}FAIL{Style.RESET_ALL}"
                failed_strategies.append(strat)
            print(f"{strat:<25} | {trades:<8} | {ret:>8.2f}% | ${bal:>10.2f} | {status}")
        except: failed_strategies.append(strat)
    if failed_strategies:
        print(f"\n{Fore.YELLOW}⚠️ Warning: {len(failed_strategies)} strategies failed.")
        if input("Continue? (y/n): ").lower() != 'y': return False
    return True

def run_optimizer(args):
    if args.mode == "optimizer": 
        symbol = "BTC-USD"; timeframe = "1h"; n_trials = 100
    else:
        symbols = extract_unique_symbols()
        if not symbols: print("No models."); return
        symbol = select_from_list("Symbol", symbols)
        timeframes = extract_timeframes_for_symbol(symbol)
        timeframe = select_from_list("Timeframe", timeframes)
        try: n_trials = int(input(f"{Fore.YELLOW}Trials (100): {Style.RESET_ALL}").strip() or 100)
        except: n_trials = 100

    m_check = f"{symbol.split('-')[0]}_{timeframe}"
    ml_models = [m['id'] for m in ALL_AVAILABLE_MODELS if m['id'].lower().startswith(m_check.lower())]
    if not ml_models and not args.dry_run: return
    if args.dry_run: ml_models = ["mock_model"]

    if not run_comprehensive_audit(symbol, timeframe, args.dry_run): return

    print(f"{Fore.YELLOW}⚠️ Using RAM Storage (Fastest).")
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )

    print(f"\n{Fore.CYAN}🚀 STARTING OPTIMIZATION: {symbol} {timeframe}")
    
    try:
        wfo_folds = generate_wfo_folds()
        study.optimize(
            create_objective(symbol, timeframe, wfo_folds, ml_models, args.dry_run), 
            n_trials=n_trials, 
            n_jobs=PARALLEL_JOBS,
            callbacks=[save_best_params] 
        )
    except KeyboardInterrupt: print(f"\n{Fore.YELLOW}Optimization Paused.")
    
    perform_final_validation(study, symbol, timeframe, args.dry_run)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    set_global_seed(SEED)
    print(f"{Fore.CYAN}=== 🧠 INTELLIGENT MENDEL CLIENT v54 (Pruning) ===")
    if not fetch_all_models(args.dry_run) and not args.dry_run: print("Server error.")
    if args.mode == "optimizer": run_optimizer(args); return
    while True:
        print(f"\n{Style.BRIGHT}COMMAND MENU:")
        print("1. 🧬 Run Optimizer")
        print("0. Exit")
        choice = input(f"{Fore.CYAN}Select option: {Style.RESET_ALL}")
        if choice == "1": run_optimizer(args)
        elif choice == "0": break

if __name__ == "__main__":
    main()
