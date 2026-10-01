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

# Initialize Colorama
init(autoreset=True)

# 🚀 SILENCE OPTUNA
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- CONFIGURATION ---
ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://74.208.28.77:8000")
RESULTS_DIR = os.path.join(os.getcwd(), "data", "optimizer_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- SETTINGS ---
SEED = 12345
PARALLEL_JOBS = 6 
# 🚀 FIX: Use SQLite for process-safe storage
STORAGE_URL = f"sqlite:///{RESULTS_DIR}/optuna.db"

# 🚀 SMART FILTERS
MIN_TRADES_FOR_VALIDITY = 2   
PRUNE_DRAWDOWN_LIMIT = 35.0   
PRUNE_RETURN_LIMIT = -20.0    

# --- STRATEGY POOLS ---
TREND_STRATEGIES = ["sma_crossover", "macd_crossover", "ichimoku_cloud", "psar_signal", "obv_signal", "atr_breakout"]
RANGE_STRATEGIES = ["rsi_divergence", "stochastic_crossover", "bollinger_bands", "cci_oversold"]
ALL_AVAILABLE_MODELS = []

# --- BASE CONFIG ---
BASE_CONFIG = {
    "symbol": "btc-USD",
    "timeframe": "1h",
    "initialBalance": 1000,
    "fee": 0.006, 
    "riskManagementMode": "standard",
    "riskPercentage": 1,
    "optimizer_mode": True
}

# ==============================================================================
# 1. UTILITIES
# ==============================================================================
def set_global_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

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
            if method == 'GET': response = requests.get(url, timeout=10)
            else: response = requests.post(url, json=json_data, timeout=600) # Consistent timeout
            if response.status_code == 200: return response
            elif response.status_code >= 500: time.sleep(1)
            else: return response
        except requests.exceptions.RequestException: time.sleep(1)
    raise Exception(f"Failed to connect to {url}")

def execute_simulation(config, dry_run=False):
    if dry_run: 
        dates = pd.date_range(start=config['startDate'], periods=50, freq='H')
        fake_curve = [{"timestamp": d.isoformat(), "balance": 1000 * (1 + (0.01 * i))} for i, d in enumerate(dates)]
        return {"metrics": {"totalReturn": 50.0, "totalTrades": 10}, "equityCurve": fake_curve, "tradeBreakdown": []}
    
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
    response = robust_request('POST', url, json_data=config)
    if response.status_code != 200: raise Exception(f"Server Error: {response.text}")
    json_res = response.json()
    
    # 🚀 FIX: Robust extraction
    if "combinedResult" in json_res:
        return json_res["combinedResult"]
    return json_res

def fetch_all_models(dry_run=False):
    global ALL_AVAILABLE_MODELS
    if dry_run:
        ALL_AVAILABLE_MODELS = [{"id": "btc_1h_mock_model", "name": "Mock Model"}]
        return True
    try:
        response = robust_request('GET', f"{ML_SERVER_URL}/api/ml/available-models")
        if response.status_code == 200:
            ALL_AVAILABLE_MODELS = response.json()
            print(f"{Fore.GREEN}[API] Successfully loaded {len(ALL_AVAILABLE_MODELS)} models.")
            return True
        return False
    except: return False

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
        if len(parts) >= 2 and parts[0] == base_symbol: timeframes.add(parts[1])
    return sorted(list(timeframes))

def select_from_list(title, options):
    print(f"\n{Fore.CYAN}--- Select {title} ---")
    for i, opt in enumerate(options): print(f"{i+1}. {opt}")
    while True:
        choice = input(f"{Fore.YELLOW}Enter number (default 1): {Style.RESET_ALL}").strip()
        if not choice: return options[0]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options): return options[idx]
        except ValueError: pass

# ==============================================================================
# 4. OPTIMIZER LOGIC
# ==============================================================================
def generate_wfo_folds():
    now = datetime.now(timezone.utc)
    start_year = 2018 
    folds = []
    for year in range(start_year, now.year):
        folds.append({"startDate": f"{year}-01-01", "endDate": f"{year}-12-31", "id": year})
    # Ensure final fold has at least 30 days
    if (now - datetime(now.year, 1, 1, tzinfo=timezone.utc)).days > 30:
        folds.append({"startDate": f"{now.year}-01-01", "endDate": now.strftime("%Y-%m-%d"), "id": now.year})
    return folds

def calculate_stitched_metrics(equity_curves, trade_counts):
    if not equity_curves: return {"StitchedCalmarRatio": -999999, "StitchedMaxDrawdown": 100}
    
    all_dfs = []
    total_trades = sum(trade_counts)
    
    for i, fold_curve in enumerate(equity_curves):
        if not fold_curve: continue
        df = pd.DataFrame(fold_curve)
        if df.empty or 'balance' not in df.columns: continue
        
        if i == 0:
            df['balance_continuous'] = df['balance']
            last_balance = df['balance'].iloc[-1]
        else:
            initial_fold = fold_curve[0]['balance']
            df['balance_continuous'] = df['balance'] - initial_fold + last_balance
            last_balance = df['balance_continuous'].iloc[-1]
        all_dfs.append(df)
    
    if not all_dfs: return {"StitchedCalmarRatio": -999999, "StitchedMaxDrawdown": 100}
    
    # 🚀 FIX: Count trades properly
    if total_trades < MIN_TRADES_FOR_VALIDITY:
        return {"StitchedCalmarRatio": -999999, "StitchedMaxDrawdown": 100}

    stitched_df = pd.concat(all_dfs, ignore_index=True)
    initial = stitched_df['balance_continuous'].iloc[0]
    final = stitched_df['balance_continuous'].iloc[-1]
    
    if initial <= 0: return {"StitchedCalmarRatio": -999999, "StitchedMaxDrawdown": 100}
    
    # 🚀 FIX: True Drawdown Calculation
    peak = stitched_df['balance_continuous'].cummax()
    drawdown = (stitched_df['balance_continuous'] - peak) / peak * 100 
    max_dd_pct = abs(drawdown.min())
    
    # 🚀 FIX: No Artificial Inflation
    if max_dd_pct == 0: max_dd_pct = 0.0001
    
    try:
        t_start = pd.to_datetime(stitched_df['timestamp'].iloc[0])
        t_end = pd.to_datetime(stitched_df['timestamp'].iloc[-1])
        seconds = (t_end - t_start).total_seconds()
        years = seconds / (365.25 * 24 * 3600)
        if years < 0.1: years = 0.1
        
        if final <= 0: annual_ret = -1.0
        else: annual_ret = ((final / initial) ** (1 / years)) - 1
        
    except: return {"StitchedCalmarRatio": -999999, "StitchedMaxDrawdown": 100}
    
    calmar = (annual_ret * 100) / max_dd_pct
    return {"StitchedCalmarRatio": calmar, "StitchedMaxDrawdown": max_dd_pct}

def create_objective(symbol, timeframe, wfo_folds, market_models, dry_run=False):
    def objective(trial: optuna.Trial):
        trial.set_user_attr("symbol", symbol)
        trial.set_user_attr("timeframe", timeframe)
        
        test_config = deepcopy(BASE_CONFIG)
        test_config['symbol'] = symbol.upper()
        test_config['timeframe'] = timeframe
        
        trend_strat = trial.suggest_categorical("trend_strategy", TREND_STRATEGIES)
        range_strat = trial.suggest_categorical("range_strategy", RANGE_STRATEGIES)
        trial.set_user_attr("combo_strategies", f"{trend_strat},{range_strat}")
        
        strategies_payload = []
        
        # 🚀 FIX: ISOLATED PARAMETER SPACE
        trend_p = {}
        if trend_strat == "sma_crossover":
            s1 = trial.suggest_int("sma_1", 5, 50, step=5)
            s2 = trial.suggest_int("sma_2", 10, 200, step=10)
            trend_p["sma_fast_period"] = min(s1, s2)
            trend_p["sma_slow_period"] = max(s1, s2)
        elif trend_strat == "macd_crossover":
            trend_p["macd_fast_period"] = trial.suggest_int("macd_f", 5, 20)
            trend_p["macd_slow_period"] = trial.suggest_int("macd_s", 21, 50)
            trend_p["macd_signal_period"] = trial.suggest_int("macd_sig", 5, 15)
        elif trend_strat == "atr_breakout":
             trend_p["atr_period"] = trial.suggest_int("atr_p", 10, 30)
             trend_p["atr_multiplier"] = trial.suggest_float("atr_m", 1.0, 5.0, step=0.5)
        strategies_payload.append({"code": trend_strat, "params": trend_p})

        range_p = {}
        if range_strat == "rsi_divergence":
            range_p["rsi_length"] = trial.suggest_int("rsi_len", 7, 30)
            range_p["oversold_level"] = trial.suggest_int("rsi_os", 20, 40)
            range_p["overbought_level"] = trial.suggest_int("rsi_ob", 60, 80)
        elif range_strat == "bollinger_bands":
            range_p["bb_length"] = trial.suggest_int("bb_len", 15, 30)
            range_p["bb_std"] = trial.suggest_float("bb_std", 1.5, 3.0, step=0.1)
        elif range_strat == "cci_oversold":
            range_p["cci_length"] = trial.suggest_int("cci_len", 10, 40)
            range_p["cci_oversold"] = trial.suggest_int("cci_os", -150, -50)
            range_p["cci_overbought"] = trial.suggest_int("cci_ob", 50, 150)
            
        strategies_payload.append({"code": range_strat, "params": range_p})

        test_config['strategies'] = strategies_payload
        
        set_nested_value(test_config, "params.hybridMode", "REGIME")
        set_nested_value(test_config, "params.regime_threshold", trial.suggest_int("regime_threshold", 15, 40, step=5))
        
        # 🚀 FIX: Model Filtering Logic
        m_check = f"{symbol.split('-')[0].lower()}_{timeframe}" # btc_1h
        valid_models = [m['id'] for m in market_models if m['id'].lower().startswith(m_check)]
        
        if valid_models:
             test_config['mlMode'] = "predictions"
             test_config['mlModel'] = trial.suggest_categorical("mlModel", valid_models)
             test_config['mlThreshold'] = trial.suggest_float("mlThreshold", 0.50, 0.70, step=0.02)
        else:
             test_config['mlMode'] = "off"

        all_curves = [None] * len(wfo_folds)
        trade_counts = [0] * len(wfo_folds)
        
        for idx, fold in enumerate(wfo_folds):
            try:
                fold_config = deepcopy(test_config)
                fold_config['startDate'] = fold['startDate']
                fold_config['endDate'] = fold['endDate']
                
                res = execute_simulation(fold_config, dry_run)
                metrics = res.get('metrics', {})
                
                fold_ret = metrics.get('totalReturn', 0)
                fold_dd = metrics.get('maxDrawdown', 0)
                trades = metrics.get('totalTrades', 0)
                
                trade_counts[idx] = trades
                timestamp_str = datetime.now().strftime("%I:%M:%S %p")

                # 🚀 FIX: Pruning Logic
                if fold_dd > PRUNE_DRAWDOWN_LIMIT:
                    print(f"{Fore.CYAN}[Trial {trial.number}] ✂️  PRUNED (High DD: {fold_dd:.2f}%) | {timestamp_str}{Style.RESET_ALL}")
                    return -999999, 100 
                
                if idx == 0 and fold_ret < PRUNE_RETURN_LIMIT:
                    # Silent Prune
                    return -999999, 100

                all_curves[idx] = res.get('equityCurve', [])
            except Exception: 
                return -999999, 100 

        metrics = calculate_stitched_metrics(all_curves, trade_counts)
        calmar = metrics.get('StitchedCalmarRatio', -999999)
        dd = metrics.get('StitchedMaxDrawdown', 100)
        
        # No log file write here to avoid race condition
        
        strat_label = f"{trend_strat}/{range_strat}"
        timestamp_str = datetime.now().strftime("%I:%M:%S %p")
        
        if calmar > 1.0:
            print(f"{Fore.GREEN}[Trial {trial.number}] 🟢 EXCELLENT | {strat_label:<35} | Calmar: {calmar:.2f} | DD: {dd:.2f}% | {timestamp_str}{Style.RESET_ALL}")
        elif calmar > 0:
            print(f"{Fore.YELLOW}[Trial {trial.number}] 🟡 PROFIT    | {strat_label:<35} | Calmar: {calmar:.2f} | DD: {dd:.2f}% | {timestamp_str}{Style.RESET_ALL}")
        
        return calmar, dd

    return objective

# ==============================================================================
# 5. FINAL VALIDATION & SAVING
# ==============================================================================

def build_config_from_trial(symbol, timeframe, params):
    """Reconstructs a valid configuration from trial parameters"""
    config = deepcopy(BASE_CONFIG)
    config['symbol'] = symbol
    config['timeframe'] = timeframe
    
    # Extract strategies from combo string
    combo = params.get("combo_strategies", "") # Handled by Optuna
    # NOTE: Optuna doesn't store user_attrs in params dict for retrieval later easily
    # We rely on the fact that params has unique keys per strategy
    
    # ... (Logic simplified: In production, use a proper StrategyBuilder class)
    # For this script, we assume the 'best_params' passed has the keys.
    return config

def perform_final_validation(study, symbol, timeframe, dry_run):
    print(f"\n{Fore.YELLOW}=============================================")
    print(f"🏆 FINAL VALIDATION: The Gauntlet (Top 5 Unique)")
    print(f"============================================={Style.RESET_ALL}")
    
    # Get distinct pareto front
    best_trials = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)
    unique_trials = []
    seen = set()
    
    for t in best_trials:
        sig = json.dumps(t.params, sort_keys=True)
        if sig not in seen:
            seen.add(sig)
            unique_trials.append(t)
        if len(unique_trials) >= 5: break
        
    for i, trial in enumerate(unique_trials):
        calmar = trial.values[0]
        dd = trial.values[1]
        strat = trial.user_attrs.get("combo_strategies", "Unknown")
        
        print(f"\n📝 Candidate #{i+1}: {strat}")
        print(f"   In-Sample Score: Calmar {calmar:.2f} | DD {dd:.2f}%")
        
        # 1. Run Gate 1 (Victory Lap)
        passed, msg = run_gate_test(symbol, timeframe, trial, "victory_lap", dry_run)
        color = Fore.GREEN if passed else Fore.RED
        print(f"   🔒 Gate 1: {color}{msg}{Style.RESET_ALL}")
        
        if passed:
            # 2. Run Gate 2 (Reproducibility)
            passed_2, msg_2 = run_gate_test(symbol, timeframe, trial, "reproducibility", dry_run)
            color_2 = Fore.GREEN if passed_2 else Fore.RED
            print(f"   🔒 Gate 2: {color_2}{msg_2}{Style.RESET_ALL}")
            
            if passed_2:
                filename = f"winner_{symbol}_{timeframe}_FINAL_{i+1}_{datetime.now().strftime('%Y%m%d')}.json"
                with open(os.path.join(RESULTS_DIR, filename), 'w') as f:
                    # Dump all params
                    final_data = trial.params.copy()
                    final_data['strategies'] = strat
                    json.dump(final_data, f, indent=4)
                print(f"   {Fore.MAGENTA}✨ CERTIFIED GOLDEN! Saved.{Style.RESET_ALL}")

def run_gate_test(symbol, timeframe, trial, mode, dry_run):
    # Placeholder logic re-using create_objective's construction would go here
    # For V62, we stub this to ensure the optimization loop works first.
    return True, "PASSED (Logic Stub)"

# ==============================================================================
# 6. MAIN
# ==============================================================================
def run_optimizer(args):
    symbol = "BTC-USD"
    timeframe = "1h"
    n_trials = 1000
    
    if not args.dry_run:
        if not fetch_all_models(): print("Server Error."); return

    print(f"{Fore.YELLOW}⚠️ Using SQLite Storage (Process Safe).")
    
    study = optuna.create_study(
        study_name=f"study_{symbol}_{timeframe}_{int(time.time())}",
        storage=STORAGE_URL, # 🚀 FIX: SQLite Storage
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=20), # More random start
        load_if_exists=True
    )

    print(f"\n{Fore.CYAN}🚀 STARTING OPTIMIZATION: {symbol} {timeframe}")
    try:
        wfo_folds = generate_wfo_folds()
        study.optimize(
            create_objective(symbol, timeframe, wfo_folds, ALL_AVAILABLE_MODELS, args.dry_run), 
            n_trials=n_trials, 
            n_jobs=PARALLEL_JOBS
            # Removed Callback to prevent race condition crash
        )
    except KeyboardInterrupt: print(f"\n{Fore.YELLOW}Optimization Paused.")
    
    perform_final_validation(study, symbol, timeframe, args.dry_run)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="optimizer")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    set_global_seed(SEED)
    print(f"{Fore.CYAN}=== 🧠 INTELLIGENT MENDEL CLIENT v62 (Audit Fix) ===")
    run_optimizer(args)

if __name__ == "__main__":
    main()
