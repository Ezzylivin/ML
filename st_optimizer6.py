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
import uuid

init(autoreset=True)
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://127.0.0.1:8000")
RESULTS_DIR = os.path.join(os.getcwd(), "data", "optimizer_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- REPRODUCIBILITY SETUP ---
SEED = 54321
random.seed(SEED)
np.random.seed(SEED)

PARALLEL_JOBS = 3
TIMEOUT_SECONDS = 600 
MIN_TRADES_TOTAL = 20           
PRUNE_DRAWDOWN_LIMIT = 30.0     
MAX_SCORE_CAP = 50.0            

TREND_POOL = ["sma_crossover", "macd_crossover", "ichimoku_cloud", "psar_signal", "obv_signal", "atr_breakout"]
RANGE_POOL = ["rsi_divergence", "stochastic_crossover", "bollinger_bands", "cci_oversold"]
ALL_AVAILABLE_MODELS = []

BASE_CONFIG = {
    "symbol": "BTC-USD",
    "timeframe": "1h",
    "initialBalance": 1000,
    "fee": 0.006,
    "riskManagementMode": "standard", 
    "growthCapitalTarget": 0,           
    "optimizer_mode": True,
    "params": {}                
}

# --- UTILS ---
def print_header(text): print(f"\n{Fore.CYAN}{'='*60}\n {text.center(58)} \n{'='*60}{Style.RESET_ALL}")
def select_from_list(title, options):
    print(f"\n{Fore.YELLOW}>>> Select {title}:{Style.RESET_ALL}")
    for i, opt in enumerate(options): print(f"  {Fore.CYAN}{i+1}.{Style.RESET_ALL} {opt}")
    while True:
        try:
            idx = int(input(f"\n{Fore.GREEN}Enter number (default 1): {Style.RESET_ALL}").strip() or 1) - 1
            if 0 <= idx < len(options): return options[idx]
        except: pass

def robust_request(method, url, json_data=None, retries=3):
    for i in range(retries):
        try:
            if method == 'GET': r = requests.get(url, timeout=15)
            else: r = requests.post(url, json=json_data, timeout=600)
            if r.status_code == 200: return r
            elif r.status_code >= 500: time.sleep(1)
            else: return r
        except: time.sleep(1)
    raise Exception(f"Failed: {url}")

def fetch_all_models(dry_run=False):
    global ALL_AVAILABLE_MODELS
    if dry_run:
        ALL_AVAILABLE_MODELS = [{"id": "btc_1h_mock", "name": "Mock"}]
        return True
    try:
        print_header("CONNECTING TO ML SERVER")
        r = robust_request('GET', f"{ML_SERVER_URL}/api/ml/available-models")
        if r.status_code == 200:
            ALL_AVAILABLE_MODELS = r.json()
            return True
        return False
    except: return False

def execute_simulation(config, dry_run=False):
    if dry_run: return {"metrics": {"totalReturn": 5.0, "totalTrades": 40, "maxDrawdown": 2.0, "profitFactor": 1.5}, "tradeBreakdown": []}
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
    r = robust_request('POST', url, json_data=config)
    if r.status_code != 200: raise Exception(f"Server Error: {r.text}")
    j = r.json()
    return j.get("combinedResult", j)

def set_nested(d, keys, value):
    keys = keys.split('.')
    for key in keys[:-1]: d = d.setdefault(key, {})
    d[keys[-1]] = value

def format_strategy_display(strategies):
    display_parts = []
    for s in strategies:
        code = s['code'].split('_')[0].upper()
        p_str = ",".join([str(v) for k,v in s['params'].items()][:1])
        display_parts.append(f"{code}({p_str})")
    return "+".join(display_parts)

# 🚀 CORRECTED: 3-Way Split (Train / Validation / Holdout)
def get_train_test_dates(years_back=2):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=365*years_back)
    total_days = (now - start).days
    
    # 70% Train (Optimizer Learning)
    # 15% Validation (Optimizer Scoring)
    # 15% Holdout (Final Exam - Invisible to Optimizer)
    train_days = int(total_days * 0.70)
    val_days = int(total_days * 0.15)
    
    split_1 = start + timedelta(days=train_days)
    split_2 = split_1 + timedelta(days=val_days)
    
    return {
        "train_start": start.strftime("%Y-%m-%d"),
        "train_end": (split_1 - timedelta(days=1)).strftime("%Y-%m-%d"),
        "val_start": split_1.strftime("%Y-%m-%d"),
        "val_end": (split_2 - timedelta(days=1)).strftime("%Y-%m-%d"),
        "holdout_start": split_2.strftime("%Y-%m-%d"),
        "holdout_end": now.strftime("%Y-%m-%d")
    }

def calculate_stability_score(res):
    metrics = res.get('metrics', {})
    ret = metrics.get('totalReturn', 0)
    dd = abs(metrics.get('maxDrawdown', 100))
    pf = metrics.get('profitFactor', 0)
    if dd == 0: dd = 0.1 
    score = (ret * pf) / dd
    if score > MAX_SCORE_CAP: score = MAX_SCORE_CAP
    return score

# 🚀 CERTIFICATION: Uses Trade Sequence Permutation
def monte_carlo_certification(trades, iterations=1000):
    if not trades or len(trades) < 5: return 50.0 
    profits = [t['profit'] for t in trades]
    ruin_count = 0
    for _ in range(iterations):
        random.shuffle(profits)
        balance = 1000
        peak = 1000
        for p in profits:
            balance += p
            if balance > peak: peak = balance
            dd = (peak - balance) / peak * 100
            if dd > PRUNE_DRAWDOWN_LIMIT:
                ruin_count += 1
                break
    return (ruin_count / iterations) * 100

# 🚀 WFO: Correctly accepts dynamic dates to avoid leakage
def walk_forward_validation(config, start_date, end_date, windows=3):
    print(f"   {Fore.YELLOW}Running WFO ({start_date} to {end_date})...{Style.RESET_ALL}")
    
    s_dt = datetime.strptime(start_date, "%Y-%m-%d")
    e_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (e_dt - s_dt).days
    window_days = total_days // windows
    
    # FIX: Return False if too short, to prevent false positives
    if window_days < 5: 
        print(f"{Fore.RED}   WFO Skipped (Window too short){Style.RESET_ALL}")
        return False 
    
    passed_windows = 0
    for i in range(windows):
        w_start = s_dt + timedelta(days=i * window_days)
        w_end = w_start + timedelta(days=window_days)
        
        w_config = deepcopy(config)
        w_config['startDate'] = w_start.strftime("%Y-%m-%d")
        w_config['endDate'] = w_end.strftime("%Y-%m-%d")
        
        try:
            res = execute_simulation(w_config)
            ret = res.get('metrics', {}).get('totalReturn', 0)
            if ret > 0: passed_windows += 1
        except: pass
        
    print(f"   {Fore.CYAN}WFO Result: {passed_windows}/{windows} passed.{Style.RESET_ALL}")
    return passed_windows >= (windows - 1)

def verify_holdout_performance(config, dates):
    print(f"\n{Fore.CYAN}   >>> 🔒 Testing Hidden Holdout ({dates['holdout_start']} - {dates['holdout_end']})...{Style.RESET_ALL}")
    holdout_config = deepcopy(config)
    holdout_config['startDate'] = dates['holdout_start']
    holdout_config['endDate'] = dates['holdout_end']
    try:
        res = execute_simulation(holdout_config)
        m = res.get('metrics', {})
        return m.get('totalReturn', 0), m.get('maxDrawdown', 0), m.get('totalTrades', 0), res.get('tradeBreakdown', [])
    except: return -999, 100, 0, []


def save_winner(config, ret, dd, trades, rank_score, note=""):
    uid = str(uuid.uuid4())[:4]
    fname = f"std_synergy_{config['symbol']}_{config['timeframe']}_SCORE{int(rank_score)}_{int(time.time())}_{uid}.json"
    
    final_data = {
        "symbol": config['symbol'],
        "timeframe": config['timeframe'],
        "strategies": config['strategies'],
        "params": config['params'],
        "mlMode": config.get('mlMode'),
        "mlModel": config.get('mlModel'),
        "mlThreshold": config.get('mlThreshold'),
        "riskManagementMode": "standard", 
        "riskPercentage": config['riskPercentage'],
        "metrics": {"totalReturn": ret, "maxDrawdown": dd, "totalTrades": trades, "score": rank_score},
        "verified": True,
        "validation": "3_way_split_plus_wfo",
        "note": note,
        "timestamp": datetime.now().isoformat()
    }
    
    temp_path = os.path.join(RESULTS_DIR, f".tmp_{fname}")
    final_path = os.path.join(RESULTS_DIR, fname)
    
    with open(temp_path, 'w') as f: 
        json.dump(final_data, f, indent=2)
    os.rename(temp_path, final_path)
    
    print(f"{Fore.GREEN}   💾 FORTRESS SAVE: {fname}{Style.RESET_ALL}")

# --- SWARM LOGIC ---
def get_strategy_params(trial, strat_code, idx):
    p = {}
    if strat_code == "sma_crossover":
        s1 = trial.suggest_int(f"s{idx}_sma1", 5, 60, 5)
        s2 = trial.suggest_int(f"s{idx}_sma2", 20, 200, 10)
        p = {"sma_fast_period": min(s1,s2), "sma_slow_period": max(s1,s2)}
    elif strat_code == "macd_crossover":
        p = {"macd_fast_period": trial.suggest_int(f"s{idx}_mf", 8, 20), "macd_slow_period": trial.suggest_int(f"s{idx}_ms", 21, 50), "macd_signal_period": trial.suggest_int(f"s{idx}_msig", 5, 15)}
    elif strat_code == "atr_breakout":
        p = {"atr_period": trial.suggest_int(f"s{idx}_atrp", 10, 30), "atr_multiplier": trial.suggest_float(f"s{idx}_atrm", 1.5, 5.0, step=0.1)}
    elif strat_code == "psar_signal":
        p = {"psar_step": trial.suggest_float(f"s{idx}_psar", 0.01, 0.05, step=0.01)}
    elif strat_code == "ichimoku_cloud": pass
    elif strat_code == "obv_signal": p = {"obv_ma_period": trial.suggest_int(f"s{idx}_obv", 10, 50)}
    elif strat_code == "rsi_divergence":
        p = {"rsi_length": trial.suggest_int(f"s{idx}_rsi_len", 10, 30), "oversold_level": trial.suggest_int(f"s{idx}_rsi_os", 20, 45), "overbought_level": trial.suggest_int(f"s{idx}_rsi_ob", 55, 80)}
    elif strat_code == "bollinger_bands":
        p = {"bb_length": trial.suggest_int(f"s{idx}_bb_len", 15, 30), "bb_std": trial.suggest_float(f"s{idx}_bb_std", 1.5, 3.0, step=0.1)}
    elif strat_code == "stochastic_crossover":
        p = {"k_period": trial.suggest_int(f"s{idx}_k", 14, 30), "d_period": 3}
    elif strat_code == "cci_oversold":
        p = {"cci_length": trial.suggest_int(f"s{idx}_cci_len", 14, 40), "cci_oversold": -100, "cci_overbought": 100}
    return {"code": strat_code, "params": p}

def create_objective(symbol, timeframe, dates, market_models, dry_run=False):
    def objective(trial: optuna.Trial):
        test_config = deepcopy(BASE_CONFIG)
        test_config['symbol'] = symbol.upper()
        test_config['timeframe'] = timeframe
        
        n_trend = trial.suggest_int("n_trend", 0, 2)
        n_range = trial.suggest_int("n_range", 0, 2)
        if n_trend + n_range == 0: n_trend = 1
        
        selected_strategies = []
        for i in range(n_trend):
            s_name = trial.suggest_categorical(f"t_strat_{i}", TREND_POOL)
            selected_strategies.append(get_strategy_params(trial, s_name, f"t{i}"))
        for i in range(n_range):
            s_name = trial.suggest_categorical(f"r_strat_{i}", RANGE_POOL)
            selected_strategies.append(get_strategy_params(trial, s_name, f"r{i}"))

        test_config['strategies'] = selected_strategies
        
        hybrid = trial.suggest_categorical("hybrid", ["OR", "AND", "REGIME"])
        set_nested(test_config, "params.hybridMode", hybrid)
        if hybrid == "REGIME": set_nested(test_config, "params.regime_threshold", trial.suggest_int("regime_thresh", 15, 35, step=5))

        test_config['riskPercentage'] = trial.suggest_int("risk_pct", 5, 20) 
        test_config['maxPyramiding'] = trial.suggest_int("pyramiding", 1, 5) 
        set_nested(test_config, "params.maxPyramiding", test_config['maxPyramiding'])
        
        # 🚀 UPGRADE: FULL SPECTRUM PARAMETERS
        set_nested(test_config, "params.tslAtrMult", trial.suggest_float("tsl_mult", 1.5, 5.0, step=0.5))
        set_nested(test_config, "params.minAdxLevel", trial.suggest_int("min_adx", 0, 30, step=5))
        set_nested(test_config, "params.minAtrPct", trial.suggest_float("min_atr", 0.0, 0.3, step=0.05))
        set_nested(test_config, "params.trendFilterPeriod", trial.suggest_categorical("trend_filt", [0, 50, 200]))
        
        if market_models:
             test_config['mlMode'] = "predictions"
             test_config['mlModel'] = trial.suggest_categorical("mlModel", market_models)
             test_config['mlThreshold'] = trial.suggest_float("mlThreshold", 0.50, 0.70, step=0.05)
        else: test_config['mlMode'] = "off"

        swarm_str = format_strategy_display(selected_strategies)
        print(f"\n{Fore.WHITE}[Trial {trial.number}] Synergy ({hybrid}): {swarm_str}...{Style.RESET_ALL}", end="\r", flush=True)

        try:
            # PHASE 1: TRAINING (70% Data) - Optimization Target
            train_config = deepcopy(test_config)
            train_config['startDate'] = dates['train_start']
            train_config['endDate'] = dates['train_end']
            res_train = execute_simulation(train_config, dry_run)
            
            t_score = calculate_stability_score(res_train)
            t_trades = res_train.get('metrics', {}).get('totalTrades', 0)

            if t_trades < MIN_TRADES_TOTAL: 
                return (t_trades / MIN_TRADES_TOTAL) * 0.5 
            
            # 🔥 FIX: LOOSENED FILTER from 1.0 to 0.0 to prevent early rejection
            if t_score < 0.0: 
                print(f"{Fore.CYAN}[Trial {trial.number}] 💤 TRAIN WEAK | {swarm_str} ({hybrid}) | TrainScore: {t_score:.2f}{Style.RESET_ALL}")
                return t_score 

            # PHASE 2: VALIDATION (15% Data) - Selection Criteria
            val_config = deepcopy(test_config)
            val_config['startDate'] = dates['val_start']
            val_config['endDate'] = dates['val_end']
            res_val = execute_simulation(val_config, dry_run)
            
            v_score = calculate_stability_score(res_val)
            v_dd = abs(res_val.get('metrics', {}).get('maxDrawdown', 100))
            v_ret = res_val.get('metrics', {}).get('totalReturn', 0)
            v_calmar = v_ret / (v_dd if v_dd > 0 else 0.1)

            if v_score < 1.0 or v_dd > PRUNE_DRAWDOWN_LIMIT:
                print(f"{Fore.WHITE}[Trial {trial.number}] ❌ VAL FAIL | Ret: {v_ret:.2f}% | DD: {v_dd:.2f}% | {swarm_str}{Style.RESET_ALL}")
                return v_score * 0.5 

            print(f"{Fore.GREEN}[Trial {trial.number}] 🛡️ VAL PASS | Ret: {v_ret:.2f}% | DD: {v_dd:.2f}% | Calmar: {v_calmar:.2f} | {swarm_str}{Style.RESET_ALL}")
            
            # 🚀 PHASE 3: FORTRESS VALIDATION (15% Holdout - HIDDEN FROM OPTIMIZER)
            if v_score > 3.0:
                h_ret, h_dd, h_trades, h_trade_list = verify_holdout_performance(test_config, dates)
                
                if h_ret > 0:
                    ruin_prob = monte_carlo_certification(h_trade_list)
                    
                    if ruin_prob < 10.0:
                        if walk_forward_validation(test_config, dates['holdout_start'], dates['holdout_end'], windows=2):
                            save_winner(test_config, h_ret, h_dd, h_trades, v_score, note="Passed 3-Stage Fortress")

            return v_score
        except: return -999
    return objective

class EarlyStoppingCallback:
    def __init__(self, patience=1000): 
        self.patience = patience
        self.best_value = -float('inf')
        self.stagnant = 0
    def __call__(self, study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            if trial.value > self.best_value:
                self.best_value = trial.value
                self.stagnant = 0
            else: self.stagnant += 1
            if self.stagnant >= self.patience: 
                print(f"{Fore.YELLOW}🛑 Patience Limit Reached ({self.patience} trials).{Style.RESET_ALL}")
                study.stop()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not fetch_all_models(args.dry_run): return

    print_header("INCOME SYNERGY OPTIMIZER (NO LEAKAGE EDITION)")
    symbols = list(set([m['id'].split('_')[0].upper() + "-USD" for m in ALL_AVAILABLE_MODELS]))
    if not symbols: symbols = ["BTC-USD", "ETH-USD"]
    
    sym = select_from_list("Symbol", sorted(symbols))
    tf = select_from_list("Timeframe", ["1h", "4h", "15m"])
    try: n_trials = int(input(f"\n{Fore.YELLOW}Enter Max Trials (Default 5000): {Style.RESET_ALL}").strip() or 5000)
    except: n_trials = 5000
    
    dates = get_train_test_dates(years_back=2)
    
    # FIX: Split complex list comprehension into readable lines
    models = [m['id'] for m in ALL_AVAILABLE_MODELS if sym.split('-')[0].lower() in m['id'].lower() and tf in m['id'].lower()]
    if not models and ALL_AVAILABLE_MODELS:
        models = [ALL_AVAILABLE_MODELS[0]['id']]

    # 🚀 REPRODUCIBLE SAMPLER (FIXED)
    # Added n_startup_trials=1000 and multivariate=True to force exploration
    study = optuna.create_study(
        direction="maximize", 
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5000)
    )
    
    try: 
        study.optimize(
            create_objective(sym, tf, dates, models, args.dry_run), 
            n_trials=n_trials, 
            n_jobs=PARALLEL_JOBS
        )
    except KeyboardInterrupt: print("Paused.")

if __name__ == "__main__": main()
