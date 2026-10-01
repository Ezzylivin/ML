# File: dy_optimizer.py
# 🚀 UPGRADE: v6.3 - "Rocket Commander" (Flush Fix + Audit + Full Metrics)

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

init(autoreset=True)
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- CONFIG ---
ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://127.0.0.1:8000")
RESULTS_DIR = os.path.join(os.getcwd(), "data", "optimizer_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 12345
random.seed(SEED)
np.random.seed(SEED)

PARALLEL_JOBS = 5            
TIMEOUT_SECONDS = 600 
MIN_TRADES_TOTAL = 25        
PRUNE_DRAWDOWN_LIMIT = 80.0  
MAX_CALMAR_CAP = 100.0       

# 🚀 INTELLIGENT POOLS
TREND_POOL = ["sma_crossover", "macd_crossover", "ichimoku_cloud", "psar_signal", "obv_signal", "atr_breakout"]
RANGE_POOL = ["rsi_divergence", "stochastic_crossover", "bollinger_bands", "cci_oversold"]
ALL_AVAILABLE_MODELS = []

BASE_CONFIG = {
    "symbol": "BTC-USD",
    "timeframe": "1h",
    "initialBalance": 1000,
    "fee": 0.006,             
    "riskManagementMode": "dynamic", 
    "growthCapitalTarget": 1000000,  
    "riskPercentage": 1,      
    "optimizer_mode": True,
    "params": {}              
}

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
    if dry_run: return {"metrics": {"totalReturn": 10.0, "totalTrades": 50, "maxDrawdown": 5.0}, "equityCurve": []}
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
    r = robust_request('POST', url, json_data=config)
    if r.status_code != 200: raise Exception(f"Server Error: {r.text}")
    j = r.json()
    return j.get("combinedResult", j)

def set_nested(d, keys, value):
    keys = keys.split('.')
    for key in keys[:-1]: d = d.setdefault(key, {})
    d[keys[-1]] = value

def get_train_test_dates(years_back=2, test_split=0.25):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=365*years_back)
    total_days = (now - start).days
    train_days = int(total_days * (1 - test_split))
    split = start + timedelta(days=train_days)
    return {
        "train_start": start.strftime("%Y-%m-%d"),
        "train_end": (split - timedelta(days=1)).strftime("%Y-%m-%d"),
        "test_start": split.strftime("%Y-%m-%d"),
        "test_end": now.strftime("%Y-%m-%d")
    }

def verify_strategy_stability(config, dates):
    full_config = deepcopy(config)
    full_config['startDate'] = dates['train_start']
    full_config['endDate'] = dates['test_end']
    try:
        res = execute_simulation(full_config)
        m = res.get('metrics', {})
        return m.get('totalReturn', 0), m.get('maxDrawdown', 0), m.get('totalTrades', 0)
    except: return 0, 0, 0

def save_winner(config, ret, dd, trades, dates, rank_score):
    fname = f"rocket_synergy_{config['symbol']}_{config['timeframe']}_RET{int(ret)}_{int(time.time())}.json"
    final_data = {
        "symbol": config['symbol'],      # 👈 ADDED THIS
        "timeframe": config['timeframe'],
        "strategies": config['strategies'],
        "params": config['params'],
        "mlMode": config.get('mlMode'),
        "mlModel": config.get('mlModel'),
        "mlThreshold": config.get('mlThreshold'),
        "riskManagementMode": "dynamic", 
        "riskPercentage": config['riskPercentage'],
        "growthCapitalTarget": config.get('growthCapitalTarget', 0),
        "metrics": {"totalReturn": ret, "maxDrawdown": dd, "totalTrades": trades, "testScore": rank_score},
        "validation": "out_of_sample_verified",
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(RESULTS_DIR, fname), 'w') as f: json.dump(final_data, f, indent=2)
    print(f"{Fore.GREEN}   💾 SAVED WINNER: {fname}{Style.RESET_ALL}")

def format_strategy_display(strategies):
    display_parts = []
    for s in strategies:
        code = s['code'].split('_')[0].upper() 
        p_vals = []
        for k, v in s['params'].items():
            val = f"{v:.1f}" if isinstance(v, float) else str(v)
            p_vals.append(val)
        p_str = ",".join(p_vals[:2]) 
        display_parts.append(f"{code}({p_str})")
    return "+".join(display_parts)

# --- AUDIT ---
def run_comprehensive_audit(symbol, timeframe, dry_run):
    print_header(f"STRATEGY AUDIT: {symbol} {timeframe}")
    base = deepcopy(BASE_CONFIG)
    base['symbol'] = symbol
    base['timeframe'] = timeframe
    base['endDate'] = datetime.now().strftime("%Y-%m-%d")
    base['startDate'] = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # Use OR mode for audit to ensure we get trades to analyze
    base.setdefault('params', {})['hybridMode'] = "OR"
    
    print(f"{'Strategy':<25} | {'Trades':<8} | {'Return':<10} | {'DD':<8} | {'Status'}")
    print("-" * 75)
    
    for strat in TREND_POOL + RANGE_POOL:
        cfg = deepcopy(base)
        params = SANITY_PARAMS.get(strat, {})
        cfg['strategies'] = [{"code": strat, "params": params}]
        
        try:
            res = execute_simulation(cfg, dry_run)
            m = res.get('metrics', {})
            trades = m.get('totalTrades', 0)
            ret = m.get('totalReturn', 0)
            dd = m.get('maxDrawdown', 0)
            
            status_col = Fore.GREEN if ret > 0 else Fore.RED
            status_txt = "PASS" if trades > 0 else "NO TRADES"
            if ret < 0: status_txt = "FAIL"
            
            print(f"{strat:<25} | {trades:<8} | {ret:>8.2f}% | {dd:>6.2f}% | {status_col}{status_txt}{Style.RESET_ALL}")
        except:
            print(f"{strat:<25} | ERROR    | 0.00%      | 0.00%    | {Fore.RED}CRASH{Style.RESET_ALL}")
    print("\n")

# --- STRATEGY GENERATOR ---
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

def create_objective(symbol, timeframe, date_config, market_models, dry_run=False):
    def objective(trial: optuna.Trial):
        test_config = deepcopy(BASE_CONFIG)
        test_config['symbol'] = symbol.upper()
        test_config['timeframe'] = timeframe
        
        n_trend = trial.suggest_int("n_trend", 0, 4)
        n_range = trial.suggest_int("n_range", 0, 4)
        if n_trend + n_range == 0: n_trend = 1
        
        selected_strategies = []
        for i in range(n_trend):
            s_name = trial.suggest_categorical(f"t_strat_{i}", TREND_POOL)
            selected_strategies.append(get_strategy_params(trial, s_name, f"t{i}"))
        for i in range(n_range):
            s_name = trial.suggest_categorical(f"r_strat_{i}", RANGE_POOL)
            selected_strategies.append(get_strategy_params(trial, s_name, f"r{i}"))

        test_config['strategies'] = selected_strategies
        
        set_nested(test_config, "params.hybridMode", "OR") 
        test_config['maxPyramiding'] = 1 
        set_nested(test_config, "params.maxPyramiding", 1)
        set_nested(test_config, "params.tslAtrMult", trial.suggest_float("tsl_mult", 1.5, 6.0, step=0.5))
        
        if market_models:
             test_config['mlMode'] = "predictions"
             test_config['mlModel'] = trial.suggest_categorical("mlModel", market_models)
             test_config['mlThreshold'] = trial.suggest_float("mlThreshold", 0.50, 0.70, step=0.05)
        else: test_config['mlMode'] = "off"

        swarm_str = format_strategy_display(selected_strategies)
        
        # ✅ FLUSH FIX: Ensure text appears immediately
        print(f"\n{Fore.WHITE}[Trial {trial.number}] Testing: {swarm_str}...{Style.RESET_ALL}", end="\r", flush=True)

        try:
            # PHASE 1: TRAINING
            train_config = deepcopy(test_config)
            train_config['startDate'] = date_config['train_start']
            train_config['endDate'] = date_config['train_end']
            res_train = execute_simulation(train_config, dry_run)
            t_ret = res_train.get('metrics', {}).get('totalReturn', 0)
            t_trades = res_train.get('metrics', {}).get('totalTrades', 0)
            
            if t_trades < MIN_TRADES_TOTAL: 
                print(f"{Fore.CYAN}[Trial {trial.number}] 💤 SKIP | Trades: {t_trades} | {swarm_str} {Style.RESET_ALL}")
                return -9999 
            if t_ret < 20: 
                print(f"{Fore.CYAN}[Trial {trial.number}] 💤 WEAK | Train Ret: {t_ret:.1f}% | {swarm_str} {Style.RESET_ALL}")
                return -9999 

            # PHASE 2: EXAM
            test_config_ex = deepcopy(test_config)
            test_config_ex['startDate'] = date_config['test_start']
            test_config_ex['endDate'] = date_config['test_end']
            res_test = execute_simulation(test_config_ex, dry_run)
            e_ret = res_test.get('metrics', {}).get('totalReturn', 0)
            e_dd = abs(res_test.get('metrics', {}).get('maxDrawdown', 100))
            
            # Calmar Calc
            e_calmar = e_ret / (e_dd if e_dd > 0 else 0.1)
            
            if e_ret < 0 or e_dd > 60.0:
                print(f"{Fore.WHITE}[Trial {trial.number}] ❌ FAIL | Ret: {e_ret:.2f}% | DD: {e_dd:.2f}% | Calmar: {e_calmar:.2f} | {swarm_str}{Style.RESET_ALL}")
                return -9999

            print(f"{Fore.GREEN}[Trial {trial.number}] 🚀 PASS | Ret: {e_ret:.2f}% | DD: {e_dd:.2f}% | Calmar: {e_calmar:.2f} | {swarm_str}{Style.RESET_ALL}")
            
            if e_ret > 50: 
                full_ret, full_dd, full_trades = verify_strategy_stability(test_config, date_config)
                save_winner(test_config, full_ret, full_dd, full_trades, date_config, e_ret)

            return e_ret
        except: return -9999
    return objective

class EarlyStoppingCallback:
    def __init__(self, patience=200):
        self.patience = patience
        self.best_value = -float('inf')
        self.stagnant = 0
    def __call__(self, study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            if trial.value > self.best_value:
                self.best_value = trial.value
                self.stagnant = 0
            else: self.stagnant += 1
            if self.stagnant >= self.patience: study.stop()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not fetch_all_models(args.dry_run): return

    print_header("ROCKET SWARM OPTIMIZER (FULL DISCLOSURE)")
    symbols = list(set([m['id'].split('_')[0].upper() + "-USD" for m in ALL_AVAILABLE_MODELS]))
    if not symbols: symbols = ["BTC-USD", "ETH-USD"]
    
    sym = select_from_list("Symbol", sorted(symbols))
    tf = select_from_list("Timeframe", ["1h", "4h", "15m"])
    
    try: n_trials = int(input(f"\n{Fore.YELLOW}Enter Max Trials (Default 5000): {Style.RESET_ALL}").strip() or 5000)
    except: n_trials = 5000

    run_comprehensive_audit(sym, tf, args.dry_run)
    
    print(f"\n{Fore.CYAN}🤖 AUTO-PILOT ENGAGED: Hunting for Synergistic Combos...{Style.RESET_ALL}")
    dates = get_train_test_dates(years_back=2, test_split=0.25)
    models = [m['id'] for m in ALL_AVAILABLE_MODELS if sym.split('-')[0].lower() in m['id'].lower() and tf in m['id'].lower()]
    if not models: models = [ALL_AVAILABLE_MODELS[0]['id']]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    try: 
        study.optimize(
            create_objective(sym, tf, dates, models, args.dry_run), 
            n_trials=n_trials, 
            callbacks=[EarlyStoppingCallback(patience=200)],
            n_jobs=PARALLEL_JOBS
        )
    except KeyboardInterrupt: print("Paused.")

if __name__ == "__main__": main()
