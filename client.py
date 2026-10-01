# File: run_optimizer.py
# 🚀 UPGRADE: v4.0 - "Ghost Buster" Edition (Stable & Strict)
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
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- CONFIGURATION ---
ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://127.0.0.1:8000")
RESULTS_DIR = os.path.join(os.getcwd(), "data", "optimizer_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 12345
random.seed(SEED)
np.random.seed(SEED)

# --- SETTINGS ---
PARALLEL_JOBS = 6           # Keep 1 to prevent database locks
TIMEOUT_SECONDS = 600 

# 🚀 GHOST BUSTER FILTERS
MIN_TRADES_TOTAL = 20       # Reject strategies with fewer trades (Luck filter)
PRUNE_DRAWDOWN_LIMIT = 60.0 # Allow looser drawdown during discovery
MAX_CALMAR_CAP = 30.0       # Cap unrealistic ratios

# --- STRATEGY POOLS ---
TREND_STRATEGIES = ["sma_crossover", "macd_crossover", "ichimoku_cloud", "psar_signal", "obv_signal", "atr_breakout"]
RANGE_STRATEGIES = ["rsi_divergence", "stochastic_crossover", "bollinger_bands", "cci_oversold"]
ALL_AVAILABLE_MODELS = []

# --- BASE CONFIG ---
BASE_CONFIG = {
    "symbol": "BTC-USD",
    "timeframe": "1h",
    "initialBalance": 1000,
    "fee": 0.006,
    "riskManagementMode": "standard",
    "riskPercentage": 1, 
    "optimizer_mode": True
}

# --- SANITY PARAMS (For Audit Only) ---
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
# 1. UI UTILITIES
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
# 2. NETWORK
# ==============================================================================
def robust_request(method, url, json_data=None, retries=3):
    for i in range(retries):
        try:
            if method == 'GET': response = requests.get(url, timeout=15)
            else: response = requests.post(url, json=json_data, timeout=600)
            
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

def generate_wfo_folds():
    now = datetime.now(timezone.utc)
    start_year = now.year - 3
    folds = []
    for year in range(start_year, now.year):
        folds.append({"startDate": f"{year}-01-01", "endDate": f"{year}-12-31", "id": year})
    folds.append({"startDate": f"{now.year}-01-01", "endDate": now.strftime("%Y-%m-%d"), "id": now.year})
    return folds

def calculate_stitched_metrics(equity_curves, trade_counts):
    if not equity_curves: return {"StitchedCalmarRatio": -9999, "StitchedMaxDrawdown": 100}
    
    total_trades = sum(trade_counts)
    
    # 🚀 GHOST BUSTER: Reject low trades
    if total_trades < MIN_TRADES_TOTAL:
        return {"StitchedCalmarRatio": -9999, "StitchedMaxDrawdown": 100}

    all_dfs = []
    for i, fold_curve in enumerate(equity_curves):
        if not fold_curve: continue
        df = pd.DataFrame(fold_curve)
        if df.empty or 'balance' not in df.columns: continue
        
        if i == 0:
            df['balance_continuous'] = df['balance']
        else:
            try:
                initial_fold = fold_curve[0]['balance']
                prev_end_bal = all_dfs[-1]['balance_continuous'].iloc[-1]
                df['balance_continuous'] = df['balance'] - initial_fold + prev_end_bal
            except: continue
        all_dfs.append(df)
    
    if not all_dfs: return {"StitchedCalmarRatio": -9999, "StitchedMaxDrawdown": 100}

    stitched_df = pd.concat(all_dfs, ignore_index=True)
    initial = stitched_df['balance_continuous'].iloc[0]
    final = stitched_df['balance_continuous'].iloc[-1]
    
    if initial <= 0: return {"StitchedCalmarRatio": -9999, "StitchedMaxDrawdown": 100}
    
    peak = stitched_df['balance_continuous'].cummax()
    drawdown = (stitched_df['balance_continuous'] - peak) / (peak + 1e-9) * 100 
    max_dd_pct = abs(drawdown.min())
    if max_dd_pct == 0: max_dd_pct = 0.1
    
    try:
        t_start = pd.to_datetime(stitched_df['timestamp'].iloc[0])
        t_end = pd.to_datetime(stitched_df['timestamp'].iloc[-1])
        years = (t_end - t_start).total_seconds() / (365.25 * 24 * 3600)
        if years < 0.1: years = 0.1
        
        if final <= 0: annual_ret = -1.0
        else: annual_ret = ((final / initial) ** (1 / years)) - 1
        
    except: return {"StitchedCalmarRatio": -9999, "StitchedMaxDrawdown": 100}
    
    calmar = (annual_ret * 100) / max_dd_pct
    if calmar > MAX_CALMAR_CAP: calmar = 0.1 # Reject fake perfection
    
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
        
        # --- 1. Dynamic Parameters ---
        strategies_payload = []
        
        trend_p = {}
        if trend_strat == "sma_crossover":
            s1 = trial.suggest_int("sma_1", 5, 60, step=5)
            s2 = trial.suggest_int("sma_2", 20, 200, step=10)
            trend_p = {"sma_fast_period": min(s1, s2), "sma_slow_period": max(s1, s2)}
        elif trend_strat == "macd_crossover":
            trend_p = {
                "macd_fast_period": trial.suggest_int("macd_f", 8, 20),
                "macd_slow_period": trial.suggest_int("macd_s", 21, 50),
                "macd_signal_period": trial.suggest_int("macd_sig", 5, 15)
            }
        elif trend_strat == "atr_breakout":
             trend_p = {
                 "atr_period": trial.suggest_int("atr_p", 10, 30),
                 "atr_multiplier": trial.suggest_float("atr_m", 1.5, 5.0, step=0.1)
             }
        strategies_payload.append({"code": trend_strat, "params": trend_p})

        range_p = {}
        if range_strat == "rsi_divergence":
            range_p = {
                "rsi_length": trial.suggest_int("rsi_len", 10, 30),
                "oversold_level": trial.suggest_int("rsi_os", 20, 40),
                "overbought_level": trial.suggest_int("rsi_ob", 60, 80)
            }
        elif range_strat == "bollinger_bands":
            range_p = {
                "bb_length": trial.suggest_int("bb_len", 15, 30),
                "bb_std": trial.suggest_float("bb_std", 1.5, 3.0, step=0.1)
            }
        elif range_strat == "cci_oversold":
            range_p = {
                "cci_length": trial.suggest_int("cci_len", 14, 40),
                "cci_oversold": trial.suggest_int("cci_os", -150, -90),
                "cci_overbought": trial.suggest_int("cci_ob", 90, 150)
            }
        strategies_payload.append({"code": range_strat, "params": range_p})

        test_config['strategies'] = strategies_payload
        
        # --- 2. Global Params (Regime + Risk) ---
        set_nested(test_config, "params.hybridMode", "REGIME")
        set_nested(test_config, "params.regime_threshold", trial.suggest_int("regime_threshold", 15, 35, step=5))
        
        # Pyramiding
        test_config['maxPyramiding'] = trial.suggest_int("pyramiding", 1, 3)
        set_nested(test_config, "params.maxPyramiding", test_config['maxPyramiding'])

        # Stop Loss Tuning
        set_nested(test_config, "params.tslAtrMult", trial.suggest_float("tsl_mult", 1.0, 5.0, step=0.5))
        
        # ML
        if market_models:
             test_config['mlMode'] = "predictions"
             test_config['mlModel'] = trial.suggest_categorical("mlModel", market_models)
             test_config['mlThreshold'] = trial.suggest_float("mlThreshold", 0.55, 0.75, step=0.05)
        else:
             test_config['mlMode'] = "off"

        # --- 3. Execution ---
        all_curves = [None] * len(wfo_folds)
        trade_counts = [0] * len(wfo_folds)
        
        for idx, fold in enumerate(wfo_folds):
            try:
                fold_config = deepcopy(test_config)
                fold_config['startDate'] = fold['startDate']
                fold_config['endDate'] = fold['endDate']
                
                res = execute_simulation(fold_config, dry_run)
                metrics = res.get('metrics', {})
                
                fold_dd = abs(metrics.get('maxDrawdown', 0))
                trades = metrics.get('totalTrades', 0)
                
                trade_counts[idx] = trades
                timestamp_str = datetime.now().strftime("%I:%M:%S %p")

                if fold_dd > PRUNE_DRAWDOWN_LIMIT:
                    print(f"{Fore.CYAN}[Trial {trial.number}] ✂️  PRUNED (High DD: {fold_dd:.2f}%) | {timestamp_str}{Style.RESET_ALL}")
                    return -9999, 100 
                
                all_curves[idx] = res.get('equityCurve', [])
            except Exception: 
                return -9999, 100 

        metrics = calculate_stitched_metrics(all_curves, trade_counts)
        calmar = metrics.get('StitchedCalmarRatio', -9999)
        dd = metrics.get('StitchedMaxDrawdown', 100)
        
        strat_label = f"{trend_strat}/{range_strat}"
        
        if calmar > 2.0:
            print(f"{Fore.GREEN}[Trial {trial.number}] 🟢 EXCELLENT | {strat_label:<35} | Calmar: {calmar:.2f} | DD: {dd:.2f}% | Trades: {sum(trade_counts)}{Style.RESET_ALL}")
        elif calmar > 0:
            print(f"{Fore.YELLOW}[Trial {trial.number}] 🟡 PROFIT    | {strat_label:<35} | Calmar: {calmar:.2f} | DD: {dd:.2f}% | Trades: {sum(trade_counts)}{Style.RESET_ALL}")
        
        # Save Data for export
        trial.set_user_attr("config", test_config)
        trial.set_user_attr("return", calmar) # Placeholder for sorting
        trial.set_user_attr("trades", sum(trade_counts))
        trial.set_user_attr("dd", dd)
        
        return calmar, dd

    return objective

def run_comprehensive_audit(symbol, timeframe, dry_run):
    print_header(f"STRATEGY AUDIT: {symbol} {timeframe}")
    base = deepcopy(BASE_CONFIG)
    base['symbol'] = symbol
    base['timeframe'] = timeframe
    base['endDate'] = datetime.now().strftime("%Y-%m-%d")
    base['startDate'] = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    print(f"{'Strategy':<25} | {'Trades':<8} | {'Return':<10} | {'Status'}")
    print("-" * 60)
    
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
        except:
            print(f"{strat:<25} | ERROR    | 0.00%      | {Fore.RED}CRASH{Style.RESET_ALL}")

# ==============================================================================
# 5. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not fetch_all_models(args.dry_run): return

    # Menu
    print_header("SETUP OPTIMIZER")
    symbols = list(set([m['id'].split('_')[0].upper() + "-USD" for m in ALL_AVAILABLE_MODELS]))
    if not symbols: symbols = ["BTC-USD", "ETH-USD"]
    
    selected_symbol = select_from_list("Symbol", sorted(symbols))
    selected_timeframe = select_from_list("Timeframe", ["1h", "4h", "15m"])
    
    try: n_trials = int(input(f"\n{Fore.YELLOW}Enter Number of Trials (Default 50): {Style.RESET_ALL}").strip() or 50)
    except: n_trials = 50

    # Run Audit
    run_comprehensive_audit(selected_symbol, selected_timeframe, args.dry_run)

    # Run Optimizer
    print_header(f"STARTING OPTIMIZATION ({n_trials} Trials)")
    
    # Get models for symbol/timeframe
    models = [m['id'] for m in ALL_AVAILABLE_MODELS if selected_symbol.split('-')[0].lower() in m['id'].lower() and selected_timeframe in m['id'].lower()]
    if not models: models = [ALL_AVAILABLE_MODELS[0]['id']] # Fallback

    study = optuna.create_study(directions=["maximize", "minimize"], sampler=optuna.samplers.NSGAIISampler(seed=SEED))
    
    try:
        study.optimize(
            create_objective(selected_symbol, selected_timeframe, generate_wfo_folds(), models, args.dry_run), 
            n_trials=n_trials, 
            n_jobs=PARALLEL_JOBS
        )
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Optimization paused.{Style.RESET_ALL}")

    # Results
    print_header("🏆 TOP 3 GOLDEN STRATEGIES")
    best_trials = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value, reverse=True)
    
    for i, t in enumerate(best_trials[:3]):
        cal = t.value
        cfg = t.user_attrs.get('config')
        trades = t.user_attrs.get('trades')
        dd = t.user_attrs.get('dd')
        combo = t.user_attrs.get('combo')

        if not cfg: continue
        
        print_box(f"RANK #{i+1}: {combo}", [
            f"Calmar Ratio : {cal:.2f}",
            f"Max Drawdown : {dd:.2f}%",
            f"Total Trades : {trades}"
        ], color=Fore.MAGENTA if i==0 else Fore.CYAN)
        
        # Save
        fname = f"winner_{selected_symbol}_{selected_timeframe}_RANK{i+1}_{int(time.time())}.json"
        
        final_data = {
            "strategies": cfg['strategies'],
            "params": cfg['params'],
            "mlMode": "predictions",
            "mlModel": cfg.get('mlModel'),
            "mlThreshold": cfg.get('mlThreshold'),
            "metrics": {"calmar": cal, "totalTrades": trades, "maxDrawdown": dd},
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(RESULTS_DIR, fname), 'w') as f:
            json.dump(final_data, f, indent=2)
        print(f"{Fore.GREEN}   💾 Saved to: {fname}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
