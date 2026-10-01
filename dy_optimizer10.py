# File: dy_optimizer_dynamic_pareto.py
# 🚀 UPGRADE: v20.0 - "Dynamic Optimizer Supreme"
# 1. ADDED: Full CLI Support & Dynamic Trade Scaling.
# 2. OPTIMIZED: Numpy-based Monte Carlo (Matrix Vectorization).
# 3. ADDED: Result Caching & Exponential Backoff.
# 4. IMPROVED: Visualization & File Logging.
# 5. ADDED: Dynamic Strategy Pools via JSON.

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
import hashlib
from copy import deepcopy
from colorama import Fore, Style, init
import warnings
import uuid 
from typing import Dict, List, Tuple, Any, Optional

# Try importing for visualization
HAS_VIZ = False
try:
    from optuna.visualization import plot_pareto_front
    import plotly
    HAS_VIZ = True
except ImportError:
    pass

init(autoreset=True)
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- CONFIG ---
ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://127.0.0.1:8000")
RESULTS_DIR = os.path.join(os.getcwd(), "data", "optimizer_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 🚀 LOGGING CONFIG
logging.basicConfig(
    filename='optimizer_errors.log',
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger("Optimizer")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('optimizer_history.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(file_handler)

SEED = 12345
random.seed(SEED)
np.random.seed(SEED)

DEFAULT_PARALLEL_JOBS = max(1, (os.cpu_count() or 4) - 1)
TIMEOUT_SECONDS = 600 

# CONFIGURABLE TRADE FREQUENCY MULTIPLIERS
TRADE_FREQ_MULTIPLIERS = {
    '1h': 0.04,  
    '4h': 0.02,  
    '15m': 0.08  
}

# DEFAULT POOLS
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

# 🚀 CACHE STORAGE
SIMULATION_CACHE = {}

# --- UTILS ---
def print_header(text: str): print(f"\n{Fore.CYAN}{'='*60}\n {text.center(58)} \n{'='*60}{Style.RESET_ALL}")

def select_from_list(title: str, options: List[str], default_val: Optional[str] = None) -> str:
    if default_val and default_val in options: return default_val
    print(f"\n{Fore.YELLOW}>>> Select {title}:{Style.RESET_ALL}")
    for i, opt in enumerate(options): print(f"  {Fore.CYAN}{i+1}.{Style.RESET_ALL} {opt}")
    while True:
        try:
            idx = int(input(f"\n{Fore.GREEN}Enter number (default 1): {Style.RESET_ALL}").strip() or 1) - 1
            if 0 <= idx < len(options): return options[idx]
        except: pass

def robust_request(method: str, url: str, json_data: Optional[Dict] = None, retries: int = 3) -> requests.Response:
    for i in range(retries):
        try:
            if method == 'GET': r = requests.get(url, timeout=15)
            else: r = requests.post(url, json=json_data, timeout=TIMEOUT_SECONDS)
            if r.status_code == 200: return r
            elif r.status_code >= 500: 
                time.sleep(2 ** i) # Exponential Backoff
            else: return r
        except requests.exceptions.ConnectionError:
            time.sleep(2 ** i)
        except Exception:
            time.sleep(2 ** i)
    raise Exception(f"Failed: {url}")

def fetch_all_models(dry_run: bool = False) -> bool:
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
    except: 
        print(f"{Fore.RED}⚠️ ML Server Unreachable ({ML_SERVER_URL}). Proceeding in Pure TA Mode.{Style.RESET_ALL}")
        ALL_AVAILABLE_MODELS = [] 
        return True

def get_config_hash(config: Dict) -> str:
    """Generate a deterministic hash for a configuration object."""
    return hashlib.md5(json.dumps(config, sort_keys=True).encode('utf-8')).hexdigest()

def execute_simulation(config: Dict, dry_run: bool = False) -> Dict:
    if dry_run: return {"metrics": {"totalReturn": 10.0, "totalTrades": 50, "maxDrawdown": 5.0}, "equityCurve": [], "tradeBreakdown": []}
    
    # Check Cache
    config_hash = get_config_hash(config)
    if config_hash in SIMULATION_CACHE:
        return SIMULATION_CACHE[config_hash]

    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
    r = robust_request('POST', url, json_data=config)
    if r.status_code != 200: raise Exception(f"Server Error: {r.text}")
    j = r.json()
    result = j.get("combinedResult", j)
    
    # Store in Cache
    SIMULATION_CACHE[config_hash] = result
    return result

def set_nested(d: Dict, keys: str, value: Any):
    keys_list = keys.split('.')
    for key in keys_list[:-1]: d = d.setdefault(key, {})
    d[keys_list[-1]] = value

def get_train_test_dates(years_back: int = 2) -> Dict[str, str]:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=365*years_back)
    total_days = (now - start).days
    
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

# 🚀 OPTIMIZED: Batched Matrix Monte Carlo (Memory Safe)
def monte_carlo_certification(trades: List[Dict], trial_seed: int, initial_balance: float = 1000, max_iters: int = 5000, limit_dd: float = 30.0) -> Tuple[float, int]:
    # 🚀 HARDENED: Empty list check
    if not trades or len(trades) < 5: return 50.0, 0
    
    profits = np.array([t['profit'] for t in trades])
    n_trades = len(profits)
    total_iterations = min(max_iters, max(1000, n_trades * 50))
    
    rng = np.random.default_rng(trial_seed)
    ruin_count = 0
    BATCH_SIZE = 1000 
    
    for i in range(0, total_iterations, BATCH_SIZE):
        current_batch = min(BATCH_SIZE, total_iterations - i)
        noise = rng.random((current_batch, n_trades))
        shuffled_indices = np.argsort(noise, axis=1)
        shuffled_profits = profits[shuffled_indices]
        
        equity_curves = np.cumsum(shuffled_profits, axis=1) + initial_balance
        start_col = np.full((current_batch, 1), initial_balance)
        equity_full = np.hstack((start_col, equity_curves))
        
        peaks = np.maximum.accumulate(equity_full, axis=1)
        drawdowns = (peaks - equity_full) / peaks * 100
        max_dds = np.max(drawdowns, axis=1)
        
        ruin_count += np.sum(max_dds > limit_dd)
            
    return (ruin_count / total_iterations) * 100, total_iterations

def walk_forward_validation(config: Dict, start_date: str, end_date: str, windows: int = 4, quiet: bool = False) -> bool:
    if not quiet: print(f"   {Fore.YELLOW}Running Out-of-Sample WFO ({start_date} to {end_date})...{Style.RESET_ALL}")
    s_dt = datetime.strptime(start_date, "%Y-%m-%d")
    e_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (e_dt - s_dt).days
    
    # 🚀 DYNAMIC WINDOW SCALING
    MIN_DAYS_PER_WINDOW = 20
    if total_days // windows < MIN_DAYS_PER_WINDOW:
        adjusted_windows = max(2, total_days // MIN_DAYS_PER_WINDOW)
        if not quiet: print(f"   {Fore.MAGENTA}⚠️ Dataset short. Adjusted windows from {windows} to {adjusted_windows}.{Style.RESET_ALL}")
        windows = adjusted_windows

    if windows < 1 or (total_days // windows) < 10: 
        if not quiet: print(f"   {Fore.RED}WFO Failed: Not enough data for validation.{Style.RESET_ALL}")
        return False
        
    passed_windows = 0
    for i in range(windows):
        w_start = s_dt + timedelta(days=i * (total_days // windows))
        w_end = w_start + timedelta(days=(total_days // windows))
        w_config = deepcopy(config)
        w_config['startDate'] = w_start.strftime("%Y-%m-%d")
        w_config['endDate'] = w_end.strftime("%Y-%m-%d")
        try:
            res = execute_simulation(w_config)
            ret = res.get('metrics', {}).get('totalReturn', 0)
            if ret > 0: passed_windows += 1
        except Exception as e:
            logging.error(f"[WFO Critical Error] Window {i+1} Failed: {str(e)}")
            return False 
    
    success = passed_windows >= (windows - 1)
    if not quiet: 
        color = Fore.GREEN if success else Fore.RED
        print(f"   {color}WFO Result: {passed_windows}/{windows} passed.{Style.RESET_ALL}")
    if success: logger.info(f"WFO Passed for config: {config['symbol']} {config['timeframe']}")
    return success

def verify_holdout_performance(config: Dict, dates: Dict, quiet: bool = False) -> Tuple[float, float, int, List[Dict], float]:
    if not quiet: print(f"{Fore.CYAN}   >>> 🔒 Testing Hidden Holdout ({dates['holdout_start']} - {dates['holdout_end']})...{Style.RESET_ALL}")
    holdout_config = deepcopy(config)
    holdout_config['startDate'] = dates['holdout_start']
    holdout_config['endDate'] = dates['holdout_end']
    try:
        res = execute_simulation(holdout_config)
        m = res.get('metrics', {})
        ret = m.get('totalReturn', 0)
        dd = m.get('maxDrawdown', 1.0)
        trades = m.get('totalTrades', 0)
        
        safe_dd = max(1.0, abs(dd))
        calmar = ret / safe_dd
        
        return ret, dd, trades, res.get('tradeBreakdown', []), calmar
    except Exception as e:
        logger.error(f"Holdout Error: {str(e)}")
        return -999, 100, 0, [], -999

def save_winner(config: Dict, val_metrics: Dict, holdout_metrics: Dict, mc_iterations: int, val_trades: List[Dict] = [], notes: str = "", output_dir: str = ""):
    ret, dd, trades, rank_score = holdout_metrics['ret'], holdout_metrics['dd'], holdout_metrics['trades'], holdout_metrics['score']
    uid = str(uuid.uuid4())[:8] 
    fname = f"FORTRESS_DYNAMIC_{config['symbol']}_{config['timeframe']}_CALMAR{int(rank_score)}_{uid}.json"
    
    val_calmar = val_metrics['valReturn'] / max(1.0, val_metrics['valDrawdown'])
    meta_data = {
        "nStrategies": len(config["strategies"]),
        "nTrend": sum(1 for s in config["strategies"] if s["code"] in TREND_POOL),
        "nRange": sum(1 for s in config["strategies"] if s["code"] in RANGE_POOL),
        "mcIterations": mc_iterations
    }

    final_data = {
        "symbol": config['symbol'],
        "timeframe": config['timeframe'],
        "strategies": config['strategies'],
        "params": config['params'],
        "mlMode": config.get('mlMode'),
        "mlModel": config.get('mlModel'),
        "mlThreshold": config.get('mlThreshold'),
        "riskManagementMode": "dynamic", 
        "riskPercentage": config['riskPercentage'],
        "growthCapitalTarget": config.get('growthCapitalTarget', 0),
        "metrics": { "totalReturn": ret, "maxDrawdown": dd, "totalTrades": trades, "validationCalmar": val_calmar, "holdoutCalmar": rank_score },
        "meta": meta_data,
        "validationMetrics": val_metrics,
        "validationTrades": val_trades, 
        "validation": "3_WAY_SPLIT_CERTIFIED_V20.0",
        "notes": notes,
        "timestamp": datetime.now().isoformat()
    }
    
    target_dir = output_dir if output_dir else RESULTS_DIR
    os.makedirs(target_dir, exist_ok=True)
    
    temp_path = os.path.join(target_dir, f".tmp_{fname}")
    final_path = os.path.join(target_dir, fname)
    
    with open(temp_path, 'w') as f: json.dump(final_data, f, indent=2)
    os.rename(temp_path, final_path)
    print(f"{Fore.GREEN}   💾 FORTRESS SAVE: {fname}{Style.RESET_ALL}")
    logger.info(f"Winner Saved: {fname} | Calmar: {rank_score:.2f}")

def format_strategy_display(config: Dict) -> str:
    strategies = config['strategies']
    display_parts = []
    for s in strategies:
        code = s['code'].split('_')[0].upper()
        p_str = ",".join([str(v) for k,v in s['params'].items()])
        display_parts.append(f"{code}({p_str})")
    
    base_str = "+".join(display_parts)
    if config.get('mlMode') == 'predictions':
        base_str += f" + ML({config.get('mlModel', 'Unk')} > {config.get('mlThreshold', 0):.2f})"
        
    return base_str

def get_strategy_params(trial: optuna.Trial, strat_code: str, idx: str) -> Dict:
    p = {}
    if strat_code == "sma_crossover":
        s1 = trial.suggest_int(f"s{idx}_sma1", 3, 100, 1); s2 = trial.suggest_int(f"s{idx}_sma2", 10, 300, 5)
        p = {"sma_fast_period": min(s1,s2), "sma_slow_period": max(s1,s2)}
    elif strat_code == "macd_crossover":
        p = {"macd_fast_period": trial.suggest_int(f"s{idx}_mf", 3, 60), "macd_slow_period": trial.suggest_int(f"s{idx}_ms", 10, 150), "macd_signal_period": trial.suggest_int(f"s{idx}_sig", 2, 25)}
    elif strat_code == "atr_breakout":
        p = {"atr_period": trial.suggest_int(f"s{idx}_atrp", 3, 60), "atr_multiplier": trial.suggest_float(f"s{idx}_atrm", 0.5, 6.0, step=0.1)}
    elif strat_code == "psar_signal":
        p = {"psar_step": trial.suggest_float(f"s{idx}_psar", 0.001, 0.1, step=0.001)}
    elif strat_code == "obv_signal": p = {"obv_ma_period": trial.suggest_int(f"s{idx}_obv", 5, 200)}
    elif strat_code == "rsi_divergence":
        p = {"rsi_length": trial.suggest_int(f"s{idx}_rsi_len", 2, 55), "oversold_level": trial.suggest_int(f"s{idx}_rsi_os", 5, 45), "overbought_level": trial.suggest_int(f"s{idx}_rsi_ob", 55, 95)}
    elif strat_code == "bollinger_bands":
        p = {"bb_length": trial.suggest_int(f"s{idx}_bb_len", 5, 100), "bb_std": trial.suggest_float(f"s{idx}_bb_std", 0.5, 4.0, step=0.1)}
    elif strat_code == "stochastic_crossover":
        p = {"k_period": trial.suggest_int(f"s{idx}_k", 3, 60), "d_period": 3}
    elif strat_code == "cci_oversold":
        p = {"cci_length": trial.suggest_int(f"s{idx}_cci_len", 5, 100), "cci_oversold": -100, "cci_overbought": 100}
    elif strat_code == "ichimoku_cloud":
        p = {"tenkan_period": trial.suggest_int(f"s{idx}_tenkan", 7, 15), "kijun_period": trial.suggest_int(f"s{idx}_kijun", 20, 35), "senkou_span_b_period": trial.suggest_int(f"s{idx}_senkou_b", 40, 60)}
    return {"code": strat_code, "params": p}

def create_objective(symbol: str, timeframe: str, date_config: Dict, market_models: List[str], args: argparse.Namespace):
    # ADAPTIVE TRADE COUNT
    train_start_dt = datetime.strptime(date_config['train_start'], "%Y-%m-%d")
    train_end_dt = datetime.strptime(date_config['train_end'], "%Y-%m-%d")
    train_days = (train_end_dt - train_start_dt).days
    
    val_start_dt = datetime.strptime(date_config['val_start'], "%Y-%m-%d")
    val_end_dt = datetime.strptime(date_config['val_end'], "%Y-%m-%d")
    val_days = (val_end_dt - val_start_dt).days

    mult = args.trade_freq_mult if args.trade_freq_mult else TRADE_FREQ_MULTIPLIERS.get(timeframe, 0.03)
    min_trades = args.min_trades if args.min_trades else max(15, int(train_days * mult))
        
    if not args.quiet:
        print(f"{Fore.BLUE}ℹ️ Adaptive Min Trades: Train={min_trades} | Val={max(5, int(min_trades * (val_days / train_days)))}{Style.RESET_ALL}")

    def objective(trial: optuna.Trial):
        test_config = deepcopy(BASE_CONFIG)
        test_config['symbol'] = symbol.upper()
        test_config['timeframe'] = timeframe
        
        n_trend = trial.suggest_int("n_trend", 0, 3)
        n_range = trial.suggest_int("n_range", 0, 3)
        if n_trend + n_range > 4: n_range = max(1, 4 - n_trend)
        while n_trend + n_range < 2:
            if n_trend <= n_range: n_trend += 1
            else: n_range += 1
        
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
        set_nested(test_config, "params.tslAtrMult", trial.suggest_float("tsl_mult", 1.0, 10.0, step=0.5))
        set_nested(test_config, "params.minAdxLevel", trial.suggest_int("min_adx", 0, 50, step=5))
        set_nested(test_config, "params.minAtrPct", trial.suggest_float("min_atr", 0.0, 0.5, step=0.05))
        set_nested(test_config, "params.trendFilterPeriod", trial.suggest_categorical("trend_filt", [20, 50, 100, 200]))
        test_config['riskPercentage'] = trial.suggest_int("risk_pct", 1, 5)
        
        if market_models:
             test_config['mlMode'] = "predictions"
             test_config['mlModel'] = trial.suggest_categorical("mlModel", market_models)
             test_config['mlThreshold'] = trial.suggest_float("mlThreshold", 0.40, 0.80, step=0.05)
        else: test_config['mlMode'] = "off"

        swarm_str = format_strategy_display(test_config)
        if not args.quiet:
            print(f"\n{Fore.WHITE}[Trial {trial.number}] Synergy: {swarm_str}...{Style.RESET_ALL}", end="\r", flush=True)

        try:
            # PHASE 1: TRAINING
            train_config = deepcopy(test_config)
            train_config['startDate'] = date_config['train_start']
            train_config['endDate'] = date_config['train_end']
            res_train = execute_simulation(train_config, args.dry_run)
            t_ret = res_train.get('metrics', {}).get('totalReturn', 0)
            t_trades = res_train.get('metrics', {}).get('totalTrades', 0)
            
            if t_ret <= 0 or t_trades < min_trades: 
                if not args.quiet: print(f"{Fore.CYAN}[Trial {trial.number}] 💤 TRAIN FAIL | {swarm_str} | Ret: {t_ret:.1f}% | Trades: {t_trades}/{min_trades}{Style.RESET_ALL}")
                return (t_ret - 20.0), 100.0 

            # PHASE 2: VALIDATION
            val_config = deepcopy(test_config)
            val_config['startDate'] = date_config['val_start']
            val_config['endDate'] = date_config['val_end']
            res_val = execute_simulation(val_config, args.dry_run)
            v_ret = res_val.get('metrics', {}).get('totalReturn', 0)
            v_dd = abs(res_val.get('metrics', {}).get('maxDrawdown', 1.0)) 
            v_trades = res_val.get('metrics', {}).get('totalTrades', 0)

            # Pruning
            if v_dd > args.drawdown_limit:
                if not args.quiet: print(f"{Fore.YELLOW}[Trial {trial.number}] ✂️ PRUNED | DD {v_dd:.1f}% > Limit {args.drawdown_limit}%{Style.RESET_ALL}")
                raise optuna.TrialPruned(f"Drawdown {v_dd}% exceeds limit {args.drawdown_limit}%")

            val_min_trades = max(5, int(min_trades * (val_days / train_days)))
            
            if v_ret < 0 or v_trades < val_min_trades:
                if not args.quiet: print(f"{Fore.WHITE}[Trial {trial.number}] ❌ VAL FAIL | {swarm_str} | Ret: {v_ret:.2f}% | Trades: {v_trades}/{val_min_trades}{Style.RESET_ALL}")
                scaled_penalty_dd = min(100.0, v_dd * 1.5) 
                return (v_ret - 20.0), scaled_penalty_dd

            if v_dd < 1.0: v_dd = 1.0
            calmar_score = v_ret / v_dd
            
            if not args.quiet: print(f"{Fore.GREEN}[Trial {trial.number}] 🚀 VAL PASS | Calmar: {calmar_score:.2f} | Ret: {v_ret:.1f}% | DD: {v_dd:.1f}% | {swarm_str}{Style.RESET_ALL}")
            
            # PHASE 3: FORTRESS GATE
            if calmar_score >= args.calmar_min:
                h_ret, h_dd, h_trades, h_trade_list, h_calmar = verify_holdout_performance(test_config, date_config, args.quiet)
                
                if h_ret > 0:
                    if not args.quiet: print(f"   {Fore.CYAN}Running Multi-Seed Monte Carlo Gauntlet ({args.n_seeds} Seeds)...{Style.RESET_ALL}")
                    passed_mc = True
                    avg_ruin = 0
                    
                    for i in range(args.n_seeds):
                        ruin_prob, mc_iters = monte_carlo_certification(
                            h_trade_list, 
                            trial_seed=SEED + trial.number + (i * 999), 
                            initial_balance=test_config['initialBalance'],
                            max_iters=args.mc_max_iters,
                            limit_dd=args.mc_ruin_limit
                        )
                        avg_ruin += ruin_prob
                        if ruin_prob >= 10.0:
                            if not args.quiet: print(f"   {Fore.RED}❌ MC FAIL Seed {i+1}: Risk of Ruin {ruin_prob:.1f}%{Style.RESET_ALL}")
                            passed_mc = False
                            break
                    
                    if passed_mc:
                        if walk_forward_validation(test_config, date_config['holdout_start'], date_config['holdout_end'], windows=4, quiet=args.quiet):
                            val_metrics = {"valReturn": v_ret, "valDrawdown": v_dd}
                            holdout_metrics = {"ret": h_ret, "dd": h_dd, "trades": h_trades, "score": h_calmar}
                            val_trades_list = res_val.get('tradeBreakdown', [])
                            
                            save_winner(test_config, val_metrics, holdout_metrics, mc_iters, val_trades=val_trades_list, notes="Pareto Fortress (Dynamic)", output_dir=args.output_dir)
                    else:
                        pass
                else:
                    if not args.quiet: print(f"   {Fore.RED}❌ HOLDOUT FAIL: Return {h_ret:.2f}%{Style.RESET_ALL}")
            else:
                 if not args.quiet: print(f"   {Fore.YELLOW}⚠️ CALMAR FAIL: {calmar_score:.2f} (Needs {args.calmar_min}){Style.RESET_ALL}")
            
            return v_ret, v_dd

        except optuna.TrialPruned:
            raise 
        except Exception as e: 
            logging.error(f"[CRITICAL ERROR] Trial {trial.number}:\n{traceback.format_exc()}")
            if not args.quiet: print(f"{Fore.RED}[CRITICAL ERROR] Trial {trial.number} Failed. See optimizer_history.log.{Style.RESET_ALL}")
            return -50.0, 100.0 

    return objective

def main():
    parser = argparse.ArgumentParser(description="DyOptimizer Dynamic Pareto")
    parser.add_argument("--dry-run", action="store_true", help="Run without connecting to ML server")
    
    # 🚀 CLI ARGS
    parser.add_argument("--symbol", type=str, help="Symbol (e.g. BTC-USD)", default=None)
    parser.add_argument("--timeframe", type=str, help="Timeframe (e.g. 1h, 4h)", default=None)
    parser.add_argument("--trials", type=int, help="Max trials", default=None)
    parser.add_argument("--n-seeds", type=int, default=3, help="Number of Monte Carlo seeds")
    parser.add_argument("--mc-max-iters", type=int, default=5000, help="Max Monte Carlo iterations")
    parser.add_argument("--mc-ruin-limit", type=float, default=30.0, help="Max Drawdown % for Ruin calc")
    parser.add_argument("--drawdown-limit", type=float, default=40.0, help="Pruning drawdown limit %")
    parser.add_argument("--calmar-min", type=float, default=2.0, help="Minimum Calmar Ratio to Pass")
    parser.add_argument("--min-trades", type=int, help="Override minimum trades per trial")
    parser.add_argument("--trade-freq-mult", type=float, help="Override trade freq multiplier")
    parser.add_argument("--jobs", type=int, default=DEFAULT_PARALLEL_JOBS, help="Number of parallel jobs")
    parser.add_argument("--pool-file", type=str, help="Path to JSON file with custom strategy pools")
    parser.add_argument("--output-dir", type=str, help="Directory to save results", default=RESULTS_DIR)
    parser.add_argument("--quiet", action="store_true", help="Suppress console output for background runs")

    args = parser.parse_args()
    
    if not fetch_all_models(args.dry_run): return

    # DYNAMIC POOL LOADING
    global TREND_POOL, RANGE_POOL
    if args.pool_file:
        try:
            with open(args.pool_file, 'r') as f:
                pools = json.load(f)
                TREND_POOL = pools.get('trend', TREND_POOL)
                RANGE_POOL = pools.get('range', RANGE_POOL)
                print(f"{Fore.GREEN}✅ Loaded custom pools from {args.pool_file}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to load pool file: {e}{Style.RESET_ALL}")

    if not args.quiet:
        print_header("EINSTEIN FORTRESS V20.0 (DYNAMIC)")
        print(f"{Fore.CYAN}Settings: Seeds={args.n_seeds} | MC_RuinLimit={args.mc_ruin_limit}% | Jobs={args.jobs} | OutDir={args.output_dir}{Style.RESET_ALL}")

    symbols = list(set([m['id'].split('_')[0].upper() + "-USD" for m in ALL_AVAILABLE_MODELS]))
    if not symbols: symbols = ["BTC-USD", "ETH-USD"]
    
    # CLI or Interactive Selection
    sym = select_from_list("Symbol", sorted(symbols), args.symbol)
    tf = select_from_list("Timeframe", ["1h", "4h", "15m"], args.timeframe)
    
    # Smart Warning for ML
    relevant_models = [m['id'] for m in ALL_AVAILABLE_MODELS if sym.split('-')[0].lower() in m['id'].lower() and tf in m['id'].lower()]    
    if not relevant_models and len(ALL_AVAILABLE_MODELS) > 0 and not args.dry_run and not args.quiet:
        print(f"\n{Fore.MAGENTA}⚠️ WARNING: No ML models found for {sym} {tf}. Running in Pure TA Mode.{Style.RESET_ALL}")
    
    if args.trials:
        n_trials = args.trials
    else:
        try: n_trials = int(input(f"\n{Fore.YELLOW}Enter Max Trials (Default 5000): {Style.RESET_ALL}").strip() or 5000)
        except: n_trials = 5000
    
    dates = get_train_test_dates(years_back=2)

    study = optuna.create_study(
        directions=["maximize", "minimize"], 
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=200) 
    )
    try: 
        study.optimize(
            create_objective(sym, tf, dates, relevant_models, args), 
            n_trials=n_trials, 
            n_jobs=args.jobs
        )
    except KeyboardInterrupt: print("Paused.")
    
    if not args.quiet: print_header("SAVING ANALYSIS")
    try:
        # CSV Export
        df = study.trials_dataframe()
        rename_map = {
            "values_0": "Val_Return", 
            "values_1": "Val_DD", 
            "number": "Trial_ID",
            "state": "Status"
        }
        df.rename(columns=rename_map, inplace=True)
        # Keep params for reproducibility
        drop_cols = [c for c in df.columns if c.startswith("system_attrs") or c.startswith("user_attrs")]
        df.drop(columns=drop_cols, inplace=True, errors='ignore')

        output_path = args.output_dir if args.output_dir else RESULTS_DIR
        os.makedirs(output_path, exist_ok=True)
        
        csv_filename = f"Pareto_Analysis_{sym}_{tf}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        csv_path = os.path.join(output_path, csv_filename)
        df.to_csv(csv_path, index=False)
        if not args.quiet: print(f"{Fore.GREEN}✅ Pareto-Front Analysis Saved: {csv_path}{Style.RESET_ALL}")
        
        # VISUALIZATION EXPORT
        if HAS_VIZ:
            try:
                fig = plot_pareto_front(study, target_names=["Return", "Drawdown"])
                html_path = os.path.join(output_path, f"Pareto_Plot_{sym}_{tf}.html")
                fig.write_html(html_path)
                if not args.quiet: print(f"{Fore.GREEN}✅ Interactive Pareto Plot Saved: {html_path}{Style.RESET_ALL}")
            except Exception as ve:
                if not args.quiet: print(f"{Fore.YELLOW}⚠️ Visualization skipped: {ve}{Style.RESET_ALL}")

        if not args.quiet:
            print_header("TOP 5 DISCOVERED STRATEGIES")
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            top_trials = sorted(completed_trials, key=lambda t: t.values[0], reverse=True)[:5]
            
            for i, t in enumerate(top_trials):
                print(f"{Fore.CYAN}#{i+1}: Trial {t.number} | Ret: {t.values[0]:.2f}% | DD: {t.values[1]:.2f}%{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}❌ Failed to save Analysis: {e}{Style.RESET_ALL}")

if __name__ == "__main__": main()
