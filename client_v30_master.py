# File: client_v30_ultimate.py
import requests
import json
import pandas as pd
import numpy as np
import warnings
from copy import deepcopy
import time
import optuna
from datetime import datetime, timezone
import os
import random
import sys
import logging

# --- CONFIGURATION ---
ML_SERVER_URL = "http://74.208.28.77:8000"
RESULTS_DIR = "/root/Project/ML/data/optimizer_results"
OPTIMIZER_CACHE_DIR = "/root/Project/ML/cache/optimizer/"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OPTIMIZER_CACHE_DIR, exist_ok=True)

# --- OPTIMIZER CONSTANTS ---
SEED = 12345
METRIC_TOLERANCE = 1e-6
REPRODUCIBILITY_RUNS = 3
EPSILON = 1e-9
TOTAL_TRIALS = 5000  # <--- THIS IS THE NUMBER YOU REQUESTED
PARALLEL_JOBS = 4 
RISK_FREE_RATE = 0.02
MIN_CALMAR_TO_CERTIFY = 0.05

# --- STRATEGY POOLS ---
TREND_STRATEGIES = ["sma_crossover", "macd_crossover", "ichimoku_cloud", "psar_signal", "obv_signal"]
RANGE_STRATEGIES = ["rsi_divergence", "stochastic_crossover", "bollinger_bands", "cci_oversold"]
ALL_AVAILABLE_MODELS = []

# --- BASE CONFIG ---
BASE_CONFIG = {
    "symbol": "btc-USD",
    "timeframe": "1h",
    "startDate": "2021-01-01", 
    "endDate": "2021-12-31",
    "initialBalance": 300,
    "fee": 0.006, # 0.6% Real Coinbase Fee
    "riskManagementMode": "standard",
    "riskPercentage": 1,
    "growthCapitalTarget": None,
    "params": {},
    "optimizer_mode": True
}

# --- LOGGING ---
optuna.logging.set_verbosity(optuna.logging.INFO)
optuna_logger = optuna.logging.get_logger("optuna")
if not optuna_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    optuna_logger.addHandler(handler)

# ==============================================================================
# 1. API CLIENT FUNCTIONS
# ==============================================================================

def fetch_all_models():
    """GET /api/ml/available-models"""
    global ALL_AVAILABLE_MODELS
    url = f"{ML_SERVER_URL}/api/ml/available-models"
    print(f"[API] Fetching models from {url}...")
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            ALL_AVAILABLE_MODELS = response.json()
            print(f"[API] Found {len(ALL_AVAILABLE_MODELS)} models.")
            return True
        else:
            print(f"[API] ERROR: {response.text}")
            return False
    except Exception as e:
        print(f"[API] Connection Failed: {e}")
        return False

def run_simulation(config):
    """POST /api/ml/run-combo-backtest"""
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
    try:
        response = requests.post(url, json=config, timeout=600)
        if response.status_code != 200: raise Exception(f"Server Error: {response.text}")
        
        results = response.json()
        # Unwrap 'combinedResult' if present
        if "combinedResult" in results: results = results["combinedResult"]

        # Handle Caching
        if "optimizer_ticket_id" in results:
            ticket_id = results["optimizer_ticket_id"]
            cache_path = os.path.join(OPTIMIZER_CACHE_DIR, f"{ticket_id}.json")
            for _ in range(20): # Wait up to 20s
                if os.path.exists(cache_path):
                    with open(cache_path, 'r') as f: 
                        cached_res = json.load(f)
                        if "combinedResult" in cached_res: return cached_res["combinedResult"]
                        return cached_res
                time.sleep(1)
            raise Exception("Cache file timeout.")
        return results
    except Exception as e:
        raise e

def run_certification(market_id, best_params, base_config):
    """POST /api/ml/certify-strategy"""
    symbol, timeframe = market_id.split('_')
    clean_params = {}
    
    def set_nested(d, keys, value):
        keys = keys.split('.')
        for key in keys[:-1]: d = d.setdefault(key, {})
        d[keys[-1]] = value

    for key, value in best_params.items():
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)): continue
        if key in ["CalmarRatio", "MaxDrawdown"]: continue
        if key.startswith("params."):
            param_key = key.split('.', 1)[1]
            if 'params' not in clean_params: clean_params['params'] = {}
            set_nested(clean_params['params'], param_key, value)
        else:
            set_nested(clean_params, key, value)

    payload = {"symbol": symbol.upper(), "timeframe": timeframe, "best_params": clean_params, "base_config": base_config}
    url = f"{ML_SERVER_URL}/api/ml/certify-strategy"
    
    for _ in range(3):
        try:
            response = requests.post(url, json=payload, timeout=900)
            if response.status_code == 200: return response.json()
        except: time.sleep(2)
    return None

# --- LIVE BOT ENDPOINTS ---

def start_live_bot(config_dict):
    url = f"{ML_SERVER_URL}/api/bot/start"
    print(f"[Bot] Starting with config: {json.dumps(config_dict, indent=2)}")
    try:
        response = requests.post(url, json=config_dict, timeout=10)
        print(f"[Bot] Response: {response.json()}")
    except Exception as e:
        print(f"[Bot] Failed to start: {e}")

def stop_live_bot():
    url = f"{ML_SERVER_URL}/api/bot/stop"
    try:
        response = requests.post(url, timeout=10)
        print(f"[Bot] Response: {response.json()}")
    except Exception as e:
        print(f"[Bot] Failed to stop: {e}")

def get_bot_status():
    url = f"{ML_SERVER_URL}/api/bot/status"
    try:
        response = requests.get(url, timeout=10)
        status = response.json()
        print("\n=== LIVE BOT STATUS ===")
        print(f"Status: {status.get('status')}")
        print(f"Balance: ${status.get('currentBalance', 0):.2f}")
        print(f"Position: {status.get('position')}")
        metrics = status.get('performanceMetrics', {})
        print(f"PnL: ${metrics.get('totalProfit', 0):.2f}")
        print(f"Trades: {metrics.get('totalTrades', 0)}")
        print(f"Max DD: {metrics.get('maxDrawdown', 0):.2f}%")
        print("\n--- Recent Logs ---")
        for log in status.get('logs', [])[:5]:
            print(f"[{log['timestamp']}] {log['message']}")
        print("=======================\n")
    except Exception as e:
        print(f"[Bot] Failed to get status: {e}")

# ==============================================================================
# 2. OPTIMIZER HELPERS
# ==============================================================================

def set_nested_value(d, keys, value):
    keys = keys.split('.')
    for key in keys[:-1]: d = d.setdefault(key, {})
    d[keys[-1]] = value

def generate_wfo_folds():
    now = datetime.now(timezone.utc)
    start_year = 2021
    folds = []
    for year in range(start_year, now.year):
        folds.append({"startDate": f"{year}-01-01", "endDate": f"{year}-12-31"})
    folds.append({"startDate": f"{now.year}-01-01", "endDate": now.strftime("%Y-%m-%d")})
    return folds

def calculate_stitched_metrics(equity_curves):
    if not equity_curves: return {}
    all_dfs = []
    last_balance = 0
    for i, fold_curve in enumerate(equity_curves):
        if not fold_curve: continue
        df = pd.DataFrame(fold_curve)
        if 'balance' not in df.columns or df.empty: continue
        if i == 0:
            df['balance_continuous'] = df['balance']
            last_balance = df['balance'].iloc[-1]
        else:
            initial_fold_balance = fold_curve[0]['balance']
            df['balance_continuous'] = df['balance'] - initial_fold_balance + last_balance
            last_balance = df['balance_continuous'].iloc[-1]
        all_dfs.append(df)
    if not all_dfs: return {"totalTrades": 0}
    
    stitched_df = pd.concat(all_dfs, ignore_index=True)
    initial = stitched_df['balance_continuous'].iloc[0]
    final = stitched_df['balance_continuous'].iloc[-1]
    if final <= 0 or initial <= 0: return {"StitchedCalmarRatio": -999, "StitchedMaxDrawdown": 100.0}
        
    peak = stitched_df['balance_continuous'].cummax()
    drawdown = (stitched_df['balance_continuous'] - peak) / (peak + EPSILON)
    max_dd = abs(drawdown.min() * 100)
    
    days = (pd.to_datetime(stitched_df['timestamp'].iloc[-1]) - pd.to_datetime(stitched_df['timestamp'].iloc[0])).days
    if days < 1: days = 1
    annual_return = ((final / initial) ** (365.25 / days)) - 1
    calmar = (annual_return * 100) / (max_dd + EPSILON)
    return {"StitchedCalmarRatio": calmar, "StitchedMaxDrawdown": max_dd}

def get_user_market_selection():
    if not ALL_AVAILABLE_MODELS:
        print("[Menu] No models found. Defaulting to btc-USD / 1h.")
        return [("btc-USD", "1h")]
    
    market_pairs = set()
    for model in ALL_AVAILABLE_MODELS:
        try:
            parts = model['id'].split('_')
            if len(parts) >= 2:
                symbol = f"{parts[0]}-USD"
                timeframe = parts[1]
                market_pairs.add((symbol, timeframe))
        except: pass
    
    sorted_pairs = sorted(list(market_pairs))
    if not sorted_pairs: return [("btc-USD", "1h")]

    print("\n--- Available Markets ---")
    for i, (symbol, timeframe) in enumerate(sorted_pairs):
        print(f"  [{i+1}] {symbol} @ {timeframe}")
    print("  [0] TEST ALL")

    while True:
        choice_str = input("Which market(s) to test? (e.g., 1,3): ")
        if not choice_str: continue
        if choice_str.strip() == "0": return sorted_pairs
        try:
            choices = [int(c.strip()) for c in choice_str.split(',')]
            selected_pairs = []
            for c in choices:
                if 1 <= c <= len(sorted_pairs):
                    selected_pairs.append(sorted_pairs[c-1])
            if selected_pairs: return selected_pairs
        except ValueError: continue

def run_initial_probe(symbol, timeframe, first_fold, market_models):
    base_config = deepcopy(BASE_CONFIG)
    base_config.update({
        'symbol': symbol.upper(),
        'timeframe': timeframe,
        'startDate': first_fold['startDate'],
        'endDate': first_fold['endDate'],
        'mlMode': 'predictions', 
        'mlThreshold': 0.60, 
        'mlModel': market_models[0] if market_models else None,
        'optimizer_mode': False 
    })
    
    best_trend = {"name": "", "calmar": -999}
    best_range = {"name": "", "calmar": -999}

    print("\n[Probe] Running initial 1x backtests to find best base strategy...")
    
    # 1. Run ML OFF Probe to see raw performance
    base_config['mlMode'] = 'off'
    for strat_code in TREND_STRATEGIES + RANGE_STRATEGIES:
        base_config['strategies'] = [{"code": strat_code, "params": {}}]
        try:
            res = run_simulation(base_config)
            calmar = res.get('metrics', {}).get('calmarRatio', -999)
            dd = res.get('metrics', {}).get('maxDrawdown', 100)
            print(f"  -> {strat_code} (ML Off): Calmar {calmar:.2f} | DD {dd:.2f}%")
            if strat_code in TREND_STRATEGIES and calmar > best_trend['calmar']:
                best_trend.update({"name": strat_code, "calmar": calmar})
            elif strat_code in RANGE_STRATEGIES and calmar > best_range['calmar']:
                best_range.update({"name": strat_code, "calmar": calmar})
        except: pass

    if not best_trend['name'] or not best_range['name']:
        print("[Probe] WARNING: Failed to identify. Using default SMA/RSI.")
        return "sma_crossover", "rsi_divergence"
        
    print(f"[Probe] Best Pair Found: {best_trend['name']} (Trend) / {best_range['name']} (Range).")
    return best_trend['name'], best_range['name']

def create_objective(symbol, timeframe, wfo_folds, market_models):
    def objective(trial: optuna.Trial):
        test_config = deepcopy(BASE_CONFIG)
        test_config['symbol'] = symbol.upper()
        test_config['timeframe'] = timeframe
        
        trend_strat = trial.suggest_categorical("trend_strategy", TREND_STRATEGIES)
        range_strat = trial.suggest_categorical("range_strategy", RANGE_STRATEGIES)
        
        test_config['strategies'] = [{"code": trend_strat, "params": {}}, {"code": range_strat, "params": {}}]
        set_nested_value(test_config, "params.hybridMode", "REGIME")
        regime_thresh = trial.suggest_int("params.regime_threshold", 15, 40, step=5)
        set_nested_value(test_config, "params.regime_threshold", regime_thresh)
        adx_filter = trial.suggest_int("params.minAdxLevel", 0, 20, step=5)
        set_nested_value(test_config, "params.minAdxLevel", adx_filter)
        
        test_config['mlMode'] = "predictions"
        test_config['mlModel'] = trial.suggest_categorical("mlModel", market_models)
        test_config['mlThreshold'] = trial.suggest_float("mlThreshold", 0.55, 0.70, step=0.05)
        
        tsl_mult = trial.suggest_float("params.tslAtrMult", 1.5, 6.0, step=0.5)
        set_nested_value(test_config, "params.tslAtrMult", tsl_mult)
        
        active_strats = [trend_strat, range_strat]
        if "bollinger_bands" in active_strats:
             set_nested_value(test_config, "params.bb_length", trial.suggest_int("params.bb_length", 15, 30))
             set_nested_value(test_config, "params.bb_std", trial.suggest_float("params.bb_std", 1.5, 2.5, step=0.5))
        if "rsi_divergence" in active_strats:
             set_nested_value(test_config, "params.rsi_length", trial.suggest_int("params.rsi_length", 10, 25))
        
        all_curves = []
        for fold in wfo_folds:
            fold_config = deepcopy(test_config)
            fold_config['startDate'] = fold['startDate']
            fold_config['endDate'] = fold['endDate']
            try:
                res = run_simulation(fold_config)
                if res.get('metrics', {}).get('totalTrades', 0) < 2: raise optuna.TrialPruned()
                all_curves.append(res.get('equityCurve'))
            except: raise optuna.TrialPruned()
        
        metrics = calculate_stitched_metrics(all_curves)
        calmar = metrics.get('StitchedCalmarRatio', -999)
        dd = metrics.get('StitchedMaxDrawdown', 100)
        
        trial.set_user_attr("combo_strategies", f"{trend_strat},{range_strat}")
        # 🚀 RESTORED LOGGING
        print(f"  [Trial {trial.number}] {trend_strat}/{range_strat} | Calmar: {calmar:.2f} | DD: {dd:.2f}%")
        return calmar, dd
    return objective

# ==============================================================================
# 4. MAIN MENU
# ==============================================================================

def main():
    print("\n=== 🧠 INTELLIGENT MENDEL COMMAND CENTER ===")
    
    # Fetch models immediately for better UX
    if not fetch_all_models(): 
        print("Server not ready or no models.")

    while True:
        print("\n1. 🧬 Run Optimizer (Find Strategies)")
        print("2. 🤖 Start Live Paper Bot")
        print("3. 🛑 Stop Live Bot")
        print("4. 📊 Check Bot Status")
        print("5. 📉 Run Single Backtest (Winner #5 Config)")
        print("0. Exit")
        
        choice = input("\nSelect option: ")
        
        if choice == "1":
            markets = get_user_market_selection()
            for symbol, timeframe in markets:
                market_id = f"{symbol}_{timeframe}"
                wfo_folds = generate_wfo_folds()
                m_check = f"{symbol.split('-')[0]}_{timeframe}"
                ml_models = [m['id'] for m in ALL_AVAILABLE_MODELS if m['id'].lower().startswith(m_check.lower())]
                
                if not ml_models:
                    print("No ML Models found for this market.")
                    continue

                # Probe First
                T_best, R_best = run_initial_probe(symbol, timeframe, wfo_folds[0], ml_models)
                
                study_name = f"{market_id}_STUDY_{datetime.now().strftime('%H%M')}"
                study_db = f"sqlite:///{RESULTS_DIR}/{study_name}.db"
                study = optuna.create_study(study_name=study_name, storage=study_db, directions=["maximize", "minimize"])
                
                # Seed
                study.enqueue_trial({
                    "trend_strategy": T_best, "range_strategy": R_best,
                    "params.regime_threshold": 25, "mlModel": ml_models[0], "mlThreshold": 0.65
                })

                # 🚀 NEW: Configuration Dashboard Print
                print("\n" + "="*60)
                print(f" 🚀 Starting REGIME Optimization Study")
                print(f" 🎯 Market:         {symbol} @ {timeframe}")
                print(f" ⏱️ Total Trials:     {TOTAL_TRIALS}")
                print(f" ⚡ Parallel Jobs:  {PARALLEL_JOBS}")
                print(f" 💰 Fee Mode:       {BASE_CONFIG['fee']*100:.1f}%")
                print("="*60 + "\n")

                try:
                    study.optimize(create_objective(symbol, timeframe, wfo_folds, ml_models), n_trials=TOTAL_TRIALS, n_jobs=PARALLEL_JOBS)
                except KeyboardInterrupt:
                    print("Optimization paused.")
                
                # Harvest
                print("\n[Harvest] Checking results...")
                best_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values[0] > MIN_CALMAR_TO_CERTIFY]
                
                for trial in best_trials:
                    params = deepcopy(trial.params)
                    trend = params.pop("trend_strategy", "")
                    rng = params.pop("range_strategy", "")
                    combo_str = f"{trend},{rng}"
                    params['combo_strategies'] = combo_str
                    params['hybridMode'] = "REGIME"
                    
                    print(f" Checking Candidate: {combo_str} (Calmar: {trial.values[0]:.2f})...")
                    cert = run_certification(market_id, params, BASE_CONFIG)
                    
                    if cert and cert.get('certification_passed'):
                        print(f"  🎉 WINNER FOUND! Strategy: {combo_str}")
                        winner_file = os.path.join(RESULTS_DIR, f"winner_{market_id}.json")
                        with open(winner_file, "a") as f:
                            json.dump(params, f)
                            f.write("\n")

        elif choice == "2":
            print("\n--- Deploy Bot ---")
            # (Same Bot Logic as before)
            bot_config = {
                "symbol": "BTC-USD",
                "timeframe": "1h",
                "capitalAllocation": 1000,
                "mlMode": "predictions",
                "mlModel": "btc_1h_xgboost_model",
                "mlThreshold": 0.65,
                "isCombo": True,
                "strategies": [
                    {"code": "psar_signal", "params": {}},
                    {"code": "bollinger_bands", "params": {"bb_length": 28, "bb_std": 2.5}}
                ],
                "params": {
                    "hybridMode": "REGIME",
                    "regime_threshold": 15,
                    "minAdxLevel": 20,
                    "tslAtrMult": 4.5,
                    "trendFilterPeriod": 200
                }
            }
            start_live_bot(bot_config)

        elif choice == "3":
            stop_live_bot()

        elif choice == "4":
            get_bot_status()

        elif choice == "5":
            print("\n--- Running Single Backtest (Winner #5 Config) ---")
            # (Same Single Backtest Logic)
            single_config = {
                "symbol": "BTC-USD",
                "timeframe": "1h",
                "startDate": "2021-01-01",
                "endDate": "2025-11-21",
                "initialBalance": 300,
                "fee": 0.006, 
                "riskManagementMode": "standard",
                "riskPercentage": 1.0,
                "mlMode": "predictions",
                "mlModel": "btc_1h_xgboost_model",
                "mlThreshold": 0.65,
                "strategies": [
                    {"code": "psar_signal", "params": {}},
                    {"code": "bollinger_bands", "params": {"bb_length": 28, "bb_std": 2.5}}
                ],
                "params": {
                    "hybridMode": "REGIME",
                    "regime_threshold": 15,
                    "minAdxLevel": 20,
                    "tslAtrMult": 4.5,
                    "trendFilterPeriod": 200
                },
                "optimizer_mode": False
            }
            try:
                res = run_simulation(single_config)
                m = res.get("metrics", {})
                print("\n📊 RESULTS 📊")
                print(f"Total Return: {m.get('totalReturn', 0):.2f}%")
                print(f"Final Balance: ${m.get('finalBalance', 0):.2f}")
                print(f"Profit Factor: {m.get('profitFactor', 0):.2f}")
                print(f"Win Rate: {m.get('winRate', 0):.2f}%")
                print(f"Max Drawdown: {m.get('maxDrawdown', 0):.2f}%")
                print(f"Total Trades: {m.get('totalTrades', 0)}")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == "0":
            break

if __name__ == "__main__":
    main()
