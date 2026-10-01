# File: client_v33_beginner.py
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

# --- OPTIMIZER SETTINGS ---
SEED = 12345
METRIC_TOLERANCE = 1e-6
REPRODUCIBILITY_RUNS = 3
EPSILON = 1e-9
TOTAL_TRIALS = 4000
PARALLEL_JOBS = 5 
RISK_FREE_RATE = 0.02
MIN_CALMAR_TO_CERTIFY = 0.05

# --- STRATEGY POOLS ---
TREND_STRATEGIES = ["sma_crossover", "macd_crossover", "ichimoku_cloud", "psar_signal", "obv_signal"]
RANGE_STRATEGIES = ["rsi_divergence", "stochastic_crossover", "bollinger_bands", "cci_oversold"]
ALL_AVAILABLE_MODELS = []

# --- BASE CONFIG ---
BASE_CONFIG = {
    "symbol": "btc-USD",
    "timeframe": "1d",
    "startDate": "2021-01-01", 
    "endDate": "2021-12-31",
    "initialBalance": 250,
    "fee": 0.006, # 0.6% Real Coinbase Fee
    "riskManagementMode": "standard",
    "riskPercentage": 1,
    "growthCapitalTarget": None,
    "params": {},
    "optimizer_mode": True
}

# --- LOGGING ---
optuna.logging.set_verbosity(optuna.logging.WARN)
optuna_logger = optuna.logging.get_logger("optuna")
if not optuna_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    optuna_logger.addHandler(handler)

# ==============================================================================
# 1. PROFESSIONAL REPORTING ENGINE
# ==============================================================================

def get_text_verdict(calmar, drawdown):
    if calmar <= 0: return "FAIL (Losing Money)"
    if drawdown > 50: return "HIGH RISK (Crash Prone)"
    if calmar > 1.0: return "EXCELLENT (High Reward/Low Risk)"
    if calmar > 0.5: return "GOOD (Solid Performance)"
    return "MARGINAL (Profitable but Weak)"

def print_report_card(metrics, title="STRATEGY REPORT"):
    tr = metrics.get('totalReturn', 0)
    pf = metrics.get('profitFactor', 0)
    wr = metrics.get('winRate', 0)
    dd = metrics.get('maxDrawdown', 0)
    calmar = metrics.get('calmarRatio', 0)
    trades = metrics.get('totalTrades', 0)
    
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)
    print(f" Total Return (Net Profit):           {tr:+.2f}%")
    print(f" Final Balance (Cash):                ${metrics.get('finalBalance', 0):.2f}")
    print("-" * 60)
    print(f" Profit Factor (Gross Win/Loss):      {pf:.2f}  (>1.5 is healthy)")
    print(f" Win Rate (Accuracy):                 {wr:.2f}%")
    print(f" Total Trades (Volume):               {trades}")
    print("-" * 60)
    print(f" Max Drawdown (Worst Crash):          {dd:.2f}%")
    print(f" Calmar Ratio (Reward vs Risk):       {calmar:.2f}")
    print("-" * 60)
    print(f" FINAL VERDICT: {get_text_verdict(calmar, dd)}")
    print("="*60 + "\n")

# ==============================================================================
# 2. API CLIENT FUNCTIONS
# ==============================================================================

def fetch_all_models():
    global ALL_AVAILABLE_MODELS
    url = f"{ML_SERVER_URL}/api/ml/available-models"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            ALL_AVAILABLE_MODELS = response.json()
            return True
        return False
    except: return False

def run_simulation(config):
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
    try:
        response = requests.post(url, json=config, timeout=600)
        if response.status_code != 200: return {}
        
        results = response.json()
        if "combinedResult" in results: results = results["combinedResult"]
        
        if "optimizer_ticket_id" in results:
            ticket_id = results["optimizer_ticket_id"]
            cache_path = os.path.join(OPTIMIZER_CACHE_DIR, f"{ticket_id}.json")
            for _ in range(20):
                if os.path.exists(cache_path):
                    with open(cache_path, 'r') as f: 
                        cached_res = json.load(f)
                        if "combinedResult" in cached_res: return cached_res["combinedResult"]
                        return cached_res
                time.sleep(1)
            return {}
        return results
    except: return {}

def run_certification(market_id, best_params, base_config):
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
        else: set_nested(clean_params, key, value)

    payload = {"symbol": symbol.upper(), "timeframe": timeframe, "best_params": clean_params, "base_config": base_config}
    url = f"{ML_SERVER_URL}/api/ml/certify-strategy"
    try:
        response = requests.post(url, json=payload, timeout=900)
        if response.status_code == 200: return response.json()
    except: pass
    return None

def start_live_bot(config_dict):
    url = f"{ML_SERVER_URL}/api/bot/start"
    print(f"   >> Sending start command to Python Brain...")
    try:
        response = requests.post(url, json=config_dict, timeout=10)
        if response.status_code == 200: print(f"   >> Bot Successfully Started!")
        else: print(f"   >> Failed: {response.json()}")
    except: print("   >> Connection Failed")

def stop_live_bot():
    url = f"{ML_SERVER_URL}/api/bot/stop"
    try:
        requests.post(url, timeout=10)
        print("   >> Bot Stopped.")
    except: print("   >> Failed to stop")

def get_bot_status():
    url = f"{ML_SERVER_URL}/api/bot/status"
    try:
        response = requests.get(url, timeout=10)
        status = response.json()
        print("\n" + "-"*30)
        print(" LIVE BOT MONITOR")
        print("-" * 30)
        print(f" Status:      {status.get('status').upper()}")
        print(f" Balance:     ${status.get('currentBalance', 0):.2f}")
        print(f" Position:    {status.get('position')}")
        
        metrics = status.get('performanceMetrics', {})
        print(f" PnL:         ${metrics.get('totalProfit', 0):.2f}")
        print(f" Max DD:      {metrics.get('maxDrawdown', 0):.2f}%")
        print(f" Trades:      {metrics.get('totalTrades', 0)}")
        
        print("-" * 30)
        print(" RECENT ACTIVITY:")
        for log in status.get('logs', [])[:5]:
            print(f" > [{log['timestamp']}] {log['message']}")
        print("-" * 30 + "\n")
    except: print("   >> Cannot connect to Bot.")

# ==============================================================================
# 3. OPTIMIZER HELPERS
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
    drawdown = (stitched_df['balance_continuous'] - peak) / (peak + 1e-9)
    max_dd = abs(drawdown.min() * 100)
    days = (pd.to_datetime(stitched_df['timestamp'].iloc[-1]) - pd.to_datetime(stitched_df['timestamp'].iloc[0])).days
    if days < 1: days = 1
    annual_return = ((final / initial) ** (365.25 / days)) - 1
    calmar = (annual_return * 100) / (max_dd + 1e-9)
    return {"StitchedCalmarRatio": calmar, "StitchedMaxDrawdown": max_dd}

def get_user_market_selection():
    if not ALL_AVAILABLE_MODELS: return [("btc-USD", "4h")]
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
    if not sorted_pairs: return [("btc-USD", "4h")]

    print("\n--- Select Market ---")
    for i, (symbol, timeframe) in enumerate(sorted_pairs):
        print(f"  [{i+1}] {symbol} @ {timeframe}")
    print("  [0] TEST ALL")
    
    while True:
        choice_str = input("Choice: ")
        if not choice_str: continue
        if choice_str.strip() == "0": return sorted_pairs
        try:
            choices = [int(c.strip()) for c in choice_str.split(',')]
            selected_pairs = []
            for c in choices:
                if 1 <= c <= len(sorted_pairs): selected_pairs.append(sorted_pairs[c-1])
            if selected_pairs: return selected_pairs
        except: continue

def run_initial_probe(symbol, timeframe, first_fold, market_models):
    """Runs initial probe: ML Off, ML Hybrid, ML On for each strategy."""
    base_config = deepcopy(BASE_CONFIG)
    base_config.update({
        'symbol': symbol.upper(), 'timeframe': timeframe,
        'startDate': first_fold['startDate'], 'endDate': first_fold['endDate'],
        'mlModel': market_models[0] if market_models else None,
        'optimizer_mode': False 
    })
    
    print(f"\n{'='*60}")
    print(f" 🔎 INITIAL PROBE: {symbol} {timeframe}")
    print(f"    Testing all strategies across 3 ML Modes")
    print(f"{'='*60}")
    print(f"{'Strategy':<25} | {'Mode':<12} | {'Calmar (Reward/Risk)':<20} | {'Max Drawdown':<12}")
    print("-" * 75)

    best_trend, best_range = "sma_crossover", "rsi_divergence"
    best_calmar_t, best_calmar_r = -999, -999

    modes = ['off', 'predictions', 'on']
    
    for strat_code in TREND_STRATEGIES + RANGE_STRATEGIES:
        for mode in modes:
            # Config
            base_config['mlMode'] = mode
            base_config['strategies'] = [{"code": strat_code, "params": {}}]
            
            try:
                res = run_simulation(base_config)
                m = res.get('metrics', {})
                calmar = m.get('calmarRatio', -999)
                dd = m.get('maxDrawdown', 100)
                
                # Print Row
                print(f"{strat_code:<25} | {mode:<12} | {calmar:8.2f}             | {dd:6.2f}%")

                # Determine Best Seeding Strategy (Trend/Range) based on ML OFF performance (purest signal)
                if mode == 'off':
                    if strat_code in TREND_STRATEGIES and calmar > best_calmar_t:
                        best_trend = strat_code; best_calmar_t = calmar
                    elif strat_code in RANGE_STRATEGIES and calmar > best_calmar_r:
                        best_range = strat_code; best_calmar_r = calmar
            except: 
                print(f"{strat_code:<25} | {mode:<12} | {'ERROR':<20} | {'--':<12}")

    print("-" * 75)
    print(f" 👉 Selecting best seed pair: {best_trend} + {best_range}")
    return best_trend, best_range

def create_objective(symbol, timeframe, wfo_folds, market_models):
    def objective(trial: optuna.Trial):
        test_config = deepcopy(BASE_CONFIG)
        test_config['symbol'] = symbol.upper()
        test_config['timeframe'] = timeframe
        
        trend_strat = trial.suggest_categorical("trend_strategy", TREND_STRATEGIES)
        range_strat = trial.suggest_categorical("range_strategy", RANGE_STRATEGIES)
        
        test_config['strategies'] = [{"code": trend_strat, "params": {}}, {"code": range_strat, "params": {}}]
        set_nested_value(test_config, "params.hybridMode", "REGIME")
        
        set_nested_value(test_config, "params.regime_threshold", trial.suggest_int("params.regime_threshold", 15, 40, step=5))
        set_nested_value(test_config, "params.minAdxLevel", trial.suggest_int("params.minAdxLevel", 0, 20, step=5))
        test_config['mlMode'] = "predictions"
        test_config['mlModel'] = trial.suggest_categorical("mlModel", market_models)
        test_config['mlThreshold'] = trial.suggest_float("mlThreshold", 0.55, 0.70, step=0.05)
        tsl_mult = trial.suggest_float("params.tslAtrMult", 1.5, 6.0, step=0.5)
        set_nested_value(test_config, "params.tslAtrMult", tsl_mult)
        
        if "bollinger_bands" in [trend_strat, range_strat]:
             set_nested_value(test_config, "params.bb_length", trial.suggest_int("params.bb_length", 15, 30))
             set_nested_value(test_config, "params.bb_std", trial.suggest_float("params.bb_std", 1.5, 2.5, step=0.5))
        if "rsi_divergence" in [trend_strat, range_strat]:
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
        
        print(f"  [Trial {trial.number}] {trend_strat}/{range_strat} | Calmar: {calmar:.2f} | Max DD: {dd:.2f}%")
        return calmar, dd
    return objective

# ==============================================================================
# 4. MAIN MENU
# ==============================================================================

def main():
    fetch_all_models()

    while True:
        print("\n" + "="*40)
        print(" 🧠 INTELLIGENT MENDEL: COMMAND CENTER")
        print("="*40)
        print(" 1. 🧬 FIND STRATEGIES (Run Optimizer)")
        print(" 2. 🤖 LAUNCH LIVE BOT (Start)")
        print(" 3. 🛑 KILL LIVE BOT (Stop)")
        print(" 4. 📊 LIVE MONITOR (Check Status)")
        print(" 5. 📉 QUICK BACKTEST (Verify Setup)")
        print(" 0. Exit")
        
        choice = input("\nSelect option: ")
        
        if choice == "1":
            markets = get_user_market_selection()
            for symbol, timeframe in markets:
                market_id = f"{symbol}_{timeframe}"
                wfo_folds = generate_wfo_folds()
                m_check = f"{symbol.split('-')[0]}_{timeframe}"
                ml_models = [m['id'] for m in ALL_AVAILABLE_MODELS if m['id'].lower().startswith(m_check.lower())]
                
                if not ml_models:
                    print(f"⚠️ No ML models found for {symbol} {timeframe}. Skipping.")
                    continue

                T_best, R_best = run_initial_probe(symbol, timeframe, wfo_folds[0], ml_models)
                
                print(f"\n🚀 STARTING OPTIMIZATION: {symbol} {timeframe}")
                print(f"   - Trials: {TOTAL_TRIALS}")
                print(f"   - Parallel Jobs: {PARALLEL_JOBS}")
                print(f"   - Fee Reality: {BASE_CONFIG['fee']*100:.1f}% (Coinbase Taker)\n")
                
                study_name = f"{market_id}_STUDY_{datetime.now().strftime('%H%M')}"
                study_db = f"sqlite:///{RESULTS_DIR}/{study_name}.db"
                study = optuna.create_study(study_name=study_name, storage=study_db, directions=["maximize", "minimize"])
                
                study.enqueue_trial({
                    "trend_strategy": T_best, "range_strategy": R_best,
                    "params.regime_threshold": 25, "mlModel": ml_models[0], "mlThreshold": 0.65
                })

                try:
                    study.optimize(create_objective(symbol, timeframe, wfo_folds, ml_models), n_trials=TOTAL_TRIALS, n_jobs=PARALLEL_JOBS)
                except KeyboardInterrupt:
                    print("\n⚠️ Optimization paused by user.")
                
                print("\n[Harvest] Analyzing best results...")
                completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                
                if not completed_trials:
                    print("❌ No trials completed successfully.")
                else:
                    completed_trials.sort(key=lambda x: x.values[0], reverse=True)
                    best_of_all = completed_trials[0]
                    winners = [t for t in completed_trials if t.values[0] > MIN_CALMAR_TO_CERTIFY]
                    candidates = winners[:5] if winners else completed_trials[:3]
                    
                    print(f"\n🏅 BEST STRATEGY FOUND:")
                    print(f"   Strategy: {best_of_all.user_attrs.get('combo_strategies')}")
                    print(f"   Calmar (Reward/Risk):      {best_of_all.values[0]:.4f}")
                    print(f"   Max Drawdown (Crash Risk): {best_of_all.values[1]:.2f}%")
                    
                    for i, trial in enumerate(candidates):
                        params = deepcopy(trial.params)
                        trend = params.pop("trend_strategy", "")
                        rng = params.pop("range_strategy", "")
                        combo_str = f"{trend},{rng}"
                        params['combo_strategies'] = combo_str
                        params['hybridMode'] = "REGIME"
                        
                        print(f"\n🔎 Checking Candidate #{i+1}: {combo_str}")
                        cert = run_certification(market_id, params, BASE_CONFIG)
                        
                        if cert:
                            m = cert.get('holdout_metrics', {})
                            passed = cert.get('certification_passed')
                            
                            print_report_card(m, title=f"CERTIFICATION REPORT: {combo_str}")
                            
                            if i == 0: # Always save best result found
                                print(f"   💾 SAVED to winner_{market_id}.json")
                                winner_file = os.path.join(RESULTS_DIR, f"winner_{market_id}.json")
                                with open(winner_file, "a") as f:
                                    json.dump(params, f)
                                    f.write("\n")

        elif choice == "2":
            # Bot logic (same as before)
            print("\n--- Deploy Bot ---")
            bot_config = {
                "symbol": "BTC-USD", "timeframe": "4h", "capitalAllocation": 1000,
                "mlMode": "predictions", "mlModel": "btc_4h_xgboost_model", "mlThreshold": 0.65,
                "isCombo": True, "strategies": [{"code": "macd_crossover", "params": {}}, {"code": "bollinger_bands", "params": {"bb_length": 30, "bb_std": 1.5}}],
                "params": {"hybridMode": "REGIME", "regime_threshold": 15, "minAdxLevel": 5, "tslAtrMult": 6.0}
            }
            start_live_bot(bot_config)

        elif choice == "3": stop_live_bot()
        elif choice == "4": get_bot_status()
        elif choice == "5":
            # Backtest logic (same as before)
             print("\n--- Quick Backtest ---")
             single_config = {
                "symbol": "BTC-USD", "timeframe": "4h", "startDate": "2021-01-01", "endDate": "2025-11-21",
                "initialBalance": 300, "fee": 0.006, "riskManagementMode": "standard", "riskPercentage": 1.0,
                "mlMode": "predictions", "mlModel": "btc_4h_xgboost_model", "mlThreshold": 0.65,
                "strategies": [{"code": "macd_crossover", "params": {}}, {"code": "bollinger_bands", "params": {"bb_length": 30, "bb_std": 1.5}}],
                "params": {"hybridMode": "REGIME", "regime_threshold": 15, "minAdxLevel": 5, "tslAtrMult": 6.0}
             }
             res = run_simulation(single_config)
             if res.get("metrics"): print_report_card(res["metrics"], title="QUICK BACKTEST REPORT")
             else: print("❌ Error running backtest.")

        elif choice == "0": break

if __name__ == "__main__":
    main()
