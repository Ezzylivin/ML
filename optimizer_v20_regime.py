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
import argparse
import sys

# --- 🚀 GRADE IMPORTER ---
try:
    from data_quality_grader import grade_strategy_quality
except ImportError:
    print("[Optimizer] WARNING: data_quality_grader.py not found. Final grading will be skipped.")
    def grade_strategy_quality(metrics, reproducibility_pass, certification_pass):
        print("[Optimizer] Grading skipped.")
        return {"grade": "N/A", "reliability": 0}

# --- 🚀 Deterministic Seeding ---
SEED = 12345
METRIC_TOLERANCE = 1e-6
REPRODUCIBILITY_RUNS = 3
EPSILON = 1e-9
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.INFO)

# --- CONFIGURATION ---
ML_SERVER_URL = "http://74.208.28.77:8000"
RESULTS_DIR = "/root/Project/ML/data/optimizer_results"
OPTIMIZER_CACHE_DIR = "/root/Project/ML/cache/optimizer/"
TOTAL_TRIALS = 1750 
PARALLEL_JOBS = 6 
RISK_FREE_RATE = 0.02
MIN_CALMAR_TO_CERTIFY = 0.05
MAX_DD_TO_CERTIFY = 30.0

# --- 🚀 STRATEGY POOLS (Regime Mode) ---
TREND_STRATEGIES = ["sma_crossover", "macd_crossover", "ichimoku_cloud", "psar_signal", "obv_signal"]
RANGE_STRATEGIES = ["rsi_divergence", "stochastic_crossover", "bollinger_bands", "cci_oversold"]

ALL_AVAILABLE_MODELS = []

BASE_CONFIG = {
    "symbol": "btc-USD",
    "timeframe": "1h",
    "startDate": "2021-01-01", # Fixed Start
    "endDate": "2021-12-31",
    "initialBalance": 300,
    "fee": 0.006,
    "riskManagementMode": "standard",
    "riskPercentage": 1,
    "growthCapitalTarget": None,
    "params": {},
    "optimizer_mode": True
}

def set_nested_value(d, keys, value):
    keys = keys.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def fetch_all_models():
    global ALL_AVAILABLE_MODELS
    print(f"[Optimizer] Fetching all models from {ML_SERVER_URL}...")
    try:
        url = f"{ML_SERVER_URL}/api/ml/available-models"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            ALL_AVAILABLE_MODELS = response.json()
            print(f"[Optimizer] Found {len(ALL_AVAILABLE_MODELS)} models.")
            return True
        else:
            print(f"[Optimizer] ERROR fetching models: {response.text}")
            return False
    except Exception as e:
        print(f"[Optimizer] CRITICAL ERROR: {e}")
        return False

def get_user_market_selection():
    if not ALL_AVAILABLE_MODELS:
        print("[MarketScan] No models found. Defaulting to btc-USD / 1h.")
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
    if not sorted_pairs:
        return [("btc-USD", "1h")]

    print("\n--- Available Markets ---")
    for i, (symbol, timeframe) in enumerate(sorted_pairs):
        print(f"  [{i+1}] {symbol} @ {timeframe}")
    print("  [0] TEST ALL")

    while True:
        choice_str = input("Which market(s) to test? (e.g., 1,3,4): ")
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

def generate_wfo_folds():
    now = datetime.now(timezone.utc)
    current_year = now.year
    start_year = 2021 # HARDCODED START
    folds = []
    for year in range(start_year, current_year):
        folds.append({"startDate": f"{year}-01-01", "endDate": f"{year}-12-31"})
    
    # Add current year partial
    today_str = now.strftime("%Y-%m-%d")
    folds.append({"startDate": f"{current_year}-01-01", "endDate": today_str})
    
    print(f"[WFO] Generated {len(folds)} folds starting from {start_year}.")
    return folds

def run_single_backtest(config):
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
    try:
        response = requests.post(url, json=config, timeout=600)
        if response.status_code != 200:
            # print(f"[DEBUG] Server Error ({response.status_code}): {response.text}")
            raise optuna.TrialPruned(f"Server Error: {response.text}")
        
        results = response.json()
        if "optimizer_ticket_id" in results:
            ticket_id = results["optimizer_ticket_id"]
            cache_path = os.path.join(OPTIMIZER_CACHE_DIR, f"{ticket_id}.json")
            for _ in range(10):
                if os.path.exists(cache_path):
                    with open(cache_path, 'r') as f: return json.load(f)
                time.sleep(1)
            raise optuna.TrialPruned(f"Cache file {ticket_id} never appeared.")
        return results
    except Exception as e:
        raise optuna.TrialPruned(f"Backtest failed: {e}")

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
    if 'balance_continuous' not in stitched_df.columns: return {"totalTrades": 0}
    
    initial = stitched_df['balance_continuous'].iloc[0]
    final = stitched_df['balance_continuous'].iloc[-1]
    
    if final <= 0 or initial <= 0:
        return {"StitchedCalmarRatio": -999, "StitchedMaxDrawdown": 100.0}
        
    peak = stitched_df['balance_continuous'].cummax()
    drawdown = (stitched_df['balance_continuous'] - peak) / (peak + EPSILON)
    max_dd = abs(drawdown.min() * 100)
    
    days = (pd.to_datetime(stitched_df['timestamp'].iloc[-1]) - pd.to_datetime(stitched_df['timestamp'].iloc[0])).days
    if days < 1: days = 1
    annual_return = ((final / initial) ** (365.25 / days)) - 1
    calmar = (annual_return * 100) / (max_dd + EPSILON)
    
    return {"StitchedCalmarRatio": calmar, "StitchedMaxDrawdown": max_dd}

def run_initial_probe(symbol, timeframe, first_fold, market_models):
    """Runs a quick backtest on the first fold to seed the optimizer."""
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
    
    for strat_code in TREND_STRATEGIES:
        base_config['strategies'] = [{"code": strat_code, "params": {}}]
        try:
            res = run_single_backtest(base_config)
            calmar = res.get('metrics', {}).get('calmarRatio', -999)
            if calmar > best_trend['calmar']:
                best_trend.update({"name": strat_code, "calmar": calmar})
            print(f"  -> {strat_code}: {calmar:.2f}")
        except optuna.TrialPruned: pass
        
    for strat_code in RANGE_STRATEGIES:
        base_config['strategies'] = [{"code": strat_code, "params": {}}]
        try:
            res = run_single_backtest(base_config)
            calmar = res.get('metrics', {}).get('calmarRatio', -999)
            if calmar > best_range['calmar']:
                best_range.update({"name": strat_code, "calmar": calmar})
            print(f"  -> {strat_code}: {calmar:.2f}")
        except optuna.TrialPruned: pass

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
        
        test_config['strategies'] = [
            {"code": trend_strat, "params": {}},
            {"code": range_strat, "params": {}}
        ]
        
        set_nested_value(test_config, "params.hybridMode", "REGIME")
        regime_thresh = trial.suggest_int("params.regime_threshold", 15, 40, step=5)
        set_nested_value(test_config, "params.regime_threshold", regime_thresh)

        adx_filter = trial.suggest_int("params.minAdxLevel", 0, 20, step=5) 
        set_nested_value(test_config, "params.minAdxLevel", adx_filter)
        
        test_config['mlMode'] = "predictions"
        test_config['mlModel'] = trial.suggest_categorical("mlModel", market_models)
        threshold = trial.suggest_float("mlThreshold", 0.55, 0.70, step=0.05) 
        test_config['mlThreshold'] = threshold

        active_strats = [trend_strat, range_strat]
        
        if "sma_crossover" in active_strats:
            p1 = trial.suggest_int("params.sma_fast_period", 5, 40, step=5)
            p2 = trial.suggest_int("params.sma_slow_period", p1+10, 175, step=10) 
            set_nested_value(test_config, "params.sma_fast_period", p1)
            set_nested_value(test_config, "params.sma_slow_period", p2)
        if "rsi_divergence" in active_strats:
            p1 = trial.suggest_int("params.rsi_length", 10, 25)
            set_nested_value(test_config, "params.rsi_length", p1)
        if "bollinger_bands" in active_strats:
            p1 = trial.suggest_int("params.bb_length", 15, 30)
            p2 = trial.suggest_float("params.bb_std", 1.5, 2.5, step=0.5)
            set_nested_value(test_config, "params.bb_length", p1)
            set_nested_value(test_config, "params.bb_std", p2)
        if "macd_crossover" in active_strats:
            p1 = trial.suggest_int("params.macd_fast_period", 8, 20)
            p2 = trial.suggest_int("params.macd_slow_period", p1+5, 40)
            set_nested_value(test_config, "params.macd_fast_period", p1)
            set_nested_value(test_config, "params.macd_slow_period", p2)
        if "stochastic_crossover" in active_strats:
            p1 = trial.suggest_int("params.k_period", 10, 30)
            set_nested_value(test_config, "params.k_period", p1)
        if "cci_oversold" in active_strats:
            p1 = trial.suggest_int("params.cci_length", 14, 40)
            set_nested_value(test_config, "params.cci_length", p1)
            
        tsl_mult = trial.suggest_float("params.tslAtrMult", 1.5, 6.0, step=0.5)
        set_nested_value(test_config, "params.tslAtrMult", tsl_mult)

        all_curves = []
        total_trades = 0
        for fold in wfo_folds:
            fold_config = deepcopy(test_config)
            fold_config['startDate'] = fold['startDate']
            fold_config['endDate'] = fold['endDate']
            try:
                res = run_single_backtest(fold_config)
                trades = res.get('metrics', {}).get('totalTrades', 0)
                if trades < 2: raise optuna.TrialPruned("Fold failed (trades < 2)") 
                all_curves.append(res.get('equityCurve'))
                total_trades += trades
            except Exception as e: raise e

        metrics = calculate_stitched_metrics(all_curves)
        calmar = metrics.get('StitchedCalmarRatio', -999)
        dd = metrics.get('StitchedMaxDrawdown', 100)
        
        trial.set_user_attr("combo_strategies", f"{trend_strat},{range_strat}")
        
        print(f"Trial {trial.number}: {trend_strat}/{range_strat} [ADX>{regime_thresh}] | Calmar: {calmar:.2f}")
        return calmar, dd

    return objective

def run_certification_check(market_id, best_params, report_csv_file, base_config):
    symbol, timeframe = market_id.split('_')
    clean_params = {}
    for key, value in best_params.items():
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)): continue
        if key in ["CalmarRatio", "MaxDrawdown"]: continue
        if key.startswith("params."):
            param_key = key.split('.', 1)[1]
            if 'params' not in clean_params: clean_params['params'] = {}
            set_nested_value(clean_params['params'], param_key, value)
        else:
            set_nested_value(clean_params, key, value)

    payload = {"symbol": symbol.upper(), "timeframe": timeframe, "best_params": clean_params, "base_config": base_config}
    try:
        url = f"{ML_SERVER_URL}/api/ml/certify-strategy"
        response = requests.post(url, json=payload, timeout=900)
        if response.status_code != 200:
            print(f"   Certification Request Failed: {response.text}")
            return None
        return response.json()
    except Exception as e:
        print(f"   Certification Error: {e}")
        return None

def run_reproducibility_check(market_id, best_params, wfo_folds):
    symbol, timeframe = market_id.split('_')
    base_config = deepcopy(BASE_CONFIG)
    base_config['symbol'] = symbol.upper()
    base_config['timeframe'] = timeframe
    base_config['optimizer_mode'] = True
    
    for key, value in best_params.items():
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)): continue
        if key in ["CalmarRatio", "MaxDrawdown", "combo_strategies"]: continue
        if key.startswith("params."):
            param_key = key.split('.', 1)[1] 
            set_nested_value(base_config['params'], param_key, value)
        else:
            set_nested_value(base_config, key, value)

    combo_str = best_params.get("combo_strategies", "")
    if combo_str:
        combo_codes = combo_str.split(',')
        base_config['strategies'] = [{"code": code, "params": {}} for code in combo_codes]

    results_list=[]
    for i in range(REPRODUCIBILITY_RUNS):
        print(f"  Repro run {i+1}/{REPRODUCIBILITY_RUNS}...")
        all_fold_equity_curves=[]
        try:
            for fold in wfo_folds:
                fold_config = deepcopy(base_config)
                fold_config['startDate'] = fold['startDate']
                fold_config['endDate'] = fold['endDate']
                fold_results = run_single_backtest(fold_config)
                all_fold_equity_curves.append(fold_results.get('equityCurve'))
            stitched_metrics = calculate_stitched_metrics(all_fold_equity_curves)
            results_list.append(stitched_metrics)
        except Exception as e:
            print(f"  Run {i+1} failed: {e}")
            return False
            
    if not results_list: return False
    
    first_calmar = results_list[0].get('StitchedCalmarRatio', 0.0)
    for i,metrics in enumerate(results_list[1:],1):
        calmar_i = metrics.get('StitchedCalmarRatio', 0.0)
        if abs(calmar_i - first_calmar) > METRIC_TOLERANCE:
            print(f"  Mismatch: {first_calmar} vs {calmar_i}")
            return False
            
    return True

def main():
    if not fetch_all_models(): return
    markets = get_user_market_selection()
    
    for symbol, timeframe in markets:
        market_id = f"{symbol}_{timeframe}"
        
        # --- INTERACTIVE LOOK & FEEL RESTORED ---
        wfo_folds = generate_wfo_folds()
        
        m_check = f"{symbol.split('-')[0]}_{timeframe}"
        ml_models = [m['id'] for m in ALL_AVAILABLE_MODELS if m['id'].lower().startswith(m_check.lower())]
        if not ml_models:
            print(f"No ML models found for {m_check}. Skipping.")
            continue

        print(f"\n[Optimizer] Starting PROBE for market: {market_id}...")
        T_best, R_best = run_initial_probe(symbol, timeframe, wfo_folds[0], ml_models)

        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_id = f"{market_id}_REGIME_SCAN_{run_timestamp}"
        study_db_file = os.path.join(RESULTS_DIR, f"{study_id}.db")
        
        study = optuna.create_study(
            study_name=study_id,
            storage=f"sqlite:///{study_db_file}",
            load_if_exists=False,
            directions=["maximize", "minimize"]
        )
        
        print("\n" + "="*60)
        print(f" 🚀 Starting REGIME Optimization Study")
        print(f" 🎯 Strategy:       Trend/Range Regime Switching")
        print(f" 📈 Market:         {symbol.upper()} @ {timeframe}")
        print(f" 📊 WFO Folds:      {len(wfo_folds)} (Start: {wfo_folds[0]['startDate']})")
        print(f" ⏱️ Total Trials:     {TOTAL_TRIALS}")
        print(f" ⚡ Parallel Jobs:  {PARALLEL_JOBS}")
        print(f" 💾 Study DB:       {study_db_file}")
        print("="*60)
        
        # Intelligent Seeding
        study.enqueue_trial({
            "trend_strategy": T_best,
            "range_strategy": R_best,
            "params.regime_threshold": 25,
            "mlModel": ml_models[0],
            "mlThreshold": 0.65
        })

        try:
            study.optimize(create_objective(symbol, timeframe, wfo_folds, ml_models), n_trials=TOTAL_TRIALS, n_jobs=PARALLEL_JOBS)
        except Exception as e:
            print(f"Study failed: {e}")
            continue
        
        # Harvest & Certify
        print("\n[Harvest] Checking for certified winners...")
        best_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values[0] > MIN_CALMAR_TO_CERTIFY]
        
        # Filter duplicates
        unique_trials = []
        seen = set()
        for t in best_trials:
            s = json.dumps(t.params, sort_keys=True)
            if s not in seen:
                seen.add(s)
                unique_trials.append(t)
        unique_trials.sort(key=lambda x: x.values[0], reverse=True)

        print(f"[Harvest] Found {len(unique_trials)} unique candidates.")

        for trial in unique_trials:
            params = deepcopy(trial.params)
            trend = params.pop("trend_strategy", "")
            rng = params.pop("range_strategy", "")
            combo_str = f"{trend},{rng}"
            
            params['combo_strategies'] = combo_str
            params['hybridMode'] = "REGIME"
            
            print(f"\n--- Checking Candidate: {combo_str} (Hist Calmar: {trial.values[0]:.2f}) ---")
            
            cert_report = run_certification_check(market_id, params, "dummy.csv", BASE_CONFIG)
            cert_pass = False
            
            if cert_report and 'holdout_metrics' in cert_report:
                oos_calmar = cert_report['holdout_metrics'].get('calmarRatio', -1)
                print(f"   OOS Calmar: {oos_calmar:.2f}")
                if oos_calmar > 0: cert_pass = True
            elif cert_report:
                print(f"   Certification Failed: {cert_report.get('reason', 'Unknown')}")
            else:
                 print("   Certification Failed: No response from server")
            
            repro_pass = False
            if cert_pass:
                repro_pass = run_reproducibility_check(market_id, params, wfo_folds)
                print(f"   Reproducibility: {'PASSED' if repro_pass else 'FAILED'}")
            
            if cert_pass and repro_pass:
                print(f"🎉 WINNER FOUND! Strategy: {combo_str}")
                winner_file = os.path.join(RESULTS_DIR, f"winner_{market_id}_{run_timestamp}.json")
                with open(winner_file, "a") as f:
                    json.dump(params, f)
                    f.write("\n")

if __name__=="__main__":
    main()
