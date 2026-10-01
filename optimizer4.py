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

# --- 🚀 Deterministic Seeding & Constants ---
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
TOTAL_TRIALS = 60 # Your value
PARALLEL_JOBS = 6
RISK_FREE_RATE = 0.02

BASE_CONFIG = {
    "symbol": "BTC-USD",
    "timeframe": "1h",
    "startDate": "2021-01-01",
    "endDate": "2021-12-31",
    "initialBalance": 300, # Your value
    "fee": 0.001,
    "riskManagementMode": "standard",
    "riskPercentage": 1,
    "growthCapitalTarget": None,
    "params": {},
    "optimizer_mode": True
}

ALL_TA_STRATEGIES = [
    "sma_crossover", "rsi_divergence", "macd_crossover", "stochastic_crossover",
    "cci_oversold", "bollinger_bands", "ichimoku_cloud", "atr_signal",
    "obv_signal", "psar_signal"
]

ALL_AVAILABLE_MODELS = []

# --- Helper Functions ---
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
        print("[MarketScan] No models found. Defaulting to BTC-USD / 1h.")
        return [("BTC-USD", "1h")]
    market_pairs = set()
    for model in ALL_AVAILABLE_MODELS:
        try:
            parts = model['id'].split('_')
            if len(parts) >= 2:
                symbol = f"{parts[0].upper()}-USD"
                timeframe = parts[1]
                market_pairs.add((symbol, timeframe))
        except Exception:
            pass
    sorted_pairs = sorted(list(market_pairs))
    if not sorted_pairs:
        return [("BTC-USD", "1h")]
    print("\n--- Available Markets ---")
    for i, (symbol, timeframe) in enumerate(sorted_pairs):
        print(f"  [{i+1}] {symbol} @ {timeframe}")
    print("  [0] TEST ALL")
    while True:
        choice_str = input("Which market(s) to test? (e.g., 1,3,4): ")
        if not choice_str:
            continue
        if choice_str.strip() == "0":
            return sorted_pairs
        try:
            choices = [int(c.strip()) for c in choice_str.split(',')]
            selected_pairs = []
            for c in choices:
                if 1 <= c <= len(sorted_pairs):
                    selected_pairs.append(sorted_pairs[c-1])
            if selected_pairs:
                return selected_pairs
        except ValueError:
            continue

def generate_wfo_folds():
    now = datetime.now(timezone.utc)
    current_year = now.year
    start_year = current_year - 4
    folds = []
    
    for year in range(start_year, current_year):
        folds.append({
            "startDate": f"{year}-01-01",
            "endDate": f"{year}-12-31"
        })
    folds.append({
        "startDate": f"{current_year}-01-01",
        "endDate": now.strftime("%Y-%m-%d")
    })
    
    print(f"[WFO] Generated {len(folds)} folds (for 5-year period).")
    return folds

def run_single_backtest(config):
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest" if "strategies" in config else f"{ML_SERVER_URL}/api/ml/run-backtest-on"
    try:
        response = requests.post(url, json=config, timeout=600)
        if response.status_code != 200:
            raise optuna.TrialPruned(response.text)
        results = response.json()
        if "optimizer_ticket_id" in results:
            ticket_id = results["optimizer_ticket_id"]
            cache_path = os.path.join(OPTIMIZER_CACHE_DIR, f"{ticket_id}.json")
            for _ in range(10):
                if os.path.exists(cache_path):
                    with open(cache_path, 'r') as f:
                        file_results = json.load(f)
                    os.remove(cache_path)
                    return file_results
                time.sleep(1)
            raise optuna.TrialPruned(f"Cache file {ticket_id} never appeared.")
        return results.get('combinedResult', results) if "strategies" in config else results
    except Exception as e:
        raise optuna.TrialPruned(f"Backtest failed: {e}")

def calculate_stitched_metrics(equity_curves):
    if not equity_curves:
        return {}
    all_dfs = []
    last_balance = 0
    for i, fold_curve in enumerate(equity_curves):
        if not fold_curve:
            continue
        if not isinstance(fold_curve, list) or (fold_curve and not isinstance(fold_curve[0], dict)):
            print(f"[Metrics] Warning: Fold {i} has unexpected equity data type. Skipping.")
            continue
            
        df = pd.DataFrame(fold_curve)
        if 'balance' not in df.columns:
            print(f"[Metrics] Warning: Fold {i} 'balance' column not in equity curve. Skipping.")
            continue
            
        if df.empty:
            continue
            
        if i == 0:
            df['balance_continuous'] = df['balance']
            if not df['balance_continuous'].empty:
                last_balance = df['balance_continuous'].iloc[-1]
        else:
            if not fold_curve: # Skip if fold_curve is empty
                continue
            initial_fold_balance = fold_curve[0]['balance']
            df['balance_continuous'] = df['balance'] - initial_fold_balance + last_balance
            if not df['balance_continuous'].empty:
                last_balance = df['balance_continuous'].iloc[-1]
        all_dfs.append(df)
        
    if not all_dfs:
        return {"totalTrades": 0}
        
    stitched_df = pd.concat(all_dfs, ignore_index=True)
    if 'timestamp' not in stitched_df.columns:
         print("[Metrics] Error: 'timestamp' not in stitched DF. Cannot calculate metrics.")
         return {"totalTrades": 0}
         
    stitched_df['timestamp'] = pd.to_datetime(stitched_df['timestamp'])
    stitched_df = stitched_df.set_index('timestamp')
    
    if 'balance_continuous' not in stitched_df.columns:
        print("[Metrics] Error: 'balance_continuous' not in stitched DF. Cannot calculate metrics.")
        return {"totalTrades": 0}

    equity_series = stitched_df['balance_continuous']
    if equity_series.empty:
        return {"totalTrades": 0}
        
    initial_balance = equity_series.iloc[0]
    final_balance = equity_series.iloc[-1]
    
    if final_balance <= 0 or initial_balance <= 0:
        return {
            "StitchedCalmarRatio": -999,
            "StitchedSharpeRatio": -999,
            "StitchedMaxDrawdown": 100.0,
            "StitchedTotalReturn": -100.0,
        }
    total_return_pct = (final_balance / initial_balance - 1) * 100
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / (peak + EPSILON)
    max_drawdown_pct = abs(drawdown.min() * 100) if not drawdown.empty else 0
    daily_returns = equity_series.pct_change().fillna(0)
    if len(daily_returns) > 1:
        days = (equity_series.index[-1] - equity_series.index[0]).days
        days = max(days, 1)
        annual_return_rate = ((final_balance / (initial_balance + EPSILON)) ** (365.25 / days)) - 1
        annual_std_dev = daily_returns.std() * np.sqrt(365.25)
        sharpe_ratio = (annual_return_rate - RISK_FREE_RATE) / (annual_std_dev + EPSILON)
        downside_returns = daily_returns[daily_returns < 0]
        annual_downside_std = downside_returns.std() * np.sqrt(365.25) if not downside_returns.empty else 0
        calmar_ratio = (annual_return_rate * 100) / (max_drawdown_pct + EPSILON)
    else:
        sharpe_ratio = calmar_ratio = 0.0
    return {
        "StitchedCalmarRatio": calmar_ratio,
        "StitchedSharpeRatio": sharpe_ratio,
        "StitchedMaxDrawdown": max_drawdown_pct,
        "StitchedTotalReturn": total_return_pct,
    }

# --- Objective Function ---
def create_objective(symbol, timeframe, wfo_folds, strategy_code):
    def objective(trial: optuna.Trial):
        test_config = deepcopy(BASE_CONFIG)
        test_config['symbol'] = symbol
        test_config['timeframe'] = timeframe
        test_config['code'] = strategy_code

        # --- 1. General Filters ---
        trend_filter = trial.suggest_int("params.trendFilterPeriod", 0, 250, step=50)
        adx_filter = trial.suggest_int("params.minAdxLevel", 0, 35, step=5)
        min_atr_filter = trial.suggest_float("params.minAtrPct", 0.0, 1.0, step=0.1)
        
        set_nested_value(test_config, "params.trendFilterPeriod", trend_filter)
        set_nested_value(test_config, "params.minAdxLevel", adx_filter)
        set_nested_value(test_config, "params.minAtrPct", min_atr_filter)

        # --- 2. General Exit (Stop Loss) ---
        atr_tsl = trial.suggest_float("params.tslAtrMult", 1.5, 7.0, step=0.5)
        set_nested_value(test_config, "params.tslAtrMult", atr_tsl)
        
        # --- 3. ML Mode ---
        ml_mode = trial.suggest_categorical("mlMode", ["off"])
        test_config['mlMode'] = ml_mode

        # --- 4. Strategy-Specific Parameters ---
        if strategy_code == "sma_crossover":
            fast_p = trial.suggest_int("params.fast_period", 5, 50, step=5)
            slow_p = trial.suggest_int("params.slow_period", fast_p + 10, 200, step=10)
            set_nested_value(test_config, "params.fast_period", fast_p)
            set_nested_value(test_config, "params.slow_period", slow_p)

        elif strategy_code == "rsi_divergence":
            rsi_len = trial.suggest_int("params.rsi_length", 10, 30)
            oversold = trial.suggest_int("params.oversold_level", 20, 35)
            overbought = trial.suggest_int("params.overbought_level", oversold + 20, 80)
            set_nested_value(test_config, "params.rsi_length", rsi_len)
            set_nested_value(test_config, "params.oversold_level", oversold)
            set_nested_value(test_config, "params.overbought_level", overbought)

        elif strategy_code == "macd_crossover":
            fast_p = trial.suggest_int("params.fast_period", 8, 20)
            slow_p = trial.suggest_int("params.slow_period", fast_p + 5, 40)
            signal_p = trial.suggest_int("params.signal_period", 5, 12)
            set_nested_value(test_config, "params.fast_period", fast_p)
            set_nested_value(test_config, "params.slow_period", slow_p)
            set_nested_value(test_config, "params.signal_period", signal_p)

        elif strategy_code == "stochastic_crossover":
            k_p = trial.suggest_int("params.k_period", 10, 30)
            d_p = trial.suggest_int("params.d_period", 3, 9)
            set_nested_value(test_config, "params.k_period", k_p)
            set_nested_value(test_config, "params.d_period", d_p)

        elif strategy_code == "cci_oversold":
            cci_len = trial.suggest_int("params.cci_length", 14, 40)
            set_nested_value(test_config, "params.cci_length", cci_len)

        elif strategy_code == "bollinger_bands":
            bb_len = trial.suggest_int("params.bb_length", 15, 30)
            bb_std = trial.suggest_float("params.bb_std", 1.5, 3.0, step=0.5)
            set_nested_value(test_config, "params.bb_length", bb_len)
            set_nested_value(test_config, "params.bb_std", bb_std)

        elif strategy_code in ["ichimoku_cloud", "atr_signal", "obv_signal", "psar_signal"]:
            pass
        
        # --- 5. Run WFO with MANUAL Pruning ---
        all_fold_equity_curves = []
        total_trades = 0
        for i, fold in enumerate(wfo_folds):
            fold_config = deepcopy(test_config)
            fold_config['startDate'] = fold['startDate']
            fold_config['endDate'] = fold['endDate']
            try:
                results = run_single_backtest(fold_config)
                if results.get('metrics', {}).get('totalTrades', 0) < 5:
                    raise optuna.TrialPruned(f"Fold failed (trades < 5)")
                
                fold_equity = results.get('equityCurve')
                all_fold_equity_curves.append(fold_equity)
                total_trades += results.get('metrics', {}).get('totalTrades', 0)

                if i == 0: 
                    fold_metrics = calculate_stitched_metrics([fold_equity])
                    fold_calmar = fold_metrics.get('StitchedCalmarRatio', 0.0)
                    
                    if fold_calmar < -0.5: 
                        raise optuna.TrialPruned(f"Pruned: Bad Calmar {fold_calmar:.2f} on first fold")
                
            except Exception as e:
                raise optuna.TrialPruned(f"Fold failed: {e}")

        # --- 6. Final Metric Calculation ---
        stitched_metrics = calculate_stitched_metrics(all_fold_equity_curves)
        calmar_ratio = stitched_metrics.get('StitchedCalmarRatio', 0.0)
        max_drawdown = stitched_metrics.get('StitchedMaxDrawdown', 999.0)
        
        if stitched_metrics.get('StitchedCalmarRatio', 0.0) == -999:
             max_drawdown = 100.0 

        print(f"Trial #{trial.number} | {strategy_code} | Calmar: {calmar_ratio:.2f} | DD: {max_drawdown:.2f}% | Trades: {total_trades}")
        return calmar_ratio, max_drawdown

    return objective

# --- Certification & Repro (MODIFIED) ---

# <--- START OF MODIFIED SECTION ---
def run_certification_check(market_id, best_params, report_csv_file, base_config):
    """
    MODIFIED: This function now correctly "un-flattens" the 'best_params'
    dict before sending it to the certification server.
    """
    symbol, timeframe = market_id.split('_')
    
    # Create a new, clean payload config
    payload_config = deepcopy(base_config)
    payload_config['params'] = {} # Initialize params dict
    
    for key, value in best_params.items():
        # Filter out NaN/inf values
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            continue
        if key in ["CalmarRatio", "MaxDrawdown"]:
            continue
            
        if key.startswith("params."):
            # key is "params.fast_period"
            param_key = key.split('.', 1)[1] 
            # This correctly sets payload_config['params']['fast_period']
            set_nested_value(payload_config['params'], param_key, value)
        else:
            # key is "code", "mlMode", etc.
            # This correctly sets payload_config['code']
            set_nested_value(payload_config, key, value)

    # The payload 'best_params' is now the *correctly structured config*
    payload = {"symbol": symbol, "timeframe": timeframe, "best_params": payload_config, "base_config": base_config}
    
    try:
        url = f"{ML_SERVER_URL}/api/ml/certify-strategy"
        response = requests.post(url, json=payload, timeout=900)
        if response.status_code != 200:
            print(f"Certification FAILED: {response.text}")
            return None
        report = response.json()
        report_path = report_csv_file.replace('.csv','_certification.json')
        with open(report_path,'w') as f:
            json.dump(report,f,indent=4)
        print(f"Certification report saved: {report_path}")
        return report
    except Exception as e:
        print(f"Certification error: {e}")
        return None


def run_reproducibility_check(market_id, best_params, wfo_folds):
    """
    MODIFIED: This function now correctly parses the 'best_params' dict
    AND filters out NaN/inf values to perfectly match the certification check.
    """
    symbol, timeframe = market_id.split('_')
    base_config = deepcopy(BASE_CONFIG)
    base_config['symbol'] = symbol
    base_config['timeframe'] = timeframe
    base_config['optimizer_mode'] = True
    
    # --- THIS IS THE FIX ---
    for key, value in best_params.items():
        # --- ADDED THIS NAN/INF FILTER ---
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            continue
        # ---
            
        if key in ["CalmarRatio", "MaxDrawdown"]:
            continue
        
        if key.startswith("params."):
            param_key = key.split('.', 1)[1] 
            set_nested_value(base_config['params'], param_key, value)
        else:
            set_nested_value(base_config, key, value)
    # --- END FIX ---

    results_list=[]
    
    for i in range(REPRODUCIBILITY_RUNS):
        print(f"Repro run {i+1}/{REPRODUCIBILITY_RUNS}")
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
            print(f"Run {i+1} failed: {e}")
            return False
            
    if not results_list:
        print("Reproducibility FAILED: No results to compare.")
        return False
        
    first_calmar = results_list[0].get('StitchedCalmarRatio', 0.0)
    first_dd = results_list[0].get('StitchedMaxDrawdown', 999.0)
    
    for i,metrics in enumerate(results_list[1:],1):
        calmar_i = metrics.get('StitchedCalmarRatio', 0.0)
        dd_i = metrics.get('StitchedMaxDrawdown', 999.0)
        if abs(calmar_i - first_calmar) > METRIC_TOLERANCE or abs(dd_i - first_dd) > METRIC_TOLERANCE:
            print("Reproducibility FAILED")
            print(f"  Run 0: Calmar={first_calmar:.6f}, DD={first_dd:.6f}")
            print(f"  Run {i}: Calmar={calmar_i:.6f}, DD={dd_i:.6f}")
            return False
            
    print("Reproducibility PASSED")
    return True
# <--- END OF MODIFIED SECTION ---

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Optimizer v7.1 (Merged)")
    parser.add_argument('--certify',action='store_true')
    parser.add_argument('--repro-check',action='store_true')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not fetch_all_models():
        return
    markets_to_test = get_user_market_selection()
    
    print(f"Starting {len(markets_to_test)} market(s), {len(ALL_TA_STRATEGIES)} strategies each...")

    for symbol, timeframe in markets_to_test:
        market_id = f"{symbol}_{timeframe}"
        wfo_folds = generate_wfo_folds()
        
        print(f"\n[Optimizer] Starting full scan for market: {market_id}...")

        for strategy_code in ALL_TA_STRATEGIES:
            
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            study_id = f"{market_id}_{strategy_code}"
            study_db_file = os.path.join(RESULTS_DIR, f"study_{study_id}_{run_timestamp}.db")
            report_csv_file = os.path.join(RESULTS_DIR, f"report_{study_id}_{run_timestamp}.csv")

            objective_func = create_objective(symbol, timeframe, wfo_folds, strategy_code)
            
            study_name = f"study_{study_id}_{run_timestamp}"
            
            study = optuna.create_study(
                study_name=study_name,
                storage=f"sqlite:///{study_db_file}",
                load_if_exists=False,
                directions=["maximize", "minimize"]
            )
            
            print("\n" + "="*60)
            print(f" 🚀 Starting New Optimization Study")
            print(f" 🎯 Strategy:      {strategy_code}")
            print(f" 📈 Market:        {symbol} @ {timeframe}")
            print(f" 📊 WFO Folds:     {len(wfo_folds)} ({wfo_folds[0]['startDate']} to {wfo_folds[-1]['endDate']})")
            print(f" ⏱️ Total Trials:    {TOTAL_TRIALS}")
            print(f" ⚡ Parallel Jobs: {PARALLEL_JOBS}")
            print(f" 💾 Study DB:      {study_db_file}")
            print("="*60)
            
            try:
                study.optimize(objective_func, n_trials=TOTAL_TRIALS, n_jobs=PARALLEL_JOBS)
            except Exception as e:
                print(f"Study error for {strategy_code}: {e}")
                continue 
            
            best_trials = study.best_trials
            if not best_trials:
                print(f"[Optimizer] No best trials found for {strategy_code}.")
                continue 
            
            report_data = []
            for trial in best_trials:
                calmar, dd = trial.values
                params = trial.params
                params['CalmarRatio'] = calmar
                params['MaxDrawdown'] = dd
                report_data.append(params)
                
            df = pd.DataFrame(report_data).sort_values("CalmarRatio", ascending=False)
            df.to_csv(report_csv_file, index=False, float_format='%.3f')
            print(f"Report saved for {strategy_code}: {report_csv_file}\nTop result:\n{df.head(1)}")

            # --- Certification & Repro ---
            best_trial_params = df.iloc[0].to_dict()
            best_trial_params['code'] = strategy_code 
            
            certification_pass = False
            repro_pass = False
            cert_report = None
            
            if args.certify:
                cert_report = run_certification_check(market_id, best_trial_params, report_csv_file, BASE_CONFIG)
                if cert_report:
                    certification_pass = cert_report.get('certification_passed', False)
            
            if args.repro_check:
                repro_pass = run_reproducibility_check(market_id, best_trial_params, wfo_folds)
            
            final_metrics = cert_report.get('holdout_metrics', {}) if cert_report else {'calmar_ratio': best_trial_params.get('CalmarRatio', 0), 'max_drawdown': best_trial_params.get('MaxDrawdown', 100), 'win_rate': 0}
            grade_results = grade_strategy_quality(final_metrics, repro_pass, certification_pass)
            
            print(f"\n--- FINAL STRATEGY GRADE ({strategy_code}) ---")
            print(f"Grade: {grade_results['grade']}")
            print(f"Reliability: {grade_results['reliability']}%")
            print(f"Certification: {'PASSED' if certification_pass else 'FAILED'}")
            print(f"Reproducibility: {'PASSED' if repro_pass else 'FAILED'}")
            print("-------------------------------------------\n")

if __name__=="__main__":
    main()
