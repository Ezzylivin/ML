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
REPRODUCIBILITY_RUNS = 2
EPSILON = 1e-9
MIN_TRADES_PER_FOLD = 5
MIN_TOTAL_TRADES = 20

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.INFO)

# --- CONFIGURATION ---
ML_SERVER_URL = "http://74.208.28.77:8000"
RESULTS_DIR = "/root/Project/ML/data/optimizer_results"
OPTIMIZER_CACHE_DIR = "/root/Project/ML/cache/optimizer/"
TOTAL_TRIALS = 500  # Increased trials for the larger search space
PARALLEL_JOBS = 6
RISK_FREE_RATE = 0.02

# 💡 This is the list of *codes* your server's generate_ta_signals expects
ALL_TA_STRATEGIES = [
    "atr_signal", "bollinger_bands", "cci_oversold", "ichimoku_cloud",
    "macd_crossover", "sma_crossover", "obv_signal", "psar_signal",
    "rsi_divergence", "stochastic_crossover"
]

BASE_CONFIG = {
    "symbol": "BTC-USD",
    "timeframe": "1h",
    "startDate": "2021-01-01",
    "endDate": "2021-12-31",
    "initialBalance": 1000,
    "fee": 0.001,
    "riskManagementMode": "standard",
    "riskPercentage": 1,
    "params": {}, # This will be filled by the optimizer
    "optimizer_mode": True
}

ALL_AVAILABLE_MODELS = [] # We'll keep this disabled for a TA-param search

# --- 💡 NEW: JSON Sanitization Helper ---
def sanitize_for_json(obj):
    """
    Recursively clean a dict/list to make it JSON compliant.
    Removes np.nan, np.inf, -np.inf.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(elem) for elem in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None  # Convert NaN/Inf to null
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif obj is pd.NaT:
        return None
    return obj

# --- Helper Functions (From optimizer_v7.py) ---
def set_nested_value(d, keys, value):
    """Sets a value in a nested dictionary using a dot-separated key."""
    keys = keys.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def fetch_all_models():
    """Called once at the start to get the *entire* model list."""
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
        print(f"[Optimizer] CRITICAL ERROR: Could not connect to ML server. {e}")
        return False

def get_user_market_selection():
    """Parses models, finds unique markets, and asks user for selection."""
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
            print("Invalid input. Please enter numbers separated by commas.")
            continue

def generate_wfo_folds():
    """Generates a list of date ranges for Walk-Forward Optimization."""
    folds = [
        {"startDate": "2021-01-01", "endDate": "2021-12-31"},
        {"startDate": "2022-01-01", "endDate": "2022-12-31"},
        {"startDate": "2023-01-01", "endDate": "2023-12-31"},
        {"startDate": "2024-01-01", "endDate": "2024-12-31"},
        {"startDate": "2025-01-01", "endDate": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
    ]
    print(f"[WFO] Generated {len(folds)} testing folds (2021-Present).")
    return folds

def run_single_backtest(config):
    """
    Calls the correct backtest endpoint after sanitizing the config for JSON.
    """
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest" if "strategies" in config else f"{ML_SERVER_URL}/api/ml/run-backtest-on"
    
    safe_config = sanitize_for_json(config)
    
    response = requests.post(url, json=safe_config, timeout=600)
    
    if response.status_code != 200:
        print(f"    🔥 FAILED: {response.status_code} - {response.text[:150]}")
        raise optuna.TrialPruned(f"Server Error: {response.status_code} {response.text[:150]}")
        
    results = response.json()
    
    if "optimizer_ticket_id" in results:
        ticket_id = results["optimizer_ticket_id"]
        cache_path = os.path.join(OPTIMIZER_CACHE_DIR, f"{ticket_id}.json")
        for _ in range(10): # Poll for 10 seconds
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r') as f:
                        file_results = json.load(f)
                    os.remove(cache_path) # Clean up
                    return file_results
                except Exception as e:
                    raise optuna.TrialPruned(f"Cache file read error: {e}")
            time.sleep(1)
        raise optuna.TrialPruned(f"Cache file {ticket_id} never appeared.")
        
    return results.get('combinedResult', results) if "strategies" in config else results

def calculate_stitched_metrics(equity_curves: list) -> dict:
    """
    Stitches equity curves and calculates robust, continuous metrics.
    """
    if not equity_curves:
        return {}
        
    all_dfs = []
    last_balance = 0
    
    for i, fold_curve in enumerate(equity_curves):
        if not fold_curve or len(fold_curve) == 0:
            continue
            
        df = pd.DataFrame(fold_curve)
        if df.empty:
            continue
            
        if i == 0:
            df['balance_continuous'] = df['balance']
            last_balance = df['balance'].iloc[-1]
        else:
            initial_fold_balance = fold_curve[0]['balance']
            df['balance_continuous'] = df['balance'] - initial_fold_balance + last_balance
            last_balance = df['balance_continuous'].iloc[-1]
            
        all_dfs.append(df)

    if not all_dfs:
        return {"totalTrades": 0, "StitchedCalmarRatio": -999, "StitchedMaxDrawdown": 100.0}

    stitched_df = pd.concat(all_dfs, ignore_index=True)
    stitched_df['timestamp'] = pd.to_datetime(stitched_df['timestamp'])
    stitched_df = stitched_df.set_index('timestamp').sort_index()

    equity_series = stitched_df['balance_continuous']
    
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
    
    daily_returns = equity_series.resample('D').last().pct_change().fillna(0)
    
    if len(daily_returns) > 1:
        days = (equity_series.index[-1] - equity_series.index[0]).days
        days = max(days, 1) # Ensure at least 1 day
            
        annual_return_rate = ((final_balance / (initial_balance + EPSILON)) ** (365.25 / days)) - 1
        annual_std_dev = daily_returns.std() * np.sqrt(365.25)
        sharpe_ratio = (annual_return_rate - RISK_FREE_RATE) / (annual_std_dev + EPSILON)
        
        downside_returns = daily_returns[daily_returns < 0]
        annual_downside_std = downside_returns.std() * np.sqrt(365.25)
        
        calmar_ratio = (annual_return_rate * 100) / (max_drawdown_pct + EPSILON)
    else:
        sharpe_ratio = calmar_ratio = 0.0

    return {
        "StitchedCalmarRatio": calmar_ratio,
        "StitchedSharpeRatio": sharpe_ratio,
        "StitchedMaxDrawdown": max_drawdown_pct,
        "StitchedTotalReturn": total_return_pct,
    }

# --- 🚀 NEW: PARAMETER-FOCUSED OBJECTIVE FUNCTION ---
def create_objective(symbol, timeframe, wfo_folds):
    
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        
        test_config = deepcopy(BASE_CONFIG)
        test_config['symbol'] = symbol
        test_config['timeframe'] = timeframe
        
        # This will hold the dynamic TA params
        params = {} 

        # --- 1. Optimizer "Mode" ---
        # We are only testing 'single' strategies and ML is 'off'
        # This is a pure TA parameter search.
        test_config['test_type'] = "single" # Hardcoded to 'single'
        test_config['mlMode'] = "off"      # Hardcoded to 'off'
        
        # --- 2. Select a Strategy ---
        strategy_code = trial.suggest_categorical("code", ALL_TA_STRATEGIES)
        test_config['code'] = strategy_code

        # --- 3. Define Search Space for EACH Strategy ---
        if strategy_code == "atr_signal":
            params['atr_length'] = trial.suggest_int("params.atr_length", 10, 30)
        
        elif strategy_code == "bollinger_bands":
            params['bb_length'] = trial.suggest_int("params.bb_length", 15, 30)
            params['bb_std'] = trial.suggest_float("params.bb_std", 2.0, 3.5, step=0.5)

        elif strategy_code == "cci_oversold":
            params['cci_length'] = trial.suggest_int("params.cci_length", 14, 40)
            params['cci_overbought'] = trial.suggest_int("params.cci_overbought", 80, 150)
            params['cci_oversold'] = trial.suggest_int("params.cci_oversold", -150, -80)

        elif strategy_code == "ichimoku_cloud":
            params['ichi_tenkan'] = trial.suggest_int("params.ichi_tenkan", 5, 15)
            params['ichi_kijun'] = trial.suggest_int("params.ichi_kijun", 20, 40)
            params['ichi_senkou'] = trial.suggest_int("params.ichi_senkou", 40, 60)

        elif strategy_code == "macd_crossover":
            params['macd_fast'] = trial.suggest_int("params.macd_fast", 8, 16)
            params['macd_slow'] = trial.suggest_int("params.macd_slow", 20, 35)
            params['macd_signal'] = trial.suggest_int("params.macd_signal", 7, 12)
            
        elif strategy_code == "sma_crossover":
            # Use 'log' to get more values at the low end
            short = trial.suggest_int("params.sma_short", 5, 50, log=True)
            params['sma_short'] = short
            # Ensure long is always greater than short
            params['sma_long'] = trial.suggest_int("params.sma_long", short + 10, 200, log=True)
            
        elif strategy_code == "obv_signal":
            # OBV has no params, so we test filters more
            pass # No params to add

        elif strategy_code == "psar_signal":
            params['psar_step'] = trial.suggest_float("params.psar_step", 0.01, 0.05, step=0.01)
            params['psar_max'] = trial.suggest_float("params.psar_max", 0.1, 0.3, step=0.05)

        elif strategy_code == "rsi_divergence":
            params['rsi_length'] = trial.suggest_int("params.rsi_length", 10, 30)
            params['rsi_overbought'] = trial.suggest_int("params.rsi_overbought", 65, 85)
            params['rsi_oversold'] = trial.suggest_int("params.rsi_oversold", 15, 35)

        elif strategy_code == "stochastic_crossover":
            params['stoch_k'] = trial.suggest_int("params.stoch_k", 10, 30)
            params['stoch_d'] = trial.suggest_int("params.stoch_d", 3, 7)
            params['stoch_smooth_k'] = trial.suggest_int("params.stoch_smooth_k", 3, 7)

        # --- 4. Define Search Space for Filters (TSL-Only) ---
        params['trendFilterPeriod'] = trial.suggest_categorical("params.trendFilterPeriod", [0, 50, 100, 200])
        params['minAdxLevel'] = trial.suggest_categorical("params.minAdxLevel", [0, 20, 25])
        params['minAtrPct'] = trial.suggest_categorical("params.minAtrPct", [0.0, 0.1, 0.2, 0.5])
        params['tslAtrMult'] = trial.suggest_float("params.tslAtrMult", 1.5, 7.0, step=0.5)

        # Set the 'params' dict in the main config
        set_nested_value(test_config, "params", params)

        # --- 5. Run the WFO ---
        all_fold_equity_curves = []
        total_trades = 0
        
        for fold in wfo_folds:
            fold_config = deepcopy(test_config)
            fold_config['startDate'] = fold['startDate']
            fold_config['endDate'] = fold['endDate']
            
            try:
                results = run_single_backtest(fold_config)
                fold_trades = results.get('metrics', {}).get('totalTrades', 0)
                
                if fold_trades < MIN_TRADES_PER_FOLD:
                    raise optuna.TrialPruned(f"Fold failed (Trades: {fold_trades} < {MIN_TRADES_PER_FOLD})")

                all_fold_equity_curves.append(results.get('equityCurve'))
                total_trades += fold_trades

            except Exception as e:
                raise optuna.TrialPruned(f"Fold failed: {e}")

        # --- 6. Calculate Score ---
        if not all_fold_equity_curves or total_trades < MIN_TOTAL_TRADES:
            raise optuna.TrialPruned(f"Total trades {total_trades} < {MIN_TOTAL_TRADES}")
            
        stitched_metrics = calculate_stitched_metrics(all_fold_equity_curves)
        calmar_ratio = stitched_metrics.get('StitchedCalmarRatio', 0.0)
        max_drawdown = stitched_metrics.get('StitchedMaxDrawdown', 999.0)

        print(f"  ✅ Trial #{trial.number} ({strategy_code}) | "
              f"Calmar: {calmar_ratio:.2f} | DD: {max_drawdown:.2f}% | Trades: {total_trades}")
        
        return calmar_ratio, max_drawdown

    return objective

# --- 💡 FIX: Added Reproducibility Functions (Needed by main) ---
def run_certification_check(market_id, best_params, report_csv_file, base_config):
    print(f"\n--- [Certifying Best Strategy] ---")
    symbol, timeframe = market_id.split('_')
    clean_params = {k:v for k,v in best_params.items() if not (isinstance(v,float) and np.isnan(v))}
    payload = {
        "symbol": symbol, "timeframe": timeframe, 
        "best_params": clean_params, "base_config": base_config
    }
    safe_payload = sanitize_for_json(payload)
    try:
        url = f"{ML_SERVER_URL}/api/ml/certify-strategy"
        response = requests.post(url, json=safe_payload, timeout=900)
        if response.status_code != 200:
            print(f"    🔥 Certification FAILED: {response.text}")
            return None
        report = response.json()
        report_path = report_csv_file.replace('.csv','_certification.json')
        with open(report_path,'w') as f:
            json.dump(report, f, indent=4)
        print(f"    ✅ Certification report saved: {report_path}")
        return report
    except Exception as e:
        print(f"    🔥 Certification error: {e}")
        return None

def run_reproducibility_check(market_id, best_params):
    print(f"\n--- [Running Reproducibility Check] ---")
    symbol, timeframe = market_id.split('_')
    base_config = deepcopy(BASE_CONFIG)
    base_config['symbol'] = symbol
    base_config['timeframe'] = timeframe
    base_config['optimizer_mode'] = True
    
    for key,value in best_params.items():
        if key not in ["CalmarRatio", "MaxDrawdown", "combo_strategies"]:
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                set_nested_value(base_config, key, value)
    
    if "combo_strategies" in best_params and isinstance(best_params["combo_strategies"], str):
        combo_codes = [c.strip() for c in best_params["combo_strategies"].split(',')]
        base_config['strategies'] = [{"code": code, "params": {}} for code in combo_codes]

    results_list = []
    wfo_folds = generate_wfo_folds()
    
    for i in range(REPRODUCIBILITY_RUNS):
        print(f"    Repro run {i+1}/{REPRODUCIBILITY_RUNS}...")
        all_fold_equity_curves = []
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
            print(f"    🔥 Run {i+1} failed: {e}")
            return False
            
    if not results_list:
        print("    🔥 Reproducibility FAILED: No results to compare.")
        return False

    first_calmar = results_list[0]['StitchedCalmarRatio']
    first_dd = results_list[0]['StitchedMaxDrawdown']
    
    for i,metrics in enumerate(results_list[1:], 1):
        if abs(metrics['StitchedCalmarRatio'] - first_calmar) > METRIC_TOLERANCE or \
           abs(metrics['StitchedMaxDrawdown'] - first_dd) > METRIC_TOLERANCE:
            print("    🔥 Reproducibility FAILED!")
            print(f"      Run 0: Calmar={first_calmar}, DD={first_dd}")
            print(f"      Run {i}: Calmar={metrics['StitchedCalmarRatio']}, DD={metrics['StitchedMaxDrawdown']}")
            return False
            
    print("    ✅ Reproducibility PASSED")
    return True
# --- 💡 END FIX ---


# --- 💡 FIX: Added MAIN function ---
def main():
    parser = argparse.ArgumentParser(description="Optimizer v7 - TA Parameter Search")
    parser.add_argument('--certify', action='store_true', help="Run certification on hold-out data for the best strategy.")
    parser.add_argument('--repro-check', action='store_true', help="Run reproducibility check on the best strategy.")
    parser.add_argument('--trials', type=int, default=TOTAL_TRIALS, help="Number of trials to run.")
    parser.add_argument('--jobs', type=int, default=PARALLEL_JOBS, help="Number of parallel jobs.")
    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OPTIMIZER_CACHE_DIR, exist_ok=True)
    
    if not fetch_all_models(): # Still need this to get market list
        sys.exit(1)
        
    markets_to_test = get_user_market_selection()

    print("\n" + "="*50)
    print(f"🚀 Starting {len(markets_to_test)} TA Parameter Optimization Studies...")
    print(f"   (Mode: ML-OFF, TSL-Only)")
    print(f"   Trials per Study: {args.trials}")
    print(f"   Parallel Jobs: {args.jobs}")
    print("="*50)

    for symbol, timeframe in markets_to_test:
        market_start_time = time.time()
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        market_id = f"{symbol}_{timeframe}"
        
        print(f"\n--- Starting Study for: {market_id} ---")
        
        study_db_file = os.path.join(RESULTS_DIR, f"study_TA_PARAMS_{market_id}_{run_timestamp}.db")
        report_csv_file = os.path.join(RESULTS_DIR, f"report_TA_PARAMS_{market_id}_{run_timestamp}.csv")
        
        wfo_folds = generate_wfo_folds()
        objective_func = create_objective(symbol, timeframe, wfo_folds)
        
        study = optuna.create_study(
            study_name=f"study_TA_PARAMS_{market_id}_{run_timestamp}",
            storage=f"sqlite:///{study_db_file}",
            load_if_exists=False,
            directions=["maximize", "minimize"] # Maximize Calmar, Minimize Drawdown
        )
        
        try:
            study.optimize(objective_func, n_trials=args.trials, n_jobs=args.jobs)
        except Exception as e:
            print(f"    🔥 Study for {market_id} failed: {e}")
            continue
            
        market_end_time = time.time()
        print(f"\n--- 🏁 Study for {market_id} complete in {(market_end_time - market_start_time)/60:.2f} minutes ---")

        best_trials = study.best_trials
        if not best_trials:
            print(f"No complete trials found for {market_id}.")
            continue

        report_data = []
        for trial in best_trials:
            calmar, dd = trial.values
            params = trial.params
            params['CalmarRatio'] = calmar
            params['MaxDrawdown'] = dd
            if "combo_strategies" in trial.user_attrs:
                params['combo_strategies'] = trial.user_attrs['combo_strategies']
            report_data.append(params)
            
        if not report_data:
            print(f"No valid trials found in 'best_trials' list for {market_id}.")
            continue

        df = pd.DataFrame(report_data).sort_values("CalmarRatio", ascending=False)
        df.to_csv(report_csv_file, index=False, float_format='%.3f')
        print(f"    ✅ Report saved: {report_csv_file}\n\n--- 🏆 Best 10 Results ---")
        print(df.head(10).to_string(index=False))

        # --- Certification & Repro ---
        best_trial_params = df.iloc[0].to_dict()
        certification_pass = False
        repro_pass = False
        cert_report = None
        
        if args.certify:
            cert_report = run_certification_check(market_id, best_trial_params, report_csv_file, BASE_CONFIG)
            if cert_report:
                certification_pass = cert_report.get('certification_passed', False)
                
        if args.repro_check:
            repro_pass = run_reproducibility_check(market_id, best_trial_params)
            
        if cert_report:
            final_metrics = cert_report.get('holdout_metrics', {})
        else:
            final_metrics = {
                'calmar_ratio': best_trial_params.get('CalmarRatio', 0),
                'max_drawdown': best_trial_params.get('MaxDrawdown', 100), 'win_rate': 0
            }
            
        grade_results = grade_strategy_quality(final_metrics, repro_pass, certification_pass)
        
        print("\n--- 🎓 FINAL STRATEGY GRADE ---")
        print(f"  Strategy: {best_trial_params.get('code', 'N/A')}")
        print(f"  Grade: {grade_results['grade']} (Reliability: {grade_results['reliability']:.1f}%)")
        print(f"  Certification: {'PASSED' if certification_pass else 'FAILED' if args.certify else 'SKIPPED'}")
        print(f"  Reproducibility: {'PASSED' if repro_pass else 'FAILED' if args.repro_check else 'SKIPPED'}")
        print("-" * 30 + "\n")
# --- 💡 END FIX ---


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    pd.options.mode.chained_assignment = None
    
    main() # 💡 Run the main function
