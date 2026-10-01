import requests
import json
import pandas as pd
import numpy as np
import warnings
from copy import deepcopy
import time
import optuna  # The "genius" library
from datetime import datetime, timezone
import os
import random
import argparse  # 🆕 NEW: For command-line flags
import sys      # 🆕 NEW: For exiting
import glob     # 🆕 NEW: For finding reports

# --- 🚀 GRADE IMPORTER ---
try:
    from data_quality_grader import grade_strategy_quality
except ImportError:
    print("[Optimizer] WARNING: data_quality_grader.py not found. Final grading will be skipped.")
    # Define a dummy function if the import fails so the script doesn't crash
    def grade_strategy_quality(metrics, reproducibility_pass, certification_pass):
        print("[Optimizer] Grading skipped.")
        return {"grade": "N/A", "reliability": 0}
# --- End Grade Importer ---


# --- 🚀 NEW: Deterministic Seeding & Constants ---
# We define these here to ensure the client-side logic is also deterministic.
SEED = 12345
METRIC_TOLERANCE = 1e-6
REPRODUCIBILITY_RUNS = 2

# Apply seeds to all relevant libraries
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.INFO)  # Use Optuna's logger
# --- End New Seeding ---

# --- CONFIGURATION ---
ML_SERVER_URL = "http://127.0.0.1:8000"  # ✅ Point to your Gunicorn HTTP server
RESULTS_DIR = "/root/Project/ML/data/optimizer_results"
OPTIMIZER_CACHE_DIR = "/root/Project/ML/cache/optimizer/"  # ✅ Must match server

# --- Study Configuration ---
TOTAL_TRIALS = 200
PARALLEL_JOBS = 6
RISK_FREE_RATE = 0.02
BASE_CONFIG = {
    "symbol": "BTC-USD",
    "timeframe": "1h",
    "startDate": "2021-01-01",
    "endDate": "2021-12-31",
    "initialBalance": 1000,
    "fee": 0.001,  # <-- ADDED
    "riskManagementMode": "standard",
    "riskPercentage": 1,
    "growthCapitalTarget": None,  # <-- ADDED
    "params": {},
    "optimizer_mode": True  # Use the cache ticket system
}

# --- Global Lists ---
ALL_AVAILABLE_MODELS = []
ALL_TA_STRATEGIES = [
    "sma_crossover", "rsi_divergence", "macd_crossover", "stochastic_crossover",
    "cci_oversold", "bollinger_bands", "ichimoku_cloud", "atr_signal",
    "obv_signal", "psar_signal"
]
EPSILON = 1e-9

# --- Helper Function ---


def set_nested_value(d, keys, value):
    """Sets a value in a nested dictionary using a dot-separated key."""
    keys = keys.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def fetch_all_models():
    """Called once at the start to get the *entire* model list."""
    global ALL_AVAILABLE_MODELS
    print(f"[Optimizer] Fetching all available models from {ML_SERVER_URL}...")
    try:
        url = f"{ML_SERVER_URL}/api/ml/available-models"
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            ALL_AVAILABLE_MODELS = response.json()
            print(f"[Optimizer] Found {len(ALL_AVAILABLE_MODELS)} total models.")
            if len(ALL_AVAILABLE_MODELS) == 0:
                print("[Optimizer] WARNING: No models found. ML modes will be skipped.")
        else:
            print(f"[Optimizer] ERROR: Could not fetch model list. {response.text}")
            return False
    except Exception as e:
        print(f"[Optimizer] CRITICAL ERROR: Could not connect to ML server. {e}")
        return False
    return True


def get_user_market_selection():
    """
    Parses all models, finds unique (symbol, timeframe) pairs,
    and asks the user to select which ones to test.
    """
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
        print("[MarketScan] No valid model markets found. Defaulting to BTC-USD / 1h.")
        return [("BTC-USD", "1h")]

    print("\n--- 💡 Available Markets (Based on your Models) ---")
    for i, (symbol, timeframe) in enumerate(sorted_pairs):
        print(f"  [{i+1}] {symbol} @ {timeframe}")
    print("  [0] TEST ALL OF THE ABOVE")

    while True:
        try:
            choice_str = input(f"\nWhich market(s) do you want to test? (e.g., 1, 3, 4): ")
            if not choice_str:
                continue

            if choice_str.strip() == "0":
                return sorted_pairs

            choices = [int(c.strip()) for c in choice_str.split(',')]
            selected_pairs = []
            for c in choices:
                if 1 <= c <= len(sorted_pairs):
                    selected_pairs.append(sorted_pairs[c-1])
                else:
                    print(f"Invalid choice: {c}")

            if selected_pairs:
                return selected_pairs
            else:
                print("No valid markets selected.")
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")


def generate_wfo_folds():
    """Generates a list of date ranges for Walk-Forward Optimization."""
    folds = [
        {"startDate": "2023-01-01", "endDate": "2023-12-31"},
        {"startDate": "2024-01-01", "endDate": "2024-12-31"},
        {"startDate": "2025-01-01", "endDate": datetime.now(timezone.utc).strftime("%Y-%m-%d")},
    ]
    print(f"[WFO] Generated {len(folds)} testing folds.")
    return folds


def run_single_backtest(config):
    """Helper function to call the correct endpoint based on config."""
    if "strategies" in config:
        url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
    else:
        url = f"{ML_SERVER_URL}/api/ml/run-backtest-on"

    response = requests.post(url, json=config, timeout=600)

    if response.status_code != 200:
        print(f"      🔥 FAILED: {response.status_code} - {response.text[:150]}")
        raise optuna.TrialPruned(response.text)

    results = response.json()

    # Handle the cache ticket system
    if "optimizer_ticket_id" in results:
        ticket_id = results["optimizer_ticket_id"]
        cache_path = os.path.join(OPTIMIZER_CACHE_DIR, f"{ticket_id}.json")

        for _ in range(10):  # Try for 10 seconds
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    file_results = json.load(f)
                os.remove(cache_path)  # Clean up the file
                return file_results  # Return the data from the file
            time.sleep(1)

        raise optuna.TrialPruned(f"Cache file {ticket_id} never appeared.")

    # Handle non-cache mode (e.g., if optimizer_mode was False)
    if "strategies" in config:
        return results.get('combinedResult', results)
    else:
        return results


def calculate_stitched_metrics(equity_curves: list) -> dict:
    """Stitches equity curves and calculates robust, continuous metrics."""
    if not equity_curves:
        return {}

    all_dfs = []
    last_balance = 0

    for i, fold_curve in enumerate(equity_curves):
        if not fold_curve:
            continue

        df = pd.DataFrame(fold_curve)

        if i == 0:
            df['balance_continuous'] = df['balance']
            last_balance = df['balance'].iloc[-1]
        else:
            initial_fold_balance = fold_curve[0]['balance']
            df['balance_continuous'] = df['balance'] - initial_fold_balance + last_balance
            last_balance = df['balance_continuous'].iloc[-1]

        all_dfs.append(df)

    if not all_dfs:
        return {"totalTrades": 0}

    stitched_df = pd.concat(all_dfs, ignore_index=True)
    stitched_df['timestamp'] = pd.to_datetime(stitched_df['timestamp'])
    stitched_df = stitched_df.set_index('timestamp')

    equity_series = stitched_df['balance_continuous']

    initial_balance = equity_series.iloc[0]
    final_balance = equity_series.iloc[-1]

    if final_balance <= 0 or initial_balance <= 0:
        print("      ⚠️  Metrics Warning: Strategy went broke. Setting Calmar to -999.")
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
        if days < 1:
            days = 1

        annual_return_rate = ((final_balance / (initial_balance + EPSILON)) ** (365.25 / days)) - 1
        annual_std_dev = daily_returns.std() * np.sqrt(365.25)
        sharpe_ratio = (annual_return_rate - RISK_FREE_RATE) / (annual_std_dev + EPSILON)

        downside_returns = daily_returns[daily_returns < 0]
        annual_downside_std = downside_returns.std() * np.sqrt(365.25)
        sortino_ratio = (annual_return_rate - RISK_FREE_RATE) / (annual_downside_std + EPSILON)

        calmar_ratio = (annual_return_rate * 100) / (max_drawdown_pct + EPSILON)
    else:
        sharpe_ratio = sortino_ratio = calmar_ratio = 0.0

    return {
        "StitchedCalmarRatio": calmar_ratio,
        "StitchedSharpeRatio": sharpe_ratio,
        "StitchedMaxDrawdown": max_drawdown_pct,
        "StitchedTotalReturn": total_return_pct,
    }


# --- 🚀 START: UPGRADED OBJECTIVE FUNCTION (v7 - ML Edge Hunt) 🚀 ---
def create_objective(symbol, timeframe, wfo_folds):
    """
    Factory function to create the main objective function
    for a specific market.
    """

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        """
        This is a focused A/B test to see if ML can add an edge
        to our proven robust baseline strategy.
        """

        test_config = deepcopy(BASE_CONFIG)
        test_config['symbol'] = symbol
        test_config['timeframe'] = timeframe

        # --- 2. Define Your "Search Space" ---

        # --- 💡 LOCK IN THE ROBUST BASELINE STRATEGY 💡 ---
        test_type = trial.suggest_categorical("test_type", ["single"])
        code = trial.suggest_categorical("code", ["psar_signal"])
        trend_filter = trial.suggest_categorical("params.trendFilterPeriod", [50])
        adx_filter = trial.suggest_categorical("params.minAdxLevel", [0])
        min_atr_filter = trial.suggest_categorical("params.minAtrPct", [0.1])
        atr_tsl = trial.suggest_categorical("params.tslAtrMult", [6.5])
        
        # Set all the locked-in values
        test_config['code'] = code
        set_nested_value(test_config, "params.trendFilterPeriod", trend_filter)
        set_nested_value(test_config, "params.minAdxLevel", adx_filter)
        set_nested_value(test_config, "params.minAtrPct", min_atr_filter)
        set_nested_value(test_config, "params.tslAtrMult", atr_tsl)
        set_nested_value(test_config, "params.SL", None) # Ensure fixed SL is off
        set_nested_value(test_config, "params.TP", None) # Ensure fixed TP is off

        # --- 💡 THIS IS THE A/B TEST 💡 ---
        # We will test the baseline "off" (control) vs. "predictions" (experimental)
        ml_mode = trial.suggest_categorical("mlMode", ["off", "predictions"])
        test_config['mlMode'] = ml_mode
        
        # --- ML Model Setup (if 'predictions' is chosen) ---
        if test_config['mlMode'] == "predictions":
            # We must use "AND" mode to combine TA + ML
            set_nested_value(test_config, "params.hybridMode_single", "AND")

            symbol_prefix = symbol.split('-')[0].lower()
            model_prefix = f"{symbol_prefix}_{timeframe}_"
            matching_models = [m['id'] for m in ALL_AVAILABLE_MODELS if m['id'].startswith(model_prefix)]

            if not matching_models:
                # This trial is invalid if it can't run the 'predictions' part
                raise optuna.TrialPruned(f"No models found for {model_prefix} to run 'predictions' test.")
            
            # --- These are the only parameters being optimized ---
            test_config['mlModel'] = trial.suggest_categorical("mlModel", matching_models)
            test_config['mlThreshold'] = trial.suggest_float("mlThreshold", 0.50, 0.95, step=0.05)
        
        # --- END OF UPGRADES ---

        # --- 3. Run the Walk-Forward Optimization ---
        test_name = f"Trial #{trial.number}"
        all_fold_equity_curves = []
        total_trades = 0

        for fold in wfo_folds:
            fold_config = deepcopy(test_config)
            fold_config['startDate'] = fold['startDate']
            fold_config['endDate'] = fold['endDate']

            try:
                results = run_single_backtest(fold_config)

                if results.get('metrics', {}).get('totalTrades', 0) < 5:
                    raise optuna.TrialPruned(f"Fold failed (Only {results.get('metrics', {}).get('totalTrades', 0)} trades)")

                all_fold_equity_curves.append(results.get('equity'))
                total_trades += results.get('metrics', {}).get('totalTrades', 0)

            except Exception as e:
                raise optuna.TrialPruned(f"Fold failed: {e}")

        # --- 4. Calculate Stitched Metrics & Return Score ---
        if not all_fold_equity_curves:
            raise optuna.TrialPruned("No folds completed successfully.")

        stitched_metrics = calculate_stitched_metrics(all_fold_equity_curves)

        calmar_ratio = stitched_metrics.get('StitchedCalmarRatio', 0.0)
        max_drawdown = stitched_metrics.get('StitchedMaxDrawdown', 999.0)

        print(f"      ✅ AVG SUCCESS: {test_name} | Calmar: {calmar_ratio:.2f} | DD: {max_drawdown:.2f}% | Total Trades: {total_trades}")

        return calmar_ratio, max_drawdown

    return objective
# --- 🚀 END: UPGRADED OBJECTIVE FUNCTION 🚀 ---


# --- 🆕 UPGRADED: Certification Function ---

def run_certification_check(market_id: str, best_params: dict, report_csv_file: str, base_config: dict):
    """
    Calls the new server endpoint to run certification (validation, holdout, bootstrap).
    Returns the full report dict on success, None on failure.
    """
    symbol, timeframe = market_id.split('_')

    # Clean up params for JSON serialization (remove nan)
    clean_params = {}
    for key, value in best_params.items():
        if isinstance(value, float) and np.isnan(value):
            continue
        clean_params[key] = value

    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "best_params": clean_params,
        "base_config": base_config
    }

    print(f"      - Sending best strategy to server for certification...")
    try:
        url = f"{ML_SERVER_URL}/api/ml/certify-strategy"
        response = requests.post(url, json=payload, timeout=900)  # 15-minute timeout

        if response.status_code != 200:
            print(f"      🔥 CERTIFICATION FAILED: {response.status_code} - {response.text}")
            return None  # <-- Return None on failure

        report = response.json()

        # Save the certification report
        report_path = report_csv_file.replace('.csv', '_certification.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"      ✅ Certification report saved to: {report_path}")

        # Print a summary
        print(f"      --- Certification Summary ---")
        print(f"      - Certification Passed: {report.get('certification_passed')}")
        print(f"      - In-Sample Calmar: {report.get('robustness_checks', {}).get('calmar_in_sample', 0):.2f}")
        print(f"      - Holdout Calmar: {report.get('robustness_checks', {}).get('calmar_holdout', 0):.2f}")
        print(f"      - Calmar Decay: {report.get('robustness_checks', {}).get('calmar_decay_pct', 0):.2f}%")
        print(f"      - 95% CI (Holdout Calmar): [{report.get('holdout_bootstrap_calmar', {}).get('ci_lower_bound', 0):.2f}, {report.get('holdout_bootstrap_calmar', {}).get('ci_upper_bound', 0):.2f}]")

        return report  # <-- Return the full report on success

    except Exception as e:
        print(f"      🔥 CERTIFICATION FAILED: An error occurred: {e}")
        return None  # <-- Return None on failure

# --- 🆕 UPGRADED: Reproducibility Check Function ---


def run_reproducibility_check(market_id: str, best_params: dict) -> bool:
    """
    Re-runs the *exact* best trial multiple times to check for determinism.
    Returns True on pass, False on fail.
    """
    symbol, timeframe = market_id.split('_')

    # 1. Reconstruct the *exact* config for the best trial
    base_config = deepcopy(BASE_CONFIG)
    base_config['symbol'] = symbol
    base_config['timeframe'] = timeframe
    base_config['optimizer_mode'] = True  # Use cache

    for key, value in best_params.items():
        if isinstance(value, float) and np.isnan(value):
            continue
        if key in ["CalmarRatio", "MaxDrawdown"]:
            continue
        set_nested_value(base_config, key, value)

    # 2. Run the backtest N times and store metrics
    results_list = []
    wfo_folds = generate_wfo_folds()  # Folds must be identical

    for i in range(REPRODUCIBILITY_RUNS):
        print(f"      - Running reproducibility test {i+1}/{REPRODUCIBILITY_RUNS}...")
        all_fold_equity_curves = []

        try:
            for fold in wfo_folds:
                fold_config = deepcopy(base_config)
                fold_config['startDate'] = fold['startDate']
                fold_config['endDate'] = fold['endDate']

                fold_results = run_single_backtest(fold_config)
                
                all_fold_equity_curves.append(fold_results.get('equity'))

            stitched_metrics = calculate_stitched_metrics(all_fold_equity_curves)
            results_list.append(stitched_metrics)

        except Exception as e:
            print(f"            🔥 Run {i+1} failed to complete: {e}")
            results_list.append(None)  # Mark as failure
            break  # Stop checking

    # 3. Compare the results
    if len(results_list) != REPRODUCIBILITY_RUNS or any(r is None for r in results_list):
        print("      🔥 REPRODUCIBILITY FAILED: One or more runs failed to complete.")
        return False  # <-- Return False

    first_calmar = results_list[0]['StitchedCalmarRatio']
    first_dd = results_list[0]['StitchedMaxDrawdown']

    passed = True
    for i, metrics in enumerate(results_list[1:], 1):
        calmar_diff = abs(metrics['StitchedCalmarRatio'] - first_calmar)
        dd_diff = abs(metrics['StitchedMaxDrawdown'] - first_dd)

        if calmar_diff > METRIC_TOLERANCE or dd_diff > METRIC_TOLERANCE:
            passed = False
        print(f"      - Run {i+1} vs Run 1 (Calmar Diff): {calmar_diff:.8f}")
        print(f"      - Run {i+1} vs Run 1 (DD Diff): {dd_diff:.8f}")

    if passed:
        print("      ✅ REPRODUCIBILITY PASSED: All metrics are within tolerance.")
        return True  # <-- Return True
    else:
        print(f"      🔥 REPRODUCIBILITY FAILED: Metrics differed by more than {METRIC_TOLERANCE}.")
        return False  # <-- Return False


# --- 🚀 Main function (Upgraded) ---
def main():
    """
    Main function to run the "Study of Studies".
    """

    # --- 🆕 NEW: Add Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Grandmaster Optimizer (v7) - ML Edge Hunt."
    )
    parser.add_argument(
        '--certify',
        action='store_true',
        help='Run certification (holdout + bootstrap) on the best result of each market.'
    )
    # --- 🚀 FIX: Corrected argparse method name ---
    parser.add_argument(
        '--repro-check',
        action='store_true',
        help='Run a reproducibility check on the best result of each market.'
    )
    args = parser.parse_args()
    # --- End New ---

    main_start_time = time.time()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[Optimizer] Results will be saved to: {RESULTS_DIR}")

    if not fetch_all_models():
        return

    markets_to_test = get_user_market_selection()

    print("\n" + "="*50)
    print(f"🚀 Starting {len(markets_to_test)} Optimization Studies...")
    print(f"      Markets: {markets_to_test}")
    print(f"      Trials per Study: {TOTAL_TRIALS}")
    print(f"      Parallel Jobs: {PARALLEL_JOBS}")
    print(f"      --- Modes ---")
    print(f"      Certification: {'ENABLED' if args.certify else 'DISABLED'}")
    print(f"      Reproducibility: {'ENABLED' if args.repro_check else 'DISABLED'}")
    print("="*50)

    # --- STAGE 2: Walk-Forward Optimization (Loop) ---
    for (symbol, timeframe) in markets_to_test:

        market_start_time = time.time()
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        market_id = f"{symbol}_{timeframe}"

        print(f"\n--- Starting Study for: {market_id} ---")

        db_name = f"study_{market_id}_{run_timestamp}.db"
        study_db_file = os.path.join(RESULTS_DIR, db_name)

        report_name = f"report_{market_id}_{run_timestamp}.csv"
        report_csv_file = os.path.join(RESULTS_DIR, report_name)

        pareto_html_file = f"report_{market_id}_{run_timestamp}_pareto.html"
        pareto_html_path = os.path.join(RESULTS_DIR, pareto_html_file)

        importance_html_file = f"report_{market_id}_{run_timestamp}_importance.html"
        importance_html_path = os.path.join(RESULTS_DIR, importance_html_file)

        wfo_folds = generate_wfo_folds()
        objective_func = create_objective(symbol, timeframe, wfo_folds)

        study = optuna.create_study(
            study_name=f"study_{market_id}_{run_timestamp}",
            storage=f"sqlite:///{study_db_file}",
            load_if_exists=False,
            directions=["maximize", "minimize"]  # Maximize Calmar, Minimize Drawdown
        )

        try:
            study.optimize(objective_func, n_trials=TOTAL_TRIALS, n_jobs=PARALLEL_JOBS)
        except Exception as e:
            print(f"[Optimizer] CRITICAL ERROR during study.optimize: {e}")
            continue

        market_end_time = time.time()
        print(f"\n--- 🏁 Study for {market_id} complete in {(market_end_time - market_start_time)/60:.2f} minutes ---")

        # --- 5. Save the Report for this Market ---
        try:
            best_trials = study.best_trials
            if not best_trials:
                print(f"No complete trials found for {market_id}.")
                continue

            print(f"Found {len(best_trials)} 'best' trade-offs for {market_id}.")

            report_data = []
            for trial in best_trials:
                calmar, dd = trial.values
                params = trial.params
                params['CalmarRatio'] = calmar
                params['MaxDrawdown'] = dd

                if "combo_strategies" in trial.user_attrs:
                    params['combo_strategies'] = trial.user_attrs['combo_strategies']

                report_data.append(params)

            df = pd.DataFrame(report_data).sort_values(by="CalmarRatio", ascending=False)

            df.to_csv(report_csv_file, index=False, float_format='%.3f')
            print(f"\n✅ Full report for {market_id} saved to {report_csv_file}")

            print(f"\n--- 🏆 Best 10 Results for {market_id} ---")
            print(df.head(10).to_string(index=False))

            # --- Visualization ---
            try:
                fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["Calmar Ratio", "Max Drawdown"])
                fig_pareto.write_html(pareto_html_path)
                print(f"✅ Pareto-front chart saved to {pareto_html_path}")

                fig_importance = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[0], target_name="Calmar Ratio")
                fig_importance.write_html(importance_html_path)
                print(f"✅ Parameter Importance chart saved to {importance_html_path}")
            except Exception as e:
                print(f"      ⚠️ Could not generate charts: {e} (This can happen if no trials completed successfully)")

            # --- 🆕 UPGRADED: Run Certification & Repro Checks ---
            if df.empty:
                print("      ⚠️ Report is empty, skipping Certification and Reproducibility checks.")
                continue

            # Get the single best strategy (top of the CSV)
            best_trial_params = df.iloc[0].to_dict()

            # --- 💡 GRADE LOGIC: Initialize variables ---
            cert_report = None
            certification_pass = False
            repro_pass = False

            if args.certify:
                print(f"\n--- 🔬 Starting Certification for: {market_id} ---")
                cert_report = run_certification_check(market_id, best_trial_params, report_csv_file, BASE_CONFIG)
                if cert_report:
                    certification_pass = cert_report.get('certification_passed', False)

            if args.repro_check:
                print(f"\n--- 🧬 Starting Reproducibility Check for: {market_id} ---")
                repro_pass = run_reproducibility_check(market_id, best_trial_params)

            # --- 💡 GRADE LOGIC: Run the Grader ---
            if args.certify or args.repro_check:
                final_metrics = {}
                if cert_report:
                    # Use the most robust metrics from the holdout test
                    final_metrics = cert_report.get('holdout_metrics', {})
                else:
                    # Fallback to optimizer metrics if cert wasn't run
                    final_metrics = {
                        "calmar_ratio": best_trial_params.get('CalmarRatio', 0),
                        "max_drawdown": best_trial_params.get('MaxDrawdown', 100),
                        "win_rate": 0  # Win rate is not available here
                    }

                grade_results = grade_strategy_quality(
                    metrics=final_metrics,
                    reproducibility_pass=repro_pass,
                    certification_pass=certification_pass
                )

                print("\n--- 🏅 FINAL STRATEGY GRADE ---")
                print(f"  Grade: {grade_results['grade']}")
                print(f"  Reliability: {grade_results['reliability']}%")
                print(f"  Certification: {'PASSED' if certification_pass else 'FAILED'}")
                print(f"  Reproducibility: {'PASSED' if repro_pass else 'FAILED'}")

        except Exception as e:
            print(f"Error generating report or running checks for {market_id}: {e}")

    main_end_time = time.time()
    print(f"\n--- 🚀🚀🚀 Full Optimization (All Markets) finished in {(main_end_time - main_start_time) / 60:.2f} minutes 🚀🚀🚀 ---")


if __name__ == "__main__":
    main()
