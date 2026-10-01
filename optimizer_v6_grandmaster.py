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

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.INFO)

# --- CONFIGURATION ---
ML_SERVER_URL = "http://74.208.28.77:8000"
RESULTS_DIR = "/root/Project/ML/data/optimizer_results"
OPTIMIZER_CACHE_DIR = "/root/Project/ML/cache/optimizer/"
TOTAL_TRIALS = 100
PARALLEL_JOBS = 6
RISK_FREE_RATE = 0.02

BASE_CONFIG = {
    "symbol": "BTC-USD",
    "timeframe": "1h",
    "startDate": "2021-01-01",
    "endDate": "2021-12-31",
    "initialBalance": 1000,
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
    folds = [
        {"startDate": "2023-01-01", "endDate": "2023-12-31"},
        {"startDate": "2024-01-01", "endDate": "2024-12-31"},
        {"startDate": "2025-01-01", "endDate": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
    ]
    print(f"[WFO] Generated {len(folds)} folds.")
    return folds

def run_single_backtest(config):
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest" if "strategies" in config else f"{ML_SERVER_URL}/api/ml/run-backtest-on"
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

def calculate_stitched_metrics(equity_curves):
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

# --- Objective Function ---
def create_objective(symbol, timeframe, wfo_folds):
    def objective(trial: optuna.Trial):
        test_config = deepcopy(BASE_CONFIG)
        test_config['symbol'] = symbol
        test_config['timeframe'] = timeframe

        # Search Space
        test_type = trial.suggest_categorical("test_type", ["single", "combo"])
        trend_filter = trial.suggest_categorical("params.trendFilterPeriod", [0,50,100,200])
        adx_filter = trial.suggest_categorical("params.minAdxLevel", [0,20,25])
        min_atr_filter = trial.suggest_categorical("params.minAtrPct", [0.0,0.1,0.2,0.5])
        ml_mode = trial.suggest_categorical("mlMode", ["off"])
        atr_tsl = trial.suggest_float("params.tslAtrMult", 1.5, 7.0, step=0.5)

        set_nested_value(test_config, "params.trendFilterPeriod", trend_filter)
        set_nested_value(test_config, "params.minAdxLevel", adx_filter)
        set_nested_value(test_config, "params.minAtrPct", min_atr_filter)
        set_nested_value(test_config, "params.tslAtrMult", atr_tsl)
        test_config['mlMode'] = ml_mode

        if test_type == "single":
            test_config['code'] = trial.suggest_categorical("code", ALL_TA_STRATEGIES)
        else:
            num_strats = trial.suggest_int("combo_size",3,7)
            combo_codes = random.sample(ALL_TA_STRATEGIES,num_strats)
            trial.set_user_attr("combo_strategies", ", ".join(combo_codes))
            test_config['strategies'] = [{"code": code, "params": {}} for code in combo_codes]
            hybrid_mode = trial.suggest_categorical("params.hybridMode", ["AND"])
            set_nested_value(test_config, "params.hybridMode", hybrid_mode)

        # Run WFO
        all_fold_equity_curves = []
        total_trades = 0
        for fold in wfo_folds:
            fold_config = deepcopy(test_config)
            fold_config['startDate'] = fold['startDate']
            fold_config['endDate'] = fold['endDate']
            try:
                results = run_single_backtest(fold_config)
                if results.get('metrics', {}).get('totalTrades', 0) < 5:
                    raise optuna.TrialPruned(f"Fold failed (trades < 5)")
                all_fold_equity_curves.append(results.get('equity'))
                total_trades += results.get('metrics', {}).get('totalTrades', 0)
            except Exception as e:
                raise optuna.TrialPruned(f"Fold failed: {e}")

        stitched_metrics = calculate_stitched_metrics(all_fold_equity_curves)
        calmar_ratio = stitched_metrics.get('StitchedCalmarRatio', 0.0)
        max_drawdown = stitched_metrics.get('StitchedMaxDrawdown', 999.0)
        print(f"Trial #{trial.number} | Calmar: {calmar_ratio:.2f} | DD: {max_drawdown:.2f}% | Trades: {total_trades}")
        return calmar_ratio, max_drawdown

    return objective

# --- Certification & Repro ---
def run_certification_check(market_id, best_params, report_csv_file, base_config):
    symbol, timeframe = market_id.split('_')
    clean_params = {k:v for k,v in best_params.items() if not (isinstance(v,float) and np.isnan(v))}
    payload = {"symbol": symbol, "timeframe": timeframe, "best_params": clean_params, "base_config": base_config}
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

def run_reproducibility_check(market_id, best_params):
    symbol, timeframe = market_id.split('_')
    base_config = deepcopy(BASE_CONFIG)
    base_config['symbol'] = symbol
    base_config['timeframe'] = timeframe
    base_config['optimizer_mode'] = True
    for key,value in best_params.items():
        if key not in ["CalmarRatio","MaxDrawdown"]:
            set_nested_value(base_config,key,value)
    results_list=[]
    wfo_folds = generate_wfo_folds()
    for i in range(REPRODUCIBILITY_RUNS):
        print(f"Repro run {i+1}/{REPRODUCIBILITY_RUNS}")
        all_fold_equity_curves=[]
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
            print(f"Run {i+1} failed: {e}")
            return False
    first_calmar = results_list[0]['StitchedCalmarRatio']
    first_dd = results_list[0]['StitchedMaxDrawdown']
    for i,metrics in enumerate(results_list[1:],1):
        if abs(metrics['StitchedCalmarRatio']-first_calmar)>METRIC_TOLERANCE or abs(metrics['StitchedMaxDrawdown']-first_dd)>METRIC_TOLERANCE:
            print("Reproducibility FAILED")
            return False
    print("Reproducibility PASSED")
    return True

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Optimizer v7.1")
    parser.add_argument('--certify',action='store_true')
    parser.add_argument('--repro-check',action='store_true')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not fetch_all_models():
        return
    markets_to_test = get_user_market_selection()
    print(f"Starting {len(markets_to_test)} studies...")

    for symbol,timeframe in markets_to_test:
        run_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        market_id=f"{symbol}_{timeframe}"
        study_db_file=os.path.join(RESULTS_DIR,f"study_{market_id}_{run_timestamp}.db")
        report_csv_file=os.path.join(RESULTS_DIR,f"report_{market_id}_{run_timestamp}.csv")
        wfo_folds=generate_wfo_folds()
        objective_func=create_objective(symbol,timeframe,wfo_folds)
        study=optuna.create_study(
            study_name=f"study_{market_id}_{run_timestamp}",
            storage=f"sqlite:///{study_db_file}",
            load_if_exists=False,
            directions=["maximize","minimize"]
        )
        try:
            study.optimize(objective_func,n_trials=TOTAL_TRIALS,n_jobs=PARALLEL_JOBS)
        except Exception as e:
            print(f"Study error: {e}")
            continue
        best_trials=study.best_trials
        if not best_trials:
            continue
        report_data=[]
        for trial in best_trials:
            calmar,dd=trial.values
            params=trial.params
            params['CalmarRatio']=calmar
            params['MaxDrawdown']=dd
            if "combo_strategies" in trial.user_attrs:
                params['combo_strategies']=trial.user_attrs['combo_strategies']
            report_data.append(params)
        df=pd.DataFrame(report_data).sort_values("CalmarRatio",ascending=False)
        df.to_csv(report_csv_file,index=False,float_format='%.3f')
        print(f"Report saved: {report_csv_file}\nTop 10:\n{df.head(10)}")

        # --- Certification & Repro ---
        best_trial_params=df.iloc[0].to_dict()
        certification_pass=False
        repro_pass=False
        cert_report=None
        if args.certify:
            cert_report=run_certification_check(market_id,best_trial_params,report_csv_file,BASE_CONFIG)
            if cert_report:
                certification_pass=cert_report.get('certification_passed',False)
        if args.repro_check:
            repro_pass=run_reproducibility_check(market_id,best_trial_params)
        final_metrics=cert_report.get('holdout_metrics',{}) if cert_report else {'calmar_ratio':best_trial_params['CalmarRatio'],'max_drawdown':best_trial_params['MaxDrawdown'],'win_rate':0}
        grade_results=grade_strategy_quality(final_metrics,repro_pass,certification_pass)
        print("\n--- FINAL STRATEGY GRADE ---")
        print(f"Grade: {grade_results['grade']}")
        print(f"Reliability: {grade_results['reliability']}%")
        print(f"Certification: {'PASSED' if certification_pass else 'FAILED'}")
        print(f"Reproducibility: {'PASSED' if repro_pass else 'FAILED'}")

if __name__=="__main__":
    main()
