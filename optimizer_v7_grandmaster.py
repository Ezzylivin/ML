#!/usr/bin/env python3
# optimizer_v7_grandmaster.py
"""
Grandmaster Optimizer v7 - Full WFO + Certification + Persistent Bad Parameter Memory
Interactive CLI + ML Validation
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import requests

# -------------------------
# Deterministic seeding & config
# -------------------------
SEED = 12345
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.INFO)
np.seterr(all="ignore")

ML_SERVER_URL = os.environ.get("ML_SERVER_URL", "http://74.208.28.77:8000")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/root/Project/ML/data/optimizer_results")
OPTIMIZER_CACHE_DIR = os.environ.get("OPTIMIZER_CACHE_DIR", "/root/Project/ML/cache/optimizer")
BAD_PARAMS_FILE = os.path.join(OPTIMIZER_CACHE_DIR, "bad_params.json")

TOTAL_TRIALS = int(os.environ.get("TOTAL_TRIALS", "100"))
PARALLEL_JOBS = int(os.environ.get("PARALLEL_JOBS", "6"))
RISK_FREE_RATE = float(os.environ.get("RISK_FREE_RATE", "0.02"))
REPRODUCIBILITY_RUNS = int(os.environ.get("REPRODUCIBILITY_RUNS", "3"))
EPSILON = 1e-9

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OPTIMIZER_CACHE_DIR, exist_ok=True)

# -------------------------
# Globals
# -------------------------
ALL_TA_STRATEGIES = [
    "sma_crossover", "rsi_divergence", "macd_crossover", "stochastic_crossover",
    "cci_oversold", "bollinger_bands", "ichimoku_cloud", "atr_signal",
    "obv_signal", "psar_signal"
]

BASE_CONFIG = {
    "symbol": "BTC-USD",
    "timeframe": "1h",
    "startDate": "2021-01-01",
    "endDate": "2021-12-31",
    "initialBalance": 300,
    "fee": 0.001,
    "riskManagementMode": "standard",
    "riskPercentage": 1,
    "params": {},
    "optimizer_mode": True
}

# -------------------------
# Persistent Bad Parameters
# -------------------------
def load_bad_params() -> List[Dict[str, Any]]:
    if os.path.exists(BAD_PARAMS_FILE):
        try:
            with open(BAD_PARAMS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_bad_params(bad_params: List[Dict[str, Any]]):
    tmp = BAD_PARAMS_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(bad_params, f, indent=2)
        os.replace(tmp, BAD_PARAMS_FILE)
    except Exception as e:
        print(f"[BadParams] Could not save: {e}", file=sys.stderr)

BAD_PARAMS: List[Dict[str, Any]] = load_bad_params()

def is_bad_trial(params: Dict[str, Any]) -> bool:
    for b in BAD_PARAMS:
        if all(params.get(k) == v for k, v in b.items()):
            return True
    return False

# -------------------------
# Utilities
# -------------------------
def sanitize_float(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): sanitize_float(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_float(v) for v in obj]
    return obj

def safe_json_dump(obj: Any, path: str, indent: int = 2) -> None:
    sanitized = sanitize_float(obj)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(sanitized, f, indent=indent)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[safe_json_dump] Could not save JSON to {path}: {e}", file=sys.stderr)

# -------------------------
# API helpers
# -------------------------
def fetch_all_models() -> List[str]:
    try:
        r = requests.get(f"{ML_SERVER_URL}/api/ml/available-models", timeout=30)
        if r.status_code == 200:
            models = r.json()
            return models if isinstance(models, list) else []
        return []
    except Exception as e:
        print(f"[Optimizer] Failed to fetch models: {e}")
        return []

def run_single_backtest(config: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest" if config.get("strategies") else f"{ML_SERVER_URL}/api/ml/run-backtest-on"
    r = requests.post(url, json=config, timeout=600)
    if r.status_code != 200:
        raise RuntimeError(f"Backtest failed: {r.status_code} {r.text[:200]}")
    return sanitize_float(r.json())

# -------------------------
# Metrics calculation
# -------------------------
def calculate_stitched_metrics(equity_curves: List[List[Dict[str, Any]]]) -> Dict[str, float]:
    if not equity_curves:
        return {}
    all_frames = []
    last_balance = None
    for i, curve in enumerate(equity_curves):
        if not curve:
            continue
        df = pd.DataFrame(curve)
        if "timestamp" not in df.columns or "balance" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        if i == 0:
            df["balance_continuous"] = df["balance"]
            last_balance = float(df["balance"].iloc[-1])
        else:
            initial_fold_balance = float(df["balance"].iloc[0])
            df["balance_continuous"] = df["balance"] - initial_fold_balance + (last_balance or 0.0)
            last_balance = float(df["balance_continuous"].iloc[-1])
        all_frames.append(df[["balance", "balance_continuous"]].copy())
    if not all_frames:
        return {"totalTrades": 0}
    stitched = pd.concat(all_frames).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    equity_series = stitched["balance_continuous"].astype(float).fillna(method="ffill").fillna(0.0)
    if equity_series.empty:
        return {"totalTrades": 0}

    initial_balance = float(equity_series.iloc[0])
    final_balance = float(equity_series.iloc[-1])
    total_return_pct = (final_balance / (initial_balance + EPSILON) - 1.0) * 100.0
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / (peak + EPSILON)
    max_drawdown_pct = abs(float(drawdown.min()) * 100.0)
    daily_returns = equity_series.pct_change().fillna(0.0)
    days = max(1, (equity_series.index[-1] - equity_series.index[0]).days)
    annual_return_rate = ((final_balance / (initial_balance + EPSILON)) ** (365.25 / days)) - 1.0
    annual_std_dev = float(daily_returns.std()) * np.sqrt(365.25)
    sharpe_ratio = (annual_return_rate - RISK_FREE_RATE) / (annual_std_dev + EPSILON) if annual_std_dev > 0 else 0.0
    downside = daily_returns[daily_returns < 0]
    annual_downside_std = float(downside.std()) * np.sqrt(365.25) if len(downside) > 1 else 0.0
    sortino_ratio = (annual_return_rate - RISK_FREE_RATE) / (annual_downside_std + EPSILON) if annual_downside_std > 0 else 0.0
    calmar_ratio = (annual_return_rate * 100.0) / (max_drawdown_pct + EPSILON) if max_drawdown_pct > 0 else 0.0
    return {
        "StitchedCalmarRatio": float(np.nan_to_num(calmar_ratio, neginf=0.0, posinf=0.0)),
        "StitchedSharpeRatio": float(np.nan_to_num(sharpe_ratio, neginf=0.0, posinf=0.0)),
        "StitchedMaxDrawdown": float(np.nan_to_num(max_drawdown_pct, neginf=100.0, posinf=100.0)),
        "StitchedTotalReturn": float(np.nan_to_num(total_return_pct, neginf=-100.0, posinf=1000000.0)),
    }

# -------------------------
# Objective
# -------------------------
def create_objective(symbol: str, timeframe: str, wfo_folds: List[Dict[str, str]]):
    def objective(trial: optuna.Trial) -> Tuple[float, float]:
        test_config = deepcopy(BASE_CONFIG)
        test_config["symbol"] = symbol
        test_config["timeframe"] = timeframe

        test_type = trial.suggest_categorical("test_type", ["single", "combo"])
        trend_filter = trial.suggest_categorical("params.trendFilterPeriod", [0, 50, 100, 200])
        adx_filter = trial.suggest_categorical("params.minAdxLevel", [0, 20, 25])
        min_atr_filter = trial.suggest_categorical("params.minAtrPct", [0.0, 0.1, 0.2, 0.5])
        test_config["mlMode"] = trial.suggest_categorical("mlMode", ["off"])

        if test_type == "single":
            code = trial.suggest_categorical("code", ALL_TA_STRATEGIES)
            test_config["code"] = code
        else:
            num_strats = trial.suggest_int("combo_size", 3, min(len(ALL_TA_STRATEGIES), 7))
            combo_codes = random.sample(ALL_TA_STRATEGIES, num_strats)
            trial.set_user_attr("combo_strategies", ", ".join(combo_codes))
            test_config["strategies"] = [{"code": c, "params": {}} for c in combo_codes]
            test_config.setdefault("params", {})["hybridMode"] = trial.suggest_categorical("params.hybridMode", ["AND"])

        atr_tsl = trial.suggest_float("params.tslAtrMult", 1.5, 7.0, step=0.5)
        test_config.setdefault("params", {})["SL"] = None
        test_config["params"]["TP"] = None
        test_config["params"]["tslAtrMult"] = atr_tsl
        test_config["params"]["trendFilterPeriod"] = trend_filter
        test_config["params"]["minAdxLevel"] = adx_filter
        test_config["params"]["minAtrPct"] = min_atr_filter

        flat_params = {k: v for k, v in test_config.get("params", {}).items()}
        if is_bad_trial(flat_params):
            raise optuna.TrialPruned("Known bad parameters, skipping trial.")

        all_fold_equities = []
        total_trades = 0
        for fold in wfo_folds:
            fold_cfg = deepcopy(test_config)
            fold_cfg["startDate"] = fold["startDate"]
            fold_cfg["endDate"] = fold["endDate"]
            try:
                res = run_single_backtest(fold_cfg)
                metrics = res.get("metrics", {}) or {}
                n_trades = int(metrics.get("totalTrades", 0))
                if n_trades < 5:
                    BAD_PARAMS.append(flat_params)
                    save_bad_params(BAD_PARAMS)
                    raise optuna.TrialPruned(f"Few trades ({n_trades}), added to bad params")
                equity = res.get("equity") or []
                all_fold_equities.append(equity)
                total_trades += n_trades
            except optuna.TrialPruned:
                raise
            except Exception as e:
                BAD_PARAMS.append(flat_params)
                save_bad_params(BAD_PARAMS)
                raise optuna.TrialPruned(f"Fold failed: {e}")

        if not all_fold_equities:
            raise optuna.TrialPruned("No successful folds")
        stitched = calculate_stitched_metrics(all_fold_equities)
        calmar = float(stitched.get("StitchedCalmarRatio", 0.0))
        max_dd = float(stitched.get("StitchedMaxDrawdown", 999.0))
        print(f"Trial #{trial.number}: Calmar={calmar:.4f}, MaxDD={max_dd:.4f}, Trades={total_trades}")
        return calmar, max_dd
    return objective

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Grandmaster Optimizer (v7)")
    parser.add_argument("--trials", type=int, default=TOTAL_TRIALS)
    parser.add_argument("--jobs", type=int, default=PARALLEL_JOBS)
    args = parser.parse_args()
    trials = args.trials
    jobs = args.jobs

    # --- Fetch ML models ---
    print(f"[Optimizer] Fetching all models from {ML_SERVER_URL}...")
    models = fetch_all_models()
    print(f"[Optimizer] Found {len(models)} models.\n")

    # --- Available markets ---
    available_markets = [
        ("BTC-USD", "1d"), ("BTC-USD", "1h"), ("BTC-USD", "30m"), ("BTC-USD", "4h"),
        ("ETH-USD", "1d"), ("ETH-USD", "1h"), ("ETH-USD", "30m"), ("ETH-USD", "4h"),
        ("SOL-USD", "30m"), ("XRP-USD", "1d"), ("XRP-USD", "1h"), ("XRP-USD", "30m"), ("XRP-USD", "4h")
    ]
    print("--- Available Markets ---")
    for i, (symbol, tf) in enumerate(available_markets, 1):
        print(f"  [{i}] {symbol} @ {tf}")
    print("  [0] TEST ALL")
    selected = input("Which market(s) to test? (e.g., 1,3,4): ").strip()
    if selected == "0":
        selected_indices = list(range(len(available_markets)))
    else:
        selected_indices = [int(x)-1 for x in selected.split(",") if x.strip().isdigit()]
    markets = [available_markets[i] for i in selected_indices]

    # --- Print optimizer summary ---
    print("\n==================================================")
    print(f"🚀 Starting {len(markets)} TA Parameter Optimization Studies...")
    print(f"   (Mode: ML-OFF, TSL-Only)")
    print(f"   Trials per Study: {trials}")
    print(f"   Parallel Jobs: {jobs}")
    print("==================================================\n")

    # --- WFO folds ---
    wfo_folds = [
        {"startDate": "2023-01-01", "endDate": "2023-12-31"},
        {"startDate": "2024-01-01", "endDate": "2024-12-31"},
        {"startDate": "2025-01-01", "endDate": datetime.now(timezone.utc).strftime("%Y-%m-%d")},
    ]

    # --- Run studies ---
    for symbol, timeframe in markets:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_db = os.path.join(RESULTS_DIR, f"study_{symbol}_{timeframe}_{run_stamp}.db")
        print(f"--- Starting Study for: {symbol}_{timeframe} ---")
        print(f"[WFO] Generated {len(wfo_folds)} testing folds ({wfo_folds[0]['startDate']}-Present).")

        objective = create_objective(symbol, timeframe, wfo_folds)
        study = optuna.create_study(
            study_name=f"study_{symbol}_{timeframe}_{run_stamp}",
            storage=f"sqlite:///{study_db}",
            load_if_exists=False,
            directions=["maximize", "minimize"]
        )
        study.optimize(objective, n_trials=trials, n_jobs=jobs)

        # --- Certification ---
        top_trials = sorted(study.trials, key=lambda t: t.values[0], reverse=True)[:5]
        certified_results = []
        for t in top_trials:
            trial_cfg = t.params
            fold_results = []
            for _ in range(REPRODUCIBILITY_RUNS):
                try:
                    res = run_single_backtest(BASE_CONFIG | {"params": trial_cfg})
                    fold_results.append(res.get("metrics", {}))
                except Exception:
                    continue
            certified_results.append({"trial": t.number, "params": trial_cfg, "fold_results": fold_results})

        safe_json_dump(certified_results, os.path.join(RESULTS_DIR, f"certified_{symbol}_{timeframe}_{run_stamp}.json"))
        print(f"Study {symbol}_{timeframe} completed. Certification saved.\n")

if __name__ == "__main__":
    main()
