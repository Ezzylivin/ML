#!/usr/bin/env python3
# optimizer_v6_grandmaster.py
"""
Grandmaster Optimizer (v6) - Find, Certify, and Verify Strategies.
Rewritten with robust sanitization and safety checks to avoid JSON serialization
failures (NaN/inf), improve reproducibility checks, and provide clearer logging.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import requests

# Attempt to import grading helper (optional)
try:
    from data_quality_grader import grade_strategy_quality
except Exception:
    def grade_strategy_quality(metrics, reproducibility_pass, certification_pass):
        # Fallback stub
        return {"grade": "N/A", "reliability": 0}

# -------------------------
# Deterministic seeding & global config
# -------------------------
SEED = 12345
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.INFO)

# Silence numpy warnings that are expected during exploration (we still handle results)
np.seterr(all='ignore')

# -------------------------
# Configuration (edit if needed)
# -------------------------
ML_SERVER_URL = os.environ.get("ML_SERVER_URL", "http://74.208.28.77:8000")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/root/Project/ML/data/optimizer_results")
OPTIMIZER_CACHE_DIR = os.environ.get("OPTIMIZER_CACHE_DIR", "/root/Project/ML/cache/optimizer")
TOTAL_TRIALS = int(os.environ.get("TOTAL_TRIALS", "100"))
PARALLEL_JOBS = int(os.environ.get("PARALLEL_JOBS", "6"))
RISK_FREE_RATE = float(os.environ.get("RISK_FREE_RATE", "0.02"))
REPRODUCIBILITY_RUNS = int(os.environ.get("REPRODUCIBILITY_RUNS", "5"))
METRIC_TOLERANCE = float(os.environ.get("METRIC_TOLERANCE", "1e-6"))
EPSILON = 1e-9

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OPTIMIZER_CACHE_DIR, exist_ok=True)

# -------------------------
# Globals
# -------------------------
ALL_AVAILABLE_MODELS: List[Dict[str, Any]] = []
ALL_TA_STRATEGIES = [
    "sma_crossover", "rsi_divergence", "macd_crossover", "stochastic_crossover",
    "cci_oversold", "bollinger_bands", "ichimoku_cloud", "atr_signal",
    "obv_signal", "psar_signal"
]

# Base config for trials
BASE_CONFIG = {
    "symbol": "BTC-USD",
    "timeframe": "1h",
    "startDate": "2021-01-01",
    "endDate": "2021-12-31",
    "initialBalance": 300,
    "fee": 0.001,
    "riskManagementMode": "standard",
    "riskPercentage": 1,
    "growthCapitalTarget": None,
    "params": {},
    "optimizer_mode": True
}

# -------------------------
# Utilities: sanitization + JSON safe dump
# -------------------------
def sanitize_float(obj: Any) -> Any:
    """
    Recursively sanitize an object by replacing NaN/inf/-inf floats with safe values.
    - Floats that are NaN/inf/-inf -> replaced with 0.0 (safe default)
    - For dicts/lists -> apply recursively
    - Leaves other types intact
    """
    if obj is None:
        return None
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return val
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): sanitize_float(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_float(v) for v in obj]
    # Pandas objects
    try:
        import pandas as pd
        if isinstance(obj, pd.Series):
            return sanitize_float(obj.tolist())
        if isinstance(obj, pd.DataFrame):
            return sanitize_float(obj.to_dict(orient="records"))
    except Exception:
        pass
    # fallback: return as-is
    return obj

def safe_json_dump(obj: Any, path: str, indent: int = 2) -> None:
    """
    Write JSON to disk after sanitizing. Ensures failure doesn't leave partial files.
    """
    sanitized = sanitize_float(obj)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(sanitized, f, indent=indent)
        os.replace(tmp, path)
    except Exception as e:
        # If we fail to save, try a last-resort minimal serialization
        try:
            fallback = {"error": f"Failed to json.dump: {str(e)}", "content": str(sanitized)[:10000]}
            with open(path, "w") as f:
                json.dump(fallback, f, indent=2)
        except Exception:
            # give up silently but print
            print(f"[safe_json_dump] Critical: Could not save JSON to {path}: {e}", file=sys.stderr)

# -------------------------
# API call helpers
# -------------------------
def fetch_all_models() -> bool:
    """Fetch model list from ML server (non-fatal)."""
    global ALL_AVAILABLE_MODELS
    print(f"[Optimizer] Fetching available models from {ML_SERVER_URL}/api/ml/available-models ...")
    try:
        r = requests.get(f"{ML_SERVER_URL}/api/ml/available-models", timeout=30)
        if r.status_code == 200:
            ALL_AVAILABLE_MODELS = r.json() or []
            print(f"[Optimizer] Found {len(ALL_AVAILABLE_MODELS)} models.")
            return True
        else:
            print(f"[Optimizer] Warning: model endpoint returned {r.status_code}: {r.text[:200]}")
            ALL_AVAILABLE_MODELS = []
            return True  # allow optimizer to continue with default markets
    except Exception as e:
        print(f"[Optimizer] Could not contact ML server: {e}")
        ALL_AVAILABLE_MODELS = []
        return True  # Continue (we let script use defaults)

def run_single_backtest(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper to call the ML server backtest endpoints.
    Implements the optimizer cache-ticket handshake if the server returns a ticket id.
    Raises Exceptions when the server fails.
    """
    # Determine endpoint
    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest" if config.get("strategies") else f"{ML_SERVER_URL}/api/ml/run-backtest-on"
    try:
        r = requests.post(url, json=config, timeout=600)
    except Exception as e:
        raise RuntimeError(f"HTTP error contacting ML server: {e}")

    if r.status_code != 200:
        raise RuntimeError(f"Backtest endpoint failed: {r.status_code} - {r.text[:200]}")

    results = r.json()

    # If server returns optimizer_ticket_id, wait for cache file to appear
    if isinstance(results, dict) and "optimizer_ticket_id" in results:
        ticket = results["optimizer_ticket_id"]
        cache_path = os.path.join(OPTIMIZER_CACHE_DIR, f"{ticket}.json")
        # poll for a reasonable amount of time (increase if large runs)
        timeout_seconds = 30
        poll_interval = 1.0
        waited = 0.0
        while waited < timeout_seconds:
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r") as f:
                        file_results = json.load(f)
                    # cleanup
                    try:
                        os.remove(cache_path)
                    except Exception:
                        pass
                    return sanitize_float(file_results)
                except Exception as e:
                    raise RuntimeError(f"Failed to read optimizer cache file {cache_path}: {e}")
            time.sleep(poll_interval)
            waited += poll_interval
        raise RuntimeError(f"Cache file {cache_path} did not appear after {timeout_seconds}s")

    # otherwise return directly (sanitize)
    return sanitize_float(results)

# -------------------------
# Stitching & metrics
# -------------------------
def calculate_stitched_metrics(equity_curves: List[List[Dict[str, Any]]]) -> Dict[str, float]:
    """
    Stitch multiple equity curves (list of per-fold equity records) into one continuous series
    and compute metrics. Returns a dict of stitched metrics with sanitization.
    Each equity_curve is expected to be a list of dicts like {'timestamp': ..., 'balance': ...}.
    """
    if not equity_curves:
        return {}

    all_frames = []
    last_balance = None

    for i, curve in enumerate(equity_curves):
        if not curve:
            continue
        try:
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
                # align continuous balance
                df["balance_continuous"] = df["balance"] - initial_fold_balance + (last_balance if last_balance is not None else 0.0)
                last_balance = float(df["balance_continuous"].iloc[-1])

            all_frames.append(df[["balance", "balance_continuous"]].copy())
        except Exception as exc:
            print(f"[calculate_stitched_metrics] Skipping fold {i} due to error: {exc}")

    if not all_frames:
        return {"totalTrades": 0}

    stitched = pd.concat(all_frames).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]

    equity_series = stitched["balance_continuous"].astype(float).fillna(method="ffill").fillna(0.0)
    if equity_series.empty:
        return {"totalTrades": 0}

    # Basic stats with safe guards
    try:
        initial_balance = float(equity_series.iloc[0]) if len(equity_series) > 0 else 0.0
        final_balance = float(equity_series.iloc[-1]) if len(equity_series) > 0 else 0.0

        if initial_balance <= 0 or final_balance <= 0:
            # Strategy broke or invalid series
            return {
                "StitchedCalmarRatio": -999.0,
                "StitchedSharpeRatio": -999.0,
                "StitchedMaxDrawdown": 100.0,
                "StitchedTotalReturn": -100.0,
            }

        total_return_pct = (final_balance / (initial_balance + EPSILON) - 1.0) * 100.0

        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / (peak + EPSILON)
        max_drawdown_pct = abs(float(drawdown.min()) * 100.0) if not drawdown.empty else 0.0

        daily_returns = equity_series.pct_change().fillna(0.0)
        days = max(1, (equity_series.index[-1] - equity_series.index[0]).days)
        annual_return_rate = ((final_balance / (initial_balance + EPSILON)) ** (365.25 / days)) - 1.0
        annual_std_dev = float(daily_returns.std()) * np.sqrt(365.25) if len(daily_returns) > 1 else 0.0

        if annual_std_dev <= 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (annual_return_rate - RISK_FREE_RATE) / (annual_std_dev + EPSILON)

        downside = daily_returns[daily_returns < 0]
        annual_downside_std = float(downside.std()) * np.sqrt(365.25) if len(downside) > 1 else 0.0
        sortino_ratio = (annual_return_rate - RISK_FREE_RATE) / (annual_downside_std + EPSILON) if annual_downside_std > 0 else 0.0

        calmar_ratio = (annual_return_rate * 100.0) / (max_drawdown_pct + EPSILON) if max_drawdown_pct > 0 else 0.0

    except Exception as e:
        print(f"[calculate_stitched_metrics] Metric calc error: {e}")
        return {
            "StitchedCalmarRatio": 0.0,
            "StitchedSharpeRatio": 0.0,
            "StitchedMaxDrawdown": 0.0,
            "StitchedTotalReturn": 0.0,
        }

    metrics = {
        "StitchedCalmarRatio": float(np.nan_to_num(calmar_ratio, neginf=0.0, posinf=0.0)),
        "StitchedSharpeRatio": float(np.nan_to_num(sharpe_ratio, neginf=0.0, posinf=0.0)),
        "StitchedMaxDrawdown": float(np.nan_to_num(max_drawdown_pct, neginf=100.0, posinf=100.0)),
        "StitchedTotalReturn": float(np.nan_to_num(total_return_pct, neginf=-100.0, posinf=1000000.0)),
    }

    return sanitize_float(metrics)

# -------------------------
# Objective: create objective factory
# -------------------------
def create_objective(symbol: str, timeframe: str, wfo_folds: List[Dict[str, str]]):
    """
    Returns an Optuna objective that performs full walk-forward optimization for a trial.
    """

    def objective(trial: optuna.Trial) -> Tuple[float, float]:
        test_config = deepcopy(BASE_CONFIG)
        test_config["symbol"] = symbol
        test_config["timeframe"] = timeframe

        # Search space
        test_type = trial.suggest_categorical("test_type", ["single", "combo"])
        trend_filter = trial.suggest_categorical("params.trendFilterPeriod", [0, 50, 100, 200])
        adx_filter = trial.suggest_categorical("params.minAdxLevel", [0, 20, 25])
        min_atr_filter = trial.suggest_categorical("params.minAtrPct", [0.0, 0.1, 0.2, 0.5])

        # Force ML off for baseline hunt (as in v6)
        ml_mode = trial.suggest_categorical("mlMode", ["off"])
        test_config["mlMode"] = ml_mode

        if test_type == "single":
            test_config["code"] = trial.suggest_categorical("code", ALL_TA_STRATEGIES)
        else:
            num_strats = trial.suggest_int("combo_size", 3, min(len(ALL_TA_STRATEGIES), 7))
            combo_codes = random.sample(ALL_TA_STRATEGIES, num_strats)
            trial.set_user_attr("combo_strategies", ", ".join(combo_codes))
            test_config["strategies"] = [{"code": code, "params": {}} for code in combo_codes]
            hybrid_mode = trial.suggest_categorical("params.hybridMode", ["AND"])
            test_config.setdefault("params", {})["hybridMode"] = hybrid_mode

        # Robustness upgrades
        atr_tsl = trial.suggest_float("params.tslAtrMult", 1.5, 7.0, step=0.5)
        # Explicitly disable SL/TP by leaving None
        test_config.setdefault("params", {})["SL"] = None
        test_config["params"]["TP"] = None
        test_config["params"]["tslAtrMult"] = atr_tsl

        test_config["params"]["trendFilterPeriod"] = trend_filter
        test_config["params"]["minAdxLevel"] = adx_filter
        test_config["params"]["minAtrPct"] = min_atr_filter

        # Run walk-forward folds
        all_fold_equities = []
        total_trades = 0
        for fold in wfo_folds:
            fold_cfg = deepcopy(test_config)
            fold_cfg["startDate"] = fold["startDate"]
            fold_cfg["endDate"] = fold["endDate"]

            try:
                res = run_single_backtest(fold_cfg)
                # Validate result shape
                if not isinstance(res, dict):
                    raise RuntimeError("Backtest returned non-dict result")

                metrics = res.get("metrics", {}) or {}
                n_trades = int(metrics.get("totalTrades", 0))
                if n_trades < 5:
                    # prune poorly performing or thin strategies early
                    raise optuna.TrialPruned(f"Fold failed: only {n_trades} trades")

                equity = res.get("equity") or []
                all_fold_equities.append(equity)
                total_trades += n_trades

            except optuna.TrialPruned:
                raise
            except Exception as e:
                # treat fold errors as prune signals to keep search efficient
                raise optuna.TrialPruned(f"Fold failed: {e}")

        if not all_fold_equities:
            raise optuna.TrialPruned("No successful folds")

        stitched = calculate_stitched_metrics(all_fold_equities)
        calmar = stitched.get("StitchedCalmarRatio", 0.0)
        max_dd = stitched.get("StitchedMaxDrawdown", 999.0)

        # Make sure metrics are numeric and finite
        calmar = float(np.nan_to_num(calmar, neginf=0.0, posinf=0.0))
        max_dd = float(np.nan_to_num(max_dd, neginf=100.0, posinf=100.0))

        print(f"Trial #{trial.number}: Calmar={calmar:.4f}, MaxDD={max_dd:.4f}, Trades={total_trades}")
        return (calmar, max_dd)

    return objective

# -------------------------
# Certification & Reproducibility helpers
# -------------------------
def run_certification_check(market_id: str, best_params: Dict[str, Any], report_csv_file: str, base_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Call the server-side certification endpoint with safe payload and persist the report.
    Returns the report dict or None on failure.
    """
    try:
        symbol, timeframe = market_id.split("_")
    except Exception:
        print(f"[certify] Invalid market_id format: {market_id}")
        return None

    # Build payload
    clean_params = {}
    for k, v in best_params.items():
        try:
            if isinstance(v, float) and np.isnan(v):
                continue
            clean_params[k] = v
        except Exception:
            continue

    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "best_params": clean_params,
        "base_config": base_config
    }

    print(f" - Sending best strategy for certification: {market_id}")
    try:
        r = requests.post(f"{ML_SERVER_URL}/api/ml/certify-strategy", json=payload, timeout=900)
    except Exception as e:
        print(f" - Certification HTTP error: {e}")
        return None

    if r.status_code != 200:
        print(f" - Certification failed: {r.status_code} {r.text[:200]}")
        return None

    try:
        report = r.json()
    except Exception as e:
        print(f" - Failed to parse certification response JSON: {e}")
        return None

    # Save to disk safely
    cert_path = report_csv_file.replace(".csv", "_certification.json")
    safe_json_dump(report, cert_path)
    print(f" - Certification report saved to: {cert_path}")
    return sanitize_float(report)

def run_reproducibility_check(market_id: str, best_params: Dict[str, Any]) -> bool:
    """
    Re-run the exact best trial multiple times to assert determinism.
    Returns True if reproducible, False otherwise.
    """
    symbol, timeframe = market_id.split("_")
    base_cfg = deepcopy(BASE_CONFIG)
    base_cfg["symbol"] = symbol
    base_cfg["timeframe"] = timeframe
    base_cfg["optimizer_mode"] = True

    # apply best params to base_cfg
    for key, val in best_params.items():
        if isinstance(val, float) and np.isnan(val):
            continue
        # Allow nested keys like params.tslAtrMult
        if "." in key:
            # Set nested
            parts = key.split(".")
            d = base_cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = val
        else:
            base_cfg[key] = val

    results_list = []
    wfo_folds = [
        {"startDate": "2023-01-01", "endDate": "2023-12-31"},
        {"startDate": "2024-01-01", "endDate": "2024-12-31"},
        {"startDate": "2025-01-01", "endDate": datetime.now(timezone.utc).strftime("%Y-%m-%d")},
    ]

    for i in range(REPRODUCIBILITY_RUNS):
        print(f"  - Repro run {i+1}/{REPRODUCIBILITY_RUNS}")
        try:
            all_fold_equities = []
            for fold in wfo_folds:
                cfg = deepcopy(base_cfg)
                cfg["startDate"] = fold["startDate"]
                cfg["endDate"] = fold["endDate"]
                res = run_single_backtest(cfg)
                if not isinstance(res, dict):
                    print(f"    Run {i+1}, fold failed (non-dict res)")
                    results_list.append(None)
                    break
                eq = res.get("equity")
                if not eq:
                    print(f"    Run {i+1}, fold returned no equity")
                    results_list.append(None)
                    break
                all_fold_equities.append(eq)
            else:
                stitched = calculate_stitched_metrics(all_fold_equities)
                results_list.append(stitched)
        except Exception as e:
            print(f"    Run {i+1} failed: {e}")
            results_list.append(None)
            break

    # verify
    if len(results_list) != REPRODUCIBILITY_RUNS or any(r is None for r in results_list):
        print("  - Reproducibility FAILED: missing results or errors")
        return False

    # compare numeric metrics
    base_calmar = results_list[0].get("StitchedCalmarRatio", 0.0)
    base_dd = results_list[0].get("StitchedMaxDrawdown", 0.0)

    for i, r in enumerate(results_list[1:], start=1):
        calmar_diff = abs(r.get("StitchedCalmarRatio", 0.0) - base_calmar)
        dd_diff = abs(r.get("StitchedMaxDrawdown", 0.0) - base_dd)
        print(f"    Run {i+1} vs Run1: CalmarDiff={calmar_diff:.8f}, DDDiff={dd_diff:.8f}")
        if calmar_diff > METRIC_TOLERANCE or dd_diff > METRIC_TOLERANCE:
            print("  - Reproducibility FAILED: metrics differ")
            return False

    print("  - Reproducibility PASSED")
    return True

# -------------------------
# CLI + main flow
# -------------------------
def get_user_market_selection() -> List[Tuple[str, str]]:
    """
    Determine which markets to test. Interactive selection if many found,
    otherwise default to BTC-USD / 1h.
    """
    if not ALL_AVAILABLE_MODELS:
        print("[MarketScan] No models found from server. Defaulting to BTC-USD / 1h.")
        return [("BTC-USD", "1h")]

    market_pairs = set()
    for model in ALL_AVAILABLE_MODELS:
        try:
            model_id = model.get("id", "")
            parts = model_id.split("_")
            if len(parts) >= 2:
                sym = parts[0].upper() + "-USD"
                tf = parts[1]
                market_pairs.add((sym, tf))
        except Exception:
            continue

    sorted_pairs = sorted(list(market_pairs))
    if not sorted_pairs:
        return [("BTC-USD", "1h")]

    # If interactive TTY present, allow selection, else test all
    if sys.stdin.isatty():
        print("\nAvailable Markets:")
        for i, (s, t) in enumerate(sorted_pairs):
            print(f"  [{i+1}] {s} @ {t}")
        print("  [0] Test ALL")

        while True:
            choice = input("Which market(s)? (e.g., 1,3 or 0): ").strip()
            if not choice:
                continue
            if choice == "0":
                return sorted_pairs
            try:
                picks = [int(c.strip()) for c in choice.split(",")]
                selected = []
                for p in picks:
                    if 1 <= p <= len(sorted_pairs):
                        selected.append(sorted_pairs[p - 1])
                if selected:
                    return selected
            except Exception:
                print("Invalid input. Try again.")
    else:
        # Non-interactive: test all markets
        return sorted_pairs

def main():
    parser = argparse.ArgumentParser(description="Grandmaster Optimizer (v6) - Find, Certify, and Verify Strategies.")
    parser.add_argument("--certify", action="store_true", help="Run certification for best result of each market.")
    parser.add_argument("--repro-check", action="store_true", help="Run reproducibility checks.")
    parser.add_argument("--trials", type=int, default=TOTAL_TRIALS, help="Number of trials per study.")
    parser.add_argument("--jobs", type=int, default=PARALLEL_JOBS, help="Number of parallel jobs for Optuna.")
    args = parser.parse_args()

    trials = args.trials
    jobs = args.jobs

    start_time = time.time()
    print(f"[Optimizer] Results dir: {RESULTS_DIR}")
    print(f"[Optimizer] Cache dir: {OPTIMIZER_CACHE_DIR}")

    # Fetch models (non-fatal)
    fetch_all_models()

    markets = get_user_market_selection()
    print("=" * 60)
    print(f"Starting optimization on {len(markets)} market(s): {markets}")
    print(f"Trials per study: {trials} | Parallel jobs: {jobs}")
    print(f"Certify: {'ON' if args.certify else 'OFF'} | Repro check: {'ON' if args.repro_check else 'OFF'}")
    print("=" * 60)

    for symbol, timeframe in markets:
        market_start = time.time()
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        market_id = f"{symbol}_{timeframe}"

        print(f"\n--- Starting Study: {market_id} ---")
        study_db = os.path.join(RESULTS_DIR, f"study_{market_id}_{run_stamp}.db")
        report_csv = os.path.join(RESULTS_DIR, f"report_{market_id}_{run_stamp}.csv")
        pareto_html = os.path.join(RESULTS_DIR, f"report_{market_id}_{run_stamp}_pareto.html")
        importance_html = os.path.join(RESULTS_DIR, f"report_{market_id}_{run_stamp}_importance.html")

        wfo_folds = [
            {"startDate": "2023-01-01", "endDate": "2023-12-31"},
            {"startDate": "2024-01-01", "endDate": "2024-12-31"},
            {"startDate": "2025-01-01", "endDate": datetime.now(timezone.utc).strftime("%Y-%m-%d")},
        ]

        objective = create_objective(symbol, timeframe, wfo_folds)

        study = optuna.create_study(
            study_name=f"study_{market_id}_{run_stamp}",
            storage=f"sqlite:///{study_db}",
            load_if_exists=False,
            directions=["maximize", "minimize"]
        )

        try:
            study.optimize(objective, n_trials=trials, n_jobs=jobs)
        except Exception as e:
            print(f"[Optimizer] Study optimize error: {e}")
            continue

        # Save results
        try:
            best_trials = study.best_trials
            if not best_trials:
                print(f"No complete trials for {market_id}")
                continue

            rows = []
            for t in best_trials:
                calmar, dd = (t.values if t.values else (0.0, 999.0))
                params = dict(t.params) if t.params else {}
                params["CalmarRatio"] = float(np.nan_to_num(calmar, neginf=0.0, posinf=0.0))
                params["MaxDrawdown"] = float(np.nan_to_num(dd, neginf=100.0, posinf=100.0))
                if "combo_strategies" in t.user_attrs:
                    params["combo_strategies"] = t.user_attrs["combo_strategies"]
                rows.append(params)

            df = pd.DataFrame(rows).sort_values(by="CalmarRatio", ascending=False)
            # safe save CSV (explicit conversion/copy)
            df.to_csv(report_csv, index=False, float_format="%.6f")
            print(f"Report saved: {report_csv}")
            print("Top results (head):")
            print(df.head(10).to_string(index=False))

            # Try to save optuna visuals (best-effort)
            try:
                import optuna.visualization as optv
                fig = optv.plot_pareto_front(study, target_names=["Calmar Ratio", "Max Drawdown"])
                fig.write_html(pareto_html)
                fig2 = optv.plot_param_importances(study, target=lambda t: t.values[0] if t.values else 0.0, target_name="Calmar Ratio")
                fig2.write_html(importance_html)
                print(f"Pareto/importance charts saved.")
            except Exception as e:
                print(f"Could not create visualizations: {e}")

            # Certification & Repro checks
            if df.empty:
                print("Empty report, skipping certification & reproducibility.")
                continue

            best_row = df.iloc[0].to_dict()
            cert_report = None
            cert_pass = False
            repro_pass = False

            if args.certify:
                cert_report = run_certification_check(market_id, best_row, report_csv, BASE_CONFIG)
                cert_pass = bool(cert_report and cert_report.get("certification_passed"))

            if args.repro_check:
                repro_pass = run_reproducibility_check(market_id, best_row)

            # Final grade
            final_metrics = {}
            if cert_report:
                final_metrics = cert_report.get("holdout_metrics", {})
            else:
                final_metrics = {
                    "calmar_ratio": best_row.get("CalmarRatio", 0.0),
                    "max_drawdown": best_row.get("MaxDrawdown", 100.0)
                }

            grade_results = grade_strategy_quality(final_metrics, reproducibility_pass=repro_pass, certification_pass=cert_pass)
            print("\n--- FINAL STRATEGY GRADE ---")
            print(f"Grade: {grade_results.get('grade')}")
            print(f"Reliability: {grade_results.get('reliability')}%")
            print(f"Certification: {'PASSED' if cert_pass else 'FAILED'}")
            print(f"Reproducibility: {'PASSED' if repro_pass else 'FAILED'}")

        except Exception as e:
            print(f"Error saving report or running checks for {market_id}: {e}")
            continue

        elapsed = (time.time() - market_start) / 60.0
        print(f"--- Completed study for {market_id} in {elapsed:.2f} minutes ---")

    total_elapsed = (time.time() - start_time) / 60.0
    print(f"\nAll studies completed in {total_elapsed:.2f} minutes")

if __name__ == "__main__":
    main()
