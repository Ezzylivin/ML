# ============================================================
#  optimizer_pipeline_safe.py
#  ULTRA-SAFE OPTIMIZER (1A) + MEDIUM-FREQ PREFERENCE (2A)
#  Integrated version - now runnable and connects to ML server
#  Created/Updated: 2025-11-16 (integrated)
# ============================================================

import os
import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import argparse

# ------------------------------------------------------------
# Config: adjust to your server if needed
# ------------------------------------------------------------
ML_SERVER_URL = os.environ.get("ML_SERVER_URL", "http://127.0.0.1:8000")
DEFAULT_OUT_DIR = "/root/Project/ML/data/optimizer_results"

# ------------------------------------------------------------
# RNG: Deterministic for reproducibility (override via CLI)
# ------------------------------------------------------------
GLOBAL_SEED = 1337
np.random.seed(GLOBAL_SEED)

# ------------------------------------------------------------
# Utility: JSON-safe conversion
# ------------------------------------------------------------
def safe_json_number(x):
    if isinstance(x, (float, int, np.floating, np.integer)):
        if np.isnan(x) or np.isinf(x):
            return None
        return float(x)
    return x

def safe_json_dict(d):
    def _recurse(v):
        if isinstance(v, dict):
            return {k: _recurse(vv) for k, vv in v.items()}
        if isinstance(v, (float, int, np.floating, np.integer)):
            return safe_json_number(v)
        return v
    return _recurse(d)

# ------------------------------------------------------------
# Walk-Forward fold generator (non-overlapping)
# ------------------------------------------------------------
def generate_wfo_folds(df: pd.DataFrame, train_ratio=0.7, folds=5) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Non-overlapping WFO folds.
    Ultra-safe: avoids leakage.
    """
    length = len(df)
    if folds <= 0 or length < 10:
        return []

    fold_size = length // folds
    sequences = []

    for i in range(folds):
        start = i * fold_size
        end = (i + 1) * fold_size if (i + 1) * fold_size <= length else length
        fold_df = df.iloc[start:end]

        if len(fold_df) < 10:
            continue

        train_end = int(len(fold_df) * train_ratio)

        train = fold_df.iloc[:train_end]
        test = fold_df.iloc[train_end:]
        if len(train) > 0 and len(test) > 0:
            sequences.append((train, test))

    return sequences

# ------------------------------------------------------------
# Scoring metrics
# ------------------------------------------------------------
def compute_metrics(balance: pd.Series, trades: int) -> Dict[str, float]:
    """Stable, safe metrics."""
    if balance is None or len(balance) < 2:
        return {"calmar": 0.0, "max_dd": 0.0, "sharpe": 0.0, "trades": int(trades or 0)}

    pnl = balance.diff().fillna(0)
    # Use returns normalized by starting equity
    returns = pnl / (balance.shift(1).replace({0: np.nan}).fillna(method='bfill').abs())
    returns = returns.fillna(0)

    # Sharpe (simple)
    mean_ret = returns.mean()
    std_ret = returns.std() + 1e-12
    sharpe = mean_ret / std_ret

    # Drawdown
    cum = balance.cummax()
    drawdown = (balance - cum) / (cum + 1e-12)
    max_dd = abs(drawdown.min()) * 100.0  # percent

    # Annualized-ish proxy: scale by 365 for daily; this is a proxy and safe for comparisons
    calmar = (mean_ret * 252) / (max_dd/100.0 + 1e-12) if max_dd > 0 else 0.0

    return {
        "calmar": safe_json_number(calmar),
        "max_dd": safe_json_number(max_dd),
        "sharpe": safe_json_number(sharpe),
        "trades": int(trades or 0)
    }

# ------------------------------------------------------------
# Anti-overfitting safety gates (1A)
# ------------------------------------------------------------
def safety_checks(metrics: Dict[str, float]) -> bool:
    """
    ULTRA-SAFE PASS/FAIL RULES.
    """
    trades = metrics.get("trades", 0)
    calmar = metrics.get("calmar", 0.0)
    sharpe = metrics.get("sharpe", 0.0)
    dd = metrics.get("max_dd", 100.0)

    # 1. Medium-frequency trading filter (2A): trades per study window
    if not (40 <= trades <= 400):
        return False

    # 2. Risk-adjusted thresholds
    if sharpe < 0.25:
        return False
    if calmar < 0.10:
        return False

    # 3. Drawdown limit
    if dd > 35.0:
        return False

    return True

# ------------------------------------------------------------
# Monte Carlo stability
# ------------------------------------------------------------
def monte_carlo_stability(balance: pd.Series, seed_offset=0) -> bool:
    """
    Test simple return perturbations.
    """
    if balance is None or len(balance) < 50:
        return False

    np.random.seed(GLOBAL_SEED + seed_offset)
    returns = balance.diff().dropna().values
    baseline = float(balance.iloc[-1])
    if np.isnan(baseline) or np.isinf(baseline) or baseline <= 0:
        return False

    simulations = 50
    failures = 0

    for _ in range(simulations):
        noise = np.random.normal(1.0, 0.03, size=len(returns))
        perturbed = (returns * noise).cumsum() + float(balance.iloc[0])
        if perturbed[-1] < baseline * 0.5:
            failures += 1

    failure_rate = failures / simulations
    return failure_rate < 0.15

# ------------------------------------------------------------
# Backtester integration: call ML server if symbol/timeframe available
# expects config keys: symbol, timeframe, startDate, endDate, code or strategies, params, initialBalance, fee, mlMode, mlModel, mlThreshold
# ------------------------------------------------------------
def run_backtest_via_server(config: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
    """
    Calls the local ML server backtest endpoints. Returns a dict with keys:
      - balance: list of balances with timestamps (we convert to pd.Series)
      - trades: int
    """
    try:
        # choose endpoint
        if "strategies" in config and config.get("strategies"):
            url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
        else:
            url = f"{ML_SERVER_URL}/api/ml/run-backtest-on"

        payload = config.copy()
        # ensure optimizer_mode False when invoked here (we want immediate result)
        payload["optimizer_mode"] = False

        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        results = resp.json()

        # normalize: expect results['metrics'] and results['equity'] (list of {timestamp, balance})
        metrics = results.get("metrics", {})
        equity = results.get("equity", [])
        trades = metrics.get("totalTrades", metrics.get("trades", metrics.get("trades_executed", 0)))

        # convert equity list to pd.Series (timestamp index -> datetime)
        if isinstance(equity, list) and equity:
            try:
                df_eq = pd.DataFrame(equity)
                if "timestamp" in df_eq.columns and "balance" in df_eq.columns:
                    df_eq["timestamp"] = pd.to_datetime(df_eq["timestamp"])
                    df_eq = df_eq.set_index("timestamp").sort_index()
                    balance_series = df_eq["balance"]
                else:
                    # fallback: maybe structure is [balance1, balance2,...]
                    balance_series = pd.Series([float(x) for x in equity])
            except Exception:
                balance_series = pd.Series([float(x) if (x is not None) else 0.0 for x in equity])
        else:
            # fallback: if server returned summary only
            last_balance = metrics.get("ending_balance") or metrics.get("end_balance") or metrics.get("final_balance") or metrics.get("balance", 0)
            balance_series = pd.Series([float(last_balance)])

        return {"balance": balance_series, "trades": int(trades or 0), "raw_metrics": metrics}
    except Exception as e:
        raise RuntimeError(f"Server backtest failed: {e}")

# ------------------------------------------------------------
# Backtester wrapper (used by optimizer). It expects:
#   df: pandas DataFrame (optional) - used only for folds and to derive start/end if server used
#   strategy_code: str
#   params: dict
#   ml_model, ml_threshold
#   symbol/timeframe optionally provided in params or via global CLI
# ------------------------------------------------------------
def run_backtest(df: Optional[pd.DataFrame], strategy_code: str, params: Dict, ml_model: Optional[str], ml_threshold: float, global_symbol: Optional[str]=None, global_timeframe: Optional[str]=None) -> Dict[str, Any]:
    """
    Primary backtest function used by the optimizer pipeline.
    Preferred path: call ML server if symbol+timeframe available.
    Otherwise: raise NotImplementedError (explicit).
    """
    # Build config for server call
    config = {
        "symbol": params.get("symbol") or global_symbol,
        "timeframe": params.get("timeframe") or global_timeframe,
        "startDate": params.get("startDate"),
        "endDate": params.get("endDate"),
        "initialBalance": float(params.get("initialBalance", 1000.0)),
        "fee": float(params.get("fee", 0.001)),
        "params": params.get("strategy_params", params.get("params", {})),
        "mlMode": params.get("mlMode", "off"),
        "mlModel": ml_model,
        "mlThreshold": float(ml_threshold or 0.5),
    }

    # attach code or strategies
    if params.get("strategies"):
        config["strategies"] = params["strategies"]
    else:
        config["code"] = strategy_code

    # If symbol/timeframe missing, try extracting from df (best-effort)
    if (not config["symbol"] or not config["timeframe"]) and df is not None:
        # try to accept 'symbol' & 'timeframe' present in df.attrs or params
        if hasattr(df, "attrs") and df.attrs.get("symbol") and df.attrs.get("timeframe"):
            config["symbol"] = config["symbol"] or df.attrs.get("symbol")
            config["timeframe"] = config["timeframe"] or df.attrs.get("timeframe")

    if not config["symbol"] or not config["timeframe"]:
        raise NotImplementedError("run_backtest requires 'symbol' and 'timeframe' (either via params or CLI). "
                                  "This implementation integrates with the ML server; provide symbol/timeframe or implement a local backtester.")

    # Derive sensible start/end from df if missing
    if df is not None:
        if not config["startDate"]:
            try:
                config["startDate"] = df.index[0].strftime("%Y-%m-%d")
            except Exception:
                config["startDate"] = df.iloc[0].get("timestamp") if "timestamp" in df.columns else None
        if not config["endDate"]:
            try:
                config["endDate"] = df.index[-1].strftime("%Y-%m-%d")
            except Exception:
                config["endDate"] = df.iloc[-1].get("timestamp") if "timestamp" in df.columns else None

    # Call server path
    server_result = run_backtest_via_server(config)
    return server_result

# ------------------------------------------------------------
# Evaluate strategy
# ------------------------------------------------------------
def evaluate_strategy(df: pd.DataFrame, strategy_code: str, params: Dict, ml_model: Optional[str], ml_threshold: float, global_symbol: Optional[str]=None, global_timeframe: Optional[str]=None):
    """
    Main evaluation pipeline per strategy.
    """
    try:
        result = run_backtest(df, strategy_code, params, ml_model, ml_threshold, global_symbol, global_timeframe)
        balance = result.get("balance")
        trades = result.get("trades", 0)

        metrics = compute_metrics(balance, trades)

        # Safety gates
        if not safety_checks(metrics):
            return None

        # Monte Carlo
        if not monte_carlo_stability(balance):
            return None

        metrics_out = {
            "strategy": strategy_code,
            "params": params,
            "ml_model": ml_model,
            "ml_threshold": ml_threshold,
            "end_balance": safe_json_number(float(balance.iloc[-1]) if len(balance) > 0 else None),
            "raw_server_metrics": result.get("raw_metrics", {}),
            **metrics
        }

        return metrics_out
    except NotImplementedError as ne:
        # Re-raise to be visible
        raise
    except Exception as e:
        # swallow and treat as fail
        return None

# ------------------------------------------------------------
# Global optimizer (grid-search like; safe, deterministic)
# ------------------------------------------------------------
def optimize(df: pd.DataFrame, search_space: List[Dict], global_symbol: Optional[str], global_timeframe: Optional[str]):
    results = []
    for item in search_space:
        strategy_code = item.get("code")
        params = item.get("params", {})
        ml_model = item.get("mlModel")
        ml_threshold = float(item.get("mlThreshold", 0.5))

        # inject symbol/timeframe from CLI if provided
        if global_symbol:
            params = dict(params)
            params.setdefault("symbol", global_symbol)
        if global_timeframe:
            params = dict(params)
            params.setdefault("timeframe", global_timeframe)

        try:
            metrics = evaluate_strategy(df, strategy_code, params, ml_model, ml_threshold, global_symbol, global_timeframe)
            if metrics:
                results.append(metrics)
        except NotImplementedError as e:
            raise

    return sorted(results, key=lambda x: x.get("calmar", 0.0), reverse=True)

# ------------------------------------------------------------
# Certification test (WFO)
# ------------------------------------------------------------
def certify(df: pd.DataFrame, strategy: Dict, folds: int=4) -> bool:
    wfo_folds = generate_wfo_folds(df, train_ratio=0.7, folds=folds)
    if not wfo_folds:
        return False

    for (train, test) in wfo_folds:
        m = evaluate_strategy(test, strategy["strategy"], strategy["params"], strategy.get("ml_model"), strategy.get("ml_threshold"), strategy["params"].get("symbol"), strategy["params"].get("timeframe"))
        if not m:
            return False
    return True

# ------------------------------------------------------------
# Reproducibility check
# ------------------------------------------------------------
def reproducibility_test(df: pd.DataFrame, strategy: Dict, runs: int=5) -> bool:
    successes = 0
    for i in range(runs):
        m = evaluate_strategy(df, strategy["strategy"], strategy["params"], strategy.get("ml_model"), strategy.get("ml_threshold"), strategy["params"].get("symbol"), strategy["params"].get("timeframe"))
        if m:
            successes += 1
    return (successes / runs) >= 0.7

# ------------------------------------------------------------
# Entrypoint: run_optimizer_pipeline
# ------------------------------------------------------------
def run_optimizer_pipeline(data_file: Optional[str], search_space_file: Optional[str], out_dir: str, symbol: Optional[str], timeframe: Optional[str], certify_flag: bool, repro_flag: bool, seed: Optional[int]):
    """
    Loads data, loads search space (or uses default), runs optimizer+certify+repro.
    """
    global GLOBAL_SEED
    if seed is not None:
        GLOBAL_SEED = int(seed)
        np.random.seed(GLOBAL_SEED)

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Load data
    df = None
    if data_file:
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        df = pd.read_csv(data_file, parse_dates=True, index_col=0)
        # Allow callers to pass symbol/timeframe in df.attrs
        df.attrs.setdefault("symbol", symbol)
        df.attrs.setdefault("timeframe", timeframe)
    else:
        # If no file provided, we still proceed because server can supply data by symbol/timeframe.
        if not symbol or not timeframe:
            raise ValueError("Either a data file must be provided or both --symbol and --timeframe must be specified.")
        # Create minimal placeholder df with start/end if we need fold boundaries
        df = pd.DataFrame({"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []})
        df = df.set_index(pd.DatetimeIndex([]))
        df.attrs.setdefault("symbol", symbol)
        df.attrs.setdefault("timeframe", timeframe)

    # Load or build search space
    if search_space_file:
        if not os.path.exists(search_space_file):
            raise FileNotFoundError(f"Search space file not found: {search_space_file}")
        with open(search_space_file, "r") as f:
            search_space = json.load(f)
    else:
        # Default conservative search-space (small)
        search_space = [
            {"code": "psar_signal", "params": {"tslAtrMult": 4.0}},
            {"code": "atr_signal", "params": {"tslAtrMult": 3.5}},
            {"code": "sma_crossover", "params": {"sma_short": 20, "sma_long": 60}},
            {"code": "rsi_divergence", "params": {"rsi_length": 14, "rsi_overbought": 75, "rsi_oversold": 25}},
            {"code": "obv_signal", "params": {}},
        ]
        # attach symbol/timeframe to params for server integration
        for s in search_space:
            s.setdefault("params", {}).setdefault("symbol", symbol)
            s.setdefault("params", {}).setdefault("timeframe", timeframe)

    print("🔍 Running Ultra-Safe Strategy Optimization...")
    print(f"   Out dir: {out_dir}")
    print(f"   Using ML Server: {ML_SERVER_URL}")
    print(f"   Search space size: {len(search_space)}")
    print("   Seed:", GLOBAL_SEED)

    # Run optimization
    try:
        global_results = optimize(df, search_space, symbol, timeframe)
    except NotImplementedError as e:
        print("ERROR:", e)
        return None
    except Exception as e:
        print("Optimization runtime error:", e)
        return None

    csv_path = os.path.join(out_dir, f"results_{ts}.csv")
    pd.DataFrame(global_results).to_csv(csv_path, index=False)
    print(f"📁 Saved global results: {csv_path}")

    if not global_results:
        print("❌ No strategies survived the Ultra-Safe pipeline.")
        return None

    best = global_results[0]

    certified = False
    reproducible = False

    if certify_flag:
        print("🧪 Running Certification (WFO) ...")
        try:
            certified = certify(df, best, folds=4)
            print("   Certified:", certified)
        except Exception as e:
            print("   Certification error:", e)
            certified = False

    if repro_flag:
        print("🔁 Running Reproducibility ...")
        try:
            reproducible = reproducibility_test(df, best, runs=5)
            print("   Reproducible:", reproducible)
        except Exception as e:
            print("   Reproducibility error:", e)
            reproducible = False

    grade = "BAD"
    if certified and reproducible:
        grade = "A+"
    elif certified:
        grade = "B"
    elif reproducible:
        grade = "C"

    report = {
        "timestamp": ts,
        "best_strategy": safe_json_dict(best),
        "certified": bool(certified),
        "reproducible": bool(reproducible),
        "grade": grade
    }

    json_path = os.path.join(out_dir, f"certification_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"📑 Certification: {json_path}")
    print(f"🏆 FINAL GRADE: {grade}")

    return report

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def _cli():
    parser = argparse.ArgumentParser(description="Ultra-Safe Optimizer Pipeline (integrated)")
    parser.add_argument("--data-file", help="CSV data file (index is timestamp). If omitted, symbol/timeframe required.", default=None)
    parser.add_argument("--search-space", help="JSON file describing search space (list of dicts).", default=None)
    parser.add_argument("--out-dir", help="Output directory", default=DEFAULT_OUT_DIR)
    parser.add_argument("--symbol", help="Symbol (e.g., BTC-USD) for server-side backtests", default=None)
    parser.add_argument("--timeframe", help="Timeframe (e.g., 1h, 1d) for server-side backtests", default=None)
    parser.add_argument("--certify", help="Run certification (WFO)", action="store_true")
    parser.add_argument("--repro", help="Run reproducibility check", action="store_true")
    parser.add_argument("--seed", help="Override RNG seed", type=int, default=None)
    args = parser.parse_args()

    run_optimizer_pipeline(
        data_file=args.data_file,
        search_space_file=args.search_space,
        out_dir=args.out_dir,
        symbol=args.symbol,
        timeframe=args.timeframe,
        certify_flag=args.certify,
        repro_flag=args.repro,
        seed=args.seed
    )

if __name__ == "__main__":
    _cli()
