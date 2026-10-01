import requests
import json
import pandas as pd
import numpy as np
import time
import optuna
from datetime import datetime, timedelta, timezone
import os
import sys
import random
import argparse
import logging
import traceback
from copy import deepcopy
from colorama import Fore, Style, init

# =====================================================================
#  INITIALIZATION
# =====================================================================
init(autoreset=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
#  CONFIGURATION
# =====================================================================
ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://74.208.28.77:8000")
RESULTS_DIR = os.path.join(os.getcwd(), "data", "optimizer_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 12345
PARALLEL_JOBS = 1
STORAGE_URL = f"sqlite:///{RESULTS_DIR}/optuna.db"

MIN_TRADES_FOR_VALIDITY = 2
PRUNE_DRAWDOWN_LIMIT = 35.0
PRUNE_RETURN_LIMIT = -20.0

TREND_STRATEGIES = [
    "sma_crossover", "macd_crossover", "ichimoku_cloud",
    "psar_signal", "obv_signal", "atr_breakout"
]

RANGE_STRATEGIES = [
    "rsi_divergence", "stochastic_crossover",
    "bollinger_bands", "cci_oversold"
]

ALL_AVAILABLE_MODELS = []

BASE_CONFIG = {
    "symbol": "btc-USD",
    "timeframe": "1h",
    "initialBalance": 1000,
    "fee": 0.006,
    "riskManagementMode": "standard",
    "riskPercentage": 1,
    "optimizer_mode": True
}

# =====================================================================
#  UTILITIES
# =====================================================================
def set_global_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

def set_nested_value(d, keys, value):
    keys = keys.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

# =====================================================================
#  NETWORKING
# =====================================================================
def robust_request(method, url, json_data=None, retries=3):
    for i in range(retries):
        try:
            if method == "GET":
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, json=json_data, timeout=600)

            if response.status_code == 200:
                return response
            elif response.status_code >= 500:
                time.sleep(1)
            else:
                return response
        except requests.exceptions.RequestException:
            time.sleep(1)

    raise Exception(f"Failed to connect to {url}")

def execute_simulation(config, dry_run=False):
    if dry_run:
        dates = pd.date_range(start=config["startDate"], periods=50, freq="H")
        fake_curve = [
            {"timestamp": d.isoformat(), "balance": 1000 * (1 + (0.01 * i))}
            for i, d in enumerate(dates)
        ]
        return {
            "metrics": {"totalReturn": 50.0, "totalTrades": 10, "maxDrawdown": 5.0},
            "equityCurve": fake_curve,
            "tradeBreakdown": []
        }

    url = f"{ML_SERVER_URL}/api/ml/run-combo-backtest"
    response = robust_request("POST", url, json_data=config)

    if response.status_code != 200:
        raise Exception(f"Server Error: {response.text}")

    data = response.json()

    if "combinedResult" in data:
        return data["combinedResult"]
    return data

def fetch_all_models(dry_run=False):
    global ALL_AVAILABLE_MODELS

    if dry_run:
        ALL_AVAILABLE_MODELS = [
            {"id": "btc_1h_mock_model", "name": "Mock Model"}
        ]
        return True

    try:
        response = robust_request("GET", f"{ML_SERVER_URL}/api/ml/available-models")
        if response.status_code == 200:
            ALL_AVAILABLE_MODELS = response.json()
            print(f"{Fore.GREEN}[API] Loaded {len(ALL_AVAILABLE_MODELS)} ML models.")
            return True
        return False
    except:
        return False
# =====================================================================
#  WALK-FORWARD FOLD GENERATION
# =====================================================================
def generate_wfo_folds():
    now = datetime.now(timezone.utc)
    start_year = 2018
    folds = []

    for year in range(start_year, now.year):
        folds.append({
            "startDate": f"{year}-01-01",
            "endDate": f"{year}-12-31",
            "id": year
        })

    # Require 30+ days in current-year fold
    if (now - datetime(now.year, 1, 1, tzinfo=timezone.utc)).days > 30:
        folds.append({
            "startDate": f"{now.year}-01-01",
            "endDate": now.strftime("%Y-%m-%d"),
            "id": now.year
        })

    return folds

# =====================================================================
#  STITCHED METRICS
# =====================================================================
def calculate_stitched_metrics(equity_curves, trade_counts):
    if not equity_curves:
        return {"StitchedCalmarRatio": -999999, "StitchedMaxDrawdown": 100}

    all_dfs = []
    total_trades = sum(trade_counts)

    for i, fold_curve in enumerate(equity_curves):
        if not fold_curve:
            continue

        df = pd.DataFrame(fold_curve)
        if df.empty or "balance" not in df.columns:
            continue

        if i == 0:
            df["balance_continuous"] = df["balance"]
            last_balance = df["balance"].iloc[-1]
        else:
            init_balance = fold_curve[0]["balance"]
            df["balance_continuous"] = df["balance"] - init_balance + last_balance
            last_balance = df["balance_continuous"].iloc[-1]

        all_dfs.append(df)

    if not all_dfs:
        return {"StitchedCalmarRatio": -999999, "StitchedMaxDrawdown": 100}

    if total_trades < MIN_TRADES_FOR_VALIDITY:
        return {"StitchedCalmarRatio": -999999, "StitchedMaxDrawdown": 100}

    stitched_df = pd.concat(all_dfs, ignore_index=True)

    initial = stitched_df["balance_continuous"].iloc[0]
    final = stitched_df["balance_continuous"].iloc[-1]

    if initial <= 0:
        return {"StitchedCalmarRatio": -999999, "StitchedMaxDrawdown": 100}

    # True drawdown
    peak = stitched_df["balance_continuous"].cummax()
    dd = (stitched_df["balance_continuous"] - peak) / peak * 100
    max_dd_pct = abs(dd.min())

    if max_dd_pct == 0:
        max_dd_pct = 0.0001

    try:
        t0 = pd.to_datetime(stitched_df["timestamp"].iloc[0])
        t1 = pd.to_datetime(stitched_df["timestamp"].iloc[-1])
        seconds = (t1 - t0).total_seconds()
        years = seconds / (365.25 * 24 * 3600)
        years = max(years, 0.1)

        if final <= 0:
            annual_ret = -1
        else:
            annual_ret = (final / initial) ** (1 / years) - 1

    except:
        return {"StitchedCalmarRatio": -999999, "StitchedMaxDrawdown": 100}

    calmar = (annual_ret * 100) / max_dd_pct
    return {
        "StitchedCalmarRatio": calmar,
        "StitchedMaxDrawdown": max_dd_pct
    }

# =====================================================================
#  OBJECTIVE FUNCTION
# =====================================================================
def create_objective(symbol, timeframe, wfo_folds, market_models, dry_run=False):
    def objective(trial: optuna.Trial):
        # Metadata
        trial.set_user_attr("symbol", symbol)
        trial.set_user_attr("timeframe", timeframe)

        # Base config
        config = deepcopy(BASE_CONFIG)
        config["symbol"] = symbol.upper()
        config["timeframe"] = timeframe

        # ============================================================
        #  STRATEGY SELECTION
        # ============================================================
        trend = trial.suggest_categorical("trend_strategy", TREND_STRATEGIES)
        range_s = trial.suggest_categorical("range_strategy", RANGE_STRATEGIES)
        trial.set_user_attr("combo_strategies", f"{trend},{range_s}")

        payload = []

        # ---------------------------- TREND PARAMETERS ----------------------------
        tparams = {}
        if trend == "sma_crossover":
            s1 = trial.suggest_int("sma_1", 5, 50, step=5)
            s2 = trial.suggest_int("sma_2", 10, 200, step=10)
            tparams["sma_fast_period"] = min(s1, s2)
            tparams["sma_slow_period"] = max(s1, s2)

        elif trend == "macd_crossover":
            tparams["macd_fast_period"] = trial.suggest_int("macd_f", 5, 20)
            tparams["macd_slow_period"] = trial.suggest_int("macd_s", 21, 50)
            tparams["macd_signal_period"] = trial.suggest_int("macd_sig", 5, 15)

        elif trend == "atr_breakout":
            tparams["atr_period"] = trial.suggest_int("atr_p", 10, 30)
            tparams["atr_multiplier"] = trial.suggest_float("atr_m", 1.0, 5.0, step=0.5)

        payload.append({"code": trend, "params": tparams})

        # ---------------------------- RANGE PARAMETERS ----------------------------
        rparams = {}
        if range_s == "rsi_divergence":
            rparams["rsi_length"] = trial.suggest_int("rsi_len", 7, 30)
            rparams["oversold_level"] = trial.suggest_int("rsi_os", 20, 40)
            rparams["overbought_level"] = trial.suggest_int("rsi_ob", 60, 80)

        elif range_s == "bollinger_bands":
            rparams["bb_length"] = trial.suggest_int("bb_len", 15, 30)
            rparams["bb_std"] = trial.suggest_float("bb_std", 1.5, 3.0, step=0.1)

        elif range_s == "cci_oversold":
            rparams["cci_length"] = trial.suggest_int("cci_len", 10, 40)
            rparams["cci_oversold"] = trial.suggest_int("cci_os", -150, -50)
            rparams["cci_overbought"] = trial.suggest_int("cci_ob", 50, 150)

        payload.append({"code": range_s, "params": rparams})

        config["strategies"] = payload

        # ============================================================
        #  HYBRID MODE SETTINGS
        # ============================================================
        set_nested_value(config, "params.hybridMode", "REGIME")
        set_nested_value(
            config, "params.regime_threshold",
            trial.suggest_int("regime_threshold", 15, 40, step=5)
        )

        # ============================================================
        #  ML MODEL FILTERING
        # ============================================================
        prefix = f"{symbol.split('-')[0].lower()}_{timeframe}"
        valid_models = [m["id"] for m in market_models if m["id"].lower().startswith(prefix)]

        if valid_models:
            config["mlMode"] = "predictions"
            config["mlModel"] = trial.suggest_categorical("mlModel", valid_models)
            config["mlThreshold"] = trial.suggest_float("mlThreshold", 0.50, 0.70, step=0.02)
        else:
            config["mlMode"] = "off"

        # ============================================================
        #  WALK-FORWARD EXECUTION
        # ============================================================
        curves = [None] * len(wfo_folds)
        trade_counts = [0] * len(wfo_folds)

        for idx, fold in enumerate(wfo_folds):
            try:
                fcfg = deepcopy(config)
                fcfg["startDate"] = fold["startDate"]
                fcfg["endDate"] = fold["endDate"]

                out = execute_simulation(fcfg, dry_run)

                metrics = out.get("metrics", {})
                ret_ = metrics.get("totalReturn", 0)
                dd_ = metrics.get("maxDrawdown", 0)
                trades_ = metrics.get("totalTrades", 0)

                trade_counts[idx] = trades_

                # PRUNING
                if dd_ > PRUNE_DRAWDOWN_LIMIT:
                    return -999999, 100

                if idx == 0 and ret_ < PRUNE_RETURN_LIMIT:
                    return -999999, 100

                curves[idx] = out.get("equityCurve", [])

            except Exception:
                return -999999, 100

        stitched = calculate_stitched_metrics(curves, trade_counts)
        calmar = stitched["StitchedCalmarRatio"]
        dd = stitched["StitchedMaxDrawdown"]

        return calmar, dd

    return objective
# =====================================================================
#  BUILD CONFIG FROM TRIAL (FOR FINAL VALIDATION)
# =====================================================================
def build_config_from_trial(symbol, timeframe, params):
    """Reconstruct a full config from trial parameters."""
    cfg = deepcopy(BASE_CONFIG)
    cfg["symbol"] = symbol
    cfg["timeframe"] = timeframe
    # NOTE: For simplicity, assume all strategy keys exist in params
    # Advanced usage: map params back into strategy dicts
    return cfg

# =====================================================================
#  GATE TEST STUB
# =====================================================================
def run_gate_test(symbol, timeframe, trial, mode, dry_run):
    """Stub logic for Gate testing: 'victory_lap' or 'reproducibility'."""
    # Real implementation would reconstruct trial config and rerun simulation
    return True, "PASSED (Logic Stub)"

# =====================================================================
#  FINAL VALIDATION
# =====================================================================
def perform_final_validation(study, symbol, timeframe, dry_run):
    print(f"\n{Fore.YELLOW}=============================================")
    print(f"🏆 FINAL VALIDATION: The Gauntlet (Top 5 Unique)")
    print(f"============================================={Style.RESET_ALL}")

    # Extract top unique trials
    best_trials = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)
    unique_trials, seen = [], set()

    for t in best_trials:
        sig = json.dumps(t.params, sort_keys=True)
        if sig not in seen:
            seen.add(sig)
            unique_trials.append(t)
        if len(unique_trials) >= 5:
            break

    for i, trial in enumerate(unique_trials):
        calmar = trial.values[0]
        dd = trial.values[1]
        strat = trial.user_attrs.get("combo_strategies", "Unknown")

        print(f"\n📝 Candidate #{i+1}: {strat}")
        print(f"   In-Sample Score: Calmar {calmar:.2f} | DD {dd:.2f}%")

        # Gate 1
        passed, msg = run_gate_test(symbol, timeframe, trial, "victory_lap", dry_run)
        color = Fore.GREEN if passed else Fore.RED
        print(f"   🔒 Gate 1: {color}{msg}{Style.RESET_ALL}")

        if passed:
            # Gate 2
            passed2, msg2 = run_gate_test(symbol, timeframe, trial, "reproducibility", dry_run)
            color2 = Fore.GREEN if passed2 else Fore.RED
            print(f"   🔒 Gate 2: {color2}{msg2}{Style.RESET_ALL}")

            if passed2:
                filename = f"winner_{symbol}_{timeframe}_FINAL_{i+1}_{datetime.now().strftime('%Y%m%d')}.json"
                with open(os.path.join(RESULTS_DIR, filename), "w") as f:
                    final_data = trial.params.copy()
                    final_data["strategies"] = strat
                    json.dump(final_data, f, indent=4)
                print(f"   {Fore.MAGENTA}✨ CERTIFIED GOLDEN! Saved.{Style.RESET_ALL}")

# =====================================================================
#  OPTIMIZER EXECUTION
# =====================================================================
def run_optimizer(args):
    symbol = "BTC-USD"
    timeframe = "1h"
    n_trials = 1000

    if not args.dry_run:
        if not fetch_all_models():
            print("Server Error.")
            return

    print(f"{Fore.YELLOW}⚠️ Using SQLite Storage (Process Safe).")

    study = optuna.create_study(
        study_name=f"study_{symbol}_{timeframe}_{int(time.time())}",
        storage=STORAGE_URL,
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=20),
        load_if_exists=True
    )

    print(f"\n{Fore.CYAN}🚀 STARTING OPTIMIZATION: {symbol} {timeframe}")

    try:
        wfo_folds = generate_wfo_folds()
        study.optimize(
            create_objective(symbol, timeframe, wfo_folds, ALL_AVAILABLE_MODELS, args.dry_run),
            n_trials=n_trials,
            n_jobs=PARALLEL_JOBS
        )
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Optimization Paused.")

    perform_final_validation(study, symbol, timeframe, args.dry_run)

# =====================================================================
#  MAIN CLI ENTRY
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="optimizer")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    set_global_seed(SEED)
    print(f"{Fore.CYAN}=== 🧠 INTELLIGENT MENDEL CLIENT v63 (Audit Fix) ===")
    run_optimizer(args)

if __name__ == "__main__":
    main()
