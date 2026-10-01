"""
Metrics Certification Module

Implements the "certify mode" engine.
- Splits data into In-Sample (IS) and Holdout/Out-of-Sample (OOS)
- Re-runs backtests on the holdout set
- Computes bootstrapped confidence intervals for key metrics
- Generates a final certification report
"""

import pandas as pd
import numpy as np
import logging
import joblib  
import os     
from typing import Dict, Any, Callable, List
from copy import deepcopy
from datetime import datetime # Import datetime directly

# --- Import from our new config ---
try:
    from config.constants import (
        HOLDOUT_PERCENTAGE, BOOTSTRAP_SAMPLES, 
        SEED, BOOTSTRAP_CONFIDENCE_LEVEL,
        MODEL_DIR  
    )
except ImportError:
    HOLDOUT_PERCENTAGE = 0.10
    BOOTSTRAP_SAMPLES = 1000
    SEED = 12345
    BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
    MODEL_DIR = "/root/Project/ML/app/models/" # Fallback

# --- Import the backtesting engine ---
try:
    from backtest_core import run_backtest, engineer_features_for_backtest, generate_ta_signals, find_col, EPSILON
except ImportError as e:
    logging.critical(f"[Certifier] Failed to import backtest engine: {e}")
    raise e

np.random.seed(SEED)

# --- 🚀 START: JSON CLEANING FUNCTIONS 🚀 ---

def _clean_value(value: Any) -> Any:
    """Recursively converts NaN, numpy types to JSON-safe Python types."""
    if isinstance(value, (np.floating, np.float64, float)):
        if np.isnan(value) or np.isinf(value):
            return None  # Convert NaN/Inf to None (JSON 'null')
        return float(value)
    if isinstance(value, (np.integer, np.int64, int)):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, dict):
        return _deep_clean_dict(value)
    if isinstance(value, list):
        return [_clean_value(v) for v in value]
    if isinstance(value, pd.DataFrame):
        # DataFrames should be converted *before* this point, but as a fallback.
        return value.to_dict('records')
    if isinstance(value, pd.Series):
        return value.tolist()
    
    # Default for anything else (like NoneType)
    return value

def _deep_clean_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively applies _clean_value to all items in a dict."""
    if not isinstance(d, dict):
        return _clean_value(d) # Handle case where input is not a dict
        
    clean_dict = {}
    for key, value in d.items():
        clean_dict[key] = _clean_value(value)
    return clean_dict

# --- 🚀 END: JSON CLEANING FUNCTIONS 🚀 ---


def _calculate_calmar(equity_curve: pd.Series) -> float:
    if equity_curve.empty or equity_curve.iloc[0] == 0:
        return -999.0

    initial_balance = equity_curve.iloc[0]
    final_balance = equity_curve.iloc[-1]
    if final_balance <= 0:
        return -999.0

    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / (peak + EPSILON)
    max_drawdown_pct = abs(drawdown.min() * 100)
    if max_drawdown_pct < 0.01:
        max_drawdown_pct = EPSILON

    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if days < 1:
        days = 1
    annual_return_rate = ((final_balance / initial_balance) ** (365.25 / days)) - 1

    calmar_ratio = (annual_return_rate * 100) / max_drawdown_pct
    return calmar_ratio

def _run_bootstrap_analysis(df: pd.DataFrame, backtest_func: Callable, initial_balance: float) -> Dict[str, Any]:
    if df.empty:
        return {"mean": 0, "std_error": 0, "ci_low": 0, "ci_high": 0}

    if 'balance' not in df.columns:
        logging.error("[Certifier] 'equity' column not found in DataFrame for bootstrapping.")
        return {"mean": 0, "std_error": 0, "ci_low": 0, "ci_high": 0}

    daily_returns = df['balance'].pct_change().fillna(0)
    n_days = len(daily_returns)
    metric_samples = []

    for _ in range(BOOTSTRAP_SAMPLES):
        indices = np.random.randint(0, n_days, size=n_days)
        bootstrapped_returns = daily_returns.iloc[indices]
        bootstrapped_equity = (1 + bootstrapped_returns).cumprod() * initial_balance
        bootstrapped_equity.index = pd.date_range(start=df.index.min(), periods=len(bootstrapped_equity), freq='D')
        try:
            metric = _calculate_calmar(bootstrapped_equity)
            metric_samples.append(metric)
        except Exception:
            metric_samples.append(0)

    lower_percentile = (1.0 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_low = np.percentile(metric_samples, lower_percentile * 100)
    ci_high = np.percentile(metric_samples, upper_percentile * 100)

    return {"mean": np.mean(metric_samples), "std_error": np.std(metric_samples), "ci_low": ci_low, "ci_high": ci_high}


# --- 💡 START UPGRADE: Re-architected Certification Function 💡 ---

def _generate_signals_for_cert(df: pd.DataFrame, best_params: Dict[str, Any], trend_filter_period: int) -> (pd.DataFrame, str):
    """
    Helper function to replicate the full signal generation logic from the main API.
    """
    signal_column = 'final_signal'
    
    # 1. Get common params
    ml_mode = best_params.get('mlMode', 'off')
    ml_model = best_params.get('mlModel')
    ml_threshold = best_params.get('mlThreshold', 0.65)
    hybrid_mode = best_params.get('params.hybridMode', 'AND')
    hybrid_mode_single = best_params.get('params.hybridMode_single', 'AND')

    ta_code = best_params.get('code')
    strategies_config = None # This will be built from 'combo_strategies'
    combo_strategies_str = best_params.get('combo_strategies')

    if combo_strategies_str and isinstance(combo_strategies_str, str):
        codes = [s.strip() for s in combo_strategies_str.split(',')]
        strategies_config = [{"code": code} for code in codes]

    # 2. Engineer base features
    df_feat = engineer_features_for_backtest(df, trend_filter_period)
    df_feat['high_conf_prediction'] = 0
    
    # 3. Add ML predictions if needed
    if ml_mode in ['on', 'predictions']:
        if not ml_model:
            raise ValueError("ML Model name is required for ML mode.")
        model_path = os.path.join(MODEL_DIR, f"{ml_model}.joblib")
        if not os.path.exists(model_path):
             model_path = os.path.join(MODEL_DIR, f"{ml_model}.pkl")
             if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file '{ml_model}.joblib' or '.pkl' not found.")
        
        pipeline = joblib.load(model_path)
        
        if isinstance(pipeline, dict):
            model = pipeline['model']; scaler = pipeline['scaler']
            model_feature_names = pipeline['feature_names']; class_indices = pipeline['class_indices']
            missing_cols = [col for col in model_feature_names if col not in df_feat.columns]
            if missing_cols:
                raise KeyError(f"Missing feature columns for ML model: {missing_cols}")
            df_model_features = df_feat[model_feature_names].copy().fillna(0)
            X_scaled_array = scaler.transform(df_model_features)
            if hasattr(model, 'feature_names_in_') or 'LGBM' in str(type(model)):
                X_scaled_input = pd.DataFrame(X_scaled_array, columns=model_feature_names, index=df_model_features.index)
            else:
                X_scaled_input = X_scaled_array
            probabilities = model.predict_proba(X_scaled_input)
        else: 
            try:
                model_classes = pipeline.named_steps['model'].classes_
                class_indices = {'buy': np.where(model_classes == 'buy')[0][0], 'hold': np.where(model_classes == 'hold')[0][0], 'sell': np.where(model_classes == 'sell')[0][0]}
            except Exception:
                class_indices = { 'buy': 0, 'hold': 1, 'sell': 2 }
            probabilities = pipeline.predict_proba(df_feat.fillna(0))
        
        df_feat['prob_buy'] = probabilities[:, class_indices['buy']]
        df_feat['prob_sell'] = probabilities[:, class_indices['sell']]
        df_feat['prob_hold'] = probabilities[:, class_indices['hold']]
        df_feat.loc[(df_feat['prob_buy'] > ml_threshold) & (df_feat['prob_buy'] > df_feat['prob_sell']) & (df_feat['prob_buy'] > df_feat['prob_hold']), 'high_conf_prediction'] = 1
        df_feat.loc[(df_feat['prob_sell'] > ml_threshold) & (df_feat['prob_sell'] > df_feat['prob_buy']) & (df_feat['prob_sell'] > df_feat['prob_hold']), 'high_conf_prediction'] = -1

    # 4. Generate final signal
    if ta_code: # Single Strategy
        df_feat = generate_ta_signals(df_feat, ta_code)
        if ml_mode == 'predictions':
            df_feat[signal_column] = 0
            if hybrid_mode_single == 'AND':
                df_feat.loc[(df_feat['ta_signal'] == 1) & (df_feat['high_conf_prediction'] == 1), signal_column] = 1
                df_feat.loc[(df_feat['ta_signal'] == -1) & (df_feat['high_conf_prediction'] == -1), signal_column] = -1
            else: # OR
                df_feat.loc[(df_feat['ta_signal'] == 1) | (df_feat['high_conf_prediction'] == 1), signal_column] = 1
                df_feat.loc[(df_feat['ta_signal'] == -1) | (df_feat['high_conf_prediction'] == -1), signal_column] = -1
        elif ml_mode == 'on':
            df_feat[signal_column] = df_feat['high_conf_prediction']
        else: # 'off'
            df_feat[signal_column] = df_feat['ta_signal'].fillna(0)
    
    elif strategies_config: # Combo Strategy
        signal_columns_to_combine = []
        for i, strat_config in enumerate(strategies_config):
            s_code = strat_config.get('code')
            if not s_code: continue
            signal_col = f'ta_signal_{i}'
            signal_columns_to_combine.append(signal_col)
            df_feat = generate_ta_signals(df_feat, s_code)
            df_feat[signal_col] = df_feat['ta_signal']
        
        if hybrid_mode == 'AND':
            df_feat['combined_ta'] = df_feat[signal_columns_to_combine].apply(lambda row: 1 if (row == 1).all() else (-1 if (row == -1).all() else 0), axis=1)
        else: # OR
            df_feat['combined_ta'] = df_feat[signal_columns_to_combine].apply(lambda row: 1 if (row == 1).any() else (-1 if (row == -1).any() else 0), axis=1)

        if ml_mode == 'predictions':
            df_feat[signal_column] = 0
            if hybrid_mode == 'AND':
                df_feat.loc[(df_feat['combined_ta'] == 1) & (df_feat['high_conf_prediction'] == 1), signal_column] = 1
                df_feat.loc[(df_feat['combined_ta'] == -1) & (df_feat['high_conf_prediction'] == -1), signal_column] = -1
            else: # OR
                df_feat.loc[(df_feat['combined_ta'] == 1) | (df_feat['high_conf_prediction'] == 1), signal_column] = 1
                df_feat.loc[(df_feat['combined_ta'] == -1) | (df_feat['high_conf_prediction'] == -1), signal_column] = -1
        elif ml_mode == 'on':
            df_feat[signal_column] = df_feat['high_conf_prediction']
        else: # 'off'
            df_feat[signal_column] = df_feat['combined_ta']
    
    else:
        raise ValueError("No 'code' or 'strategies' (from 'combo_strategies') found in best_params.")

    return df_feat.fillna(0), signal_column


def run_certification(
    best_params: Dict[str, Any],
    full_df: pd.DataFrame,
    base_config: Dict[str, Any]
) -> Dict[str, Any]:
    
    logging.info(f"[Certifier] Starting certification for strategy: {best_params.get('code', best_params.get('combo_strategies'))}")

    # --- 1. Build the FLAT config for run_backtest ---
    run_config = {
        "initial_balance": base_config.get('initialBalance'),
        "fee": base_config.get('fee'),
        "risk_mode": base_config.get('riskManagementMode'),
        "risk_percent": base_config.get('riskPercentage'),
        "growth_target": base_config.get('growthCapitalTarget'),
        "max_leverage": base_config.get('max_leverage', 2.0),
        "ml_mode": best_params.get('mlMode'),
        "stop_loss_pct": best_params.get('params.SL'),
        "take_profit_pct": best_params.get('params.TP'),
        "trend_filter_period": best_params.get('params.trendFilterPeriod'),
        "min_adx_level": best_params.get('params.minAdxLevel'),
        "trailing_stop_atr_mult": best_params.get('params.tslAtrMult'),
        "trailing_stop_pct": best_params.get('params.tslPct'),
        "min_atr_pct": best_params.get('params.minAtrPct') 
    }
    
    # --- 2. Split Data ---
    split_index = int(len(full_df) * (1.0 - HOLDOUT_PERCENTAGE))
    df_is_raw = full_df.iloc[:split_index].copy()
    df_oos_raw = full_df.iloc[split_index:].copy()

    logging.info(f"[Certifier] Full data: {len(full_df)} rows. IS: {len(df_is_raw)} rows. OOS: {len(df_oos_raw)} rows.")

    if df_is_raw.empty or df_oos_raw.empty:
        # --- 🚀 FIX: Clean this error report for JSON 🚀 ---
        return _deep_clean_dict({"passed": False, "error": "Data split resulted in an empty dataframe."})

    try:
        # --- 3. Generate Signals for both dataframes ---
        logging.info("[Certifier] Engineering features for full dataset...")
        
        trend_period = run_config.get('trend_filter_period')
        
        df_full_signals, signal_column = _generate_signals_for_cert(
            full_df.copy(), 
            best_params, 
            trend_period
        )
        
        df_is = df_full_signals.iloc[:split_index].copy()
        df_oos = df_full_signals.iloc[split_index:].copy()

        # --- 4. Run Backtests ---
        logging.info("[Certifier] Running In-Sample backtest...")
        is_results = run_backtest(
            df=df_is, 
            signal_column=signal_column, 
            **run_config
        )

        logging.info("[Certifier] Running Holdout (OOS) backtest...")
        oos_results = run_backtest(
            df=df_oos, 
            signal_column=signal_column, 
            **run_config
        )
        
    except Exception as e:
        logging.error(f"[Certifier] Backtest run failed during certification: {e}", exc_info=True)
        # --- 🚀 FIX: Clean this error report for JSON 🚀 ---
        return _deep_clean_dict({"passed": False, "error": f"Backtest run failed: {e}"})

    # --- 5. Process Results ---
    is_metrics = is_results.get('metrics', {})
    oos_metrics = oos_results.get('metrics', {})

    logging.info("[Certifier] Running bootstrap analysis on OOS results...")
    
    oos_equity_df = pd.DataFrame(oos_results.get('equity', []))

    if not oos_equity_df.empty:
        oos_equity_df['timestamp'] = pd.to_datetime(oos_equity_df['timestamp'])
        oos_equity_df = oos_equity_df.set_index('timestamp')

        def get_calmar(returns_df):
            # --- 🚀 FIX: Ensure 'equity' column exists ---
            if 'balance' in returns_df.columns:
                return _calculate_calmar(returns_df['balance'])
            return 0.0

        bootstrap_results = _run_bootstrap_analysis(oos_equity_df, get_calmar, initial_balance=run_config.get('initial_balance', 1000))
    else:
        bootstrap_results = {"mean": 0, "std_error": 0, "ci_low": 0, "ci_high": 0}

    is_calmar = is_metrics.get('calmarRatio', 0)
    oos_calmar = oos_metrics.get('calmarRatio', 0)

    passed_oos_positive = oos_calmar > 0
    calmar_decay = (is_calmar - oos_calmar) / (abs(is_calmar) + EPSILON)
    passed_robustness = calmar_decay < 0.5
    
    certification_passed = passed_oos_positive and passed_robustness

    logging.info(f"[Certifier] Certification complete. Passed: {certification_passed}")

    report = {
        "certification_passed": certification_passed,
        # --- 🚀 FIX: Remove best_params from the report ---
        # "best_params": best_params, # <-- This was the source of the NaN error
        # --- END FIX ---
        "in_sample_metrics": is_metrics,
        "holdout_metrics": oos_metrics,
        "robustness_checks": {
            "calmar_in_sample": is_calmar,
            "calmar_holdout": oos_calmar,
            "calmar_decay_pct": calmar_decay * 100,
            "passed_oos_positive": passed_oos_positive,
            "passed_robustness_check (decay < 50%)": passed_robustness,
        },
        "holdout_bootstrap_calmar": {
            "note": f"{BOOTSTRAP_SAMPLES} samples, {BOOTSTRAP_CONFIDENCE_LEVEL:.0%} CI",
            "mean": bootstrap_results.get('mean'),
            "std_error": bootstrap_results.get('std_error'),
            "ci_lower_bound": bootstrap_results.get('ci_low'),
            "ci_upper_bound": bootstrap_results.get('ci_high'),
        }
    }

    # --- 🚀 FIX: Deep clean the final report for JSON serialization ---
    # This converts all np.float64, np.nan, etc., to JSON-safe types
    final_report = _deep_clean_dict(report)
    
    return final_report

# --- 💡 END UPGRADE 💡 ---
