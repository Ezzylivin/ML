"""
Data Validation Module

Contains all routines to validate data integrity, candle continuity,
and timezone correctness before running an optimization.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

# Import the new constants
try:
    from config.constants import DATA_MISSING_TOLERANCE, DATA_DUPLICATE_TOLERANCE
except ImportError:
    # Fallback if constants file is not found (though it should be)
    DATA_MISSING_TOLERANCE = 0.001
    DATA_DUPLICATE_TOLERANCE = 0.001

def validate_dataframe(df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
    """
    Runs a series of validation checks on the loaded OHLCV dataframe.
    
    Args:
        df (pd.DataFrame): The dataframe to check (must have datetime index).
        timeframe (str): The expected timeframe (e.g., '1h', '30m').

    Returns:
        Dict[str, Any]: A dictionary containing validation results.
    """
    
    if not isinstance(df.index, pd.DatetimeIndex):
        logging.error("[Validator] DataFrame index is not a DatetimeIndex.")
        return {"is_valid": False, "error": "Index is not a DatetimeIndex"}

    results = {
        "is_valid": True,
        "checks": {},
        "errors": []
    }

    # 1. Check: Timezone (Must be UTC)
    if df.index.tz is None or str(df.index.tz).upper() != 'UTC':
        results["is_valid"] = False
        check_name = "timezone_check"
        error_msg = f"Data is not timezone-aware (UTC). Found: {df.index.tz}"
        results["checks"][check_name] = {"passed": False, "details": error_msg}
        results["errors"].append(error_msg)
    else:
        results["checks"]["timezone_check"] = {"passed": True, "details": "Data is UTC."}

    # 2. Check: Duplicated Timestamps
    duplicates = df.index.duplicated().sum()
    total_rows = len(df)
    duplicate_pct = duplicates / total_rows if total_rows > 0 else 0
    check_name = "duplicate_check"
    
    if duplicate_pct > DATA_DUPLICATE_TOLERANCE:
        results["is_valid"] = False
        error_msg = f"Duplicate timestamps found: {duplicates} ({duplicate_pct:.2%}), which exceeds tolerance of {DATA_DUPLICATE_TOLERANCE:.2%}"
        results["checks"][check_name] = {"passed": False, "details": error_msg}
        results["errors"].append(error_msg)
    else:
        results["checks"][check_name] = {"passed": True, "details": f"Duplicates: {duplicates} ({duplicate_pct:.2%})"}

    # 3. Check: Missing Candles (Continuity)
    try:
        # Resample to the expected frequency and find where our data is 'NaN'
        expected_freq = pd.to_timedelta(timeframe)
        resampled_df = df.resample(expected_freq).first()
        
        # We only care about missing data *within* the start/end dates
        start_date, end_date = df.index.min(), df.index.max()
        resampled_df = resampled_df.loc[start_date:end_date]

        missing_candles = resampled_df['close'].isnull().sum()
        total_expected_candles = len(resampled_df)
        missing_pct = missing_candles / total_expected_candles if total_expected_candles > 0 else 0
        check_name = "continuity_check"

        if missing_pct > DATA_MISSING_TOLERANCE:
            results["is_valid"] = False
            error_msg = f"Missing candles detected: {missing_candles} of {total_expected_candles} expected ({missing_pct:.2%}), exceeding tolerance of {DATA_MISSING_TOLERANCE:.2%}"
            results["checks"][check_name] = {"passed": False, "details": error_msg}
            results["errors"].append(error_msg)
        else:
            results["checks"][check_name] = {"passed": True, "details": f"Missing: {missing_candles} ({missing_pct:.2%})"}

    except Exception as e:
        check_name = "continuity_check"
        error_msg = f"Failed to check continuity: {e}"
        results["is_valid"] = False # Fail safe
        results["checks"][check_name] = {"passed": False, "details": error_msg}
        results["errors"].append(error_msg)

    # 4. Check: Data Leakage (Simple check)
    # This checks if any feature was accidentally calculated *without* a lag.
    # We define "cheating" as any feature that is NOT a lag (doesn't end in _lagX)
    # and is NOT a standard OHLCV or base indicator column.
    
    # These are columns that are *allowed* to be non-lagged
    allowed_cols = ['open', 'high', 'low', 'close', 'volume', 'ta_signal', 'final_signal', 'prob_buy', 'prob_sell', 'prob_hold', 'high_conf_prediction']
    
    # Find all feature columns
    feature_cols = [col for col in df.columns if not any(col.upper().startswith(prefix.upper()) for prefix in allowed_cols)]
    
    # Find features that are NOT lagged
    leaking_features = []
    for col in feature_cols:
        is_lagged = False
        for i in range(1, 4): # Check for _lag1, _lag2, _lag3
            if col.endswith(f"_lag{i}"):
                is_lagged = True
                break
        
        if not is_lagged:
            # It's not lagged. Is it a base indicator (like 'RSI_14')?
            # A simple check: if it contains an underscore, it's probably a TA-Lib col.
            if '_' not in col:
                leaking_features.append(col) # e.g., a feature just named 'myfeature'
    
    check_name = "leakage_check"
    if leaking_features:
        # This is a critical failure, but we won't set is_valid=False,
        # as it's more of a warning during development.
        error_msg = f"Potential data leakage! The following features are non-lagged and are not standard OHLCV: {leaking_features}"
        results["checks"][check_name] = {"passed": False, "details": error_msg}
        # We don't add this to the main "errors" list as it's not a fatal data error
    else:
        results["checks"][check_name] = {"passed": True, "details": "No obvious non-lagged feature leakage."}

    return results
