import pandas as pd
import numpy as np
import logging
import pandas_ta as ta

logger = logging.getLogger("StrategyEngine")

# --- 🔵 1. COLUMN MAPPER (The Infrastructure Fix) ---
def map_columns(df):
    """Dynamically locates columns to prevent 'KeyError' crashes."""
    cols = df.columns
    mapping = {
        'fast_sma':  next((c for c in cols if 'SMA' in c and ('20' in c or '9' in c)), None),
        'slow_sma':  next((c for c in cols if 'SMA' in c and ('50' in c or '21' in c)), None),
        'rsi':       next((c for c in cols if 'RSI' in c), None),
        'macd':      next((c for c in cols if 'MACD_' in c and 'h' not in c and 's' not in c), None),
        'macds':     next((c for c in cols if 'MACDs' in c), None),
        'st_trend':  next((c for c in cols if 'SUPERTd' in c), None),
        'psar':      next((c for c in cols if 'PSARl' in c), None),
        'bbu':       next((c for c in cols if 'BBU' in c), None),
        'bbl':       next((c for c in cols if 'BBL' in c), None),
        'adx':       next((c for c in cols if 'ADX' in c), None)
    }
    return {k: v for k, v in mapping.items() if v is not None}

# --- 🔵 2. THE SIGNAL COUNCIL (The Logic Fix) ---
def compute_signal(df, strategy_conf):
    code = strategy_conf.get('code', '').lower()
    p = strategy_conf.get('params', {})
    m = map_columns(df)
    
    if len(df) < 30: return {"signal": 0, "reason": "Insufficient Data"}
    
    r = df.iloc[-1]      # Current Candle
    prev = df.iloc[-2]   # Previous Candle

    try:
        # SMA Crossover
        if code == "sma_crossover" and 'fast_sma' in m and 'slow_sma' in m:
            if r[m['fast_sma']] > r[m['slow_sma']] and prev[m['fast_sma']] <= prev[m['slow_sma']]:
                return {"signal": 1, "reason": "SMA Golden Cross"}
            if r[m['fast_sma']] < r[m['slow_sma']] and prev[m['fast_sma']] >= prev[m['slow_sma']]:
                return {"signal": -1, "reason": "SMA Death Cross"}

        # RSI Threshold (Trigger Logic)
        elif code == "rsi_threshold" and 'rsi' in m:
            os, ob = p.get('oversold', 30), p.get('overbought', 70)
            if r[m['rsi']] > os and prev[m['rsi']] <= os: return {"signal": 1, "reason": "RSI Reversal Long"}
            if r[m['rsi']] < ob and prev[m['rsi']] >= ob: return {"signal": -1, "reason": "RSI Reversal Short"}

        # SuperTrend
        elif code == "supertrend" and 'st_trend' in m:
            if r[m['st_trend']] == 1 and prev[m['st_trend']] == -1: return {"signal": 1, "reason": "SuperTrend Bullish"}
            if r[m['st_trend']] == -1 and prev[m['st_trend']] == 1: return {"signal": -1, "reason": "SuperTrend Bearish"}

        # Bollinger Mean Reversion
        elif code == "bollinger_bands" and 'bbu' in m and 'bbl' in m:
            if r['close'] < r[m['bbl']] and prev['close'] >= prev[m['bbl']]: return {"signal": 1, "reason": "BB Entry Long"}
            if r['close'] > r[m['bbu']] and prev['close'] <= prev[m['bbu']]: return {"signal": -1, "reason": "BB Entry Short"}

    except Exception as e:
        logger.error(f"Council Failure: {e}")
        
    return {"signal": 0, "reason": "Neutral"}

# --- 🔵 3. THE MASTER GATEKEEPER (The Execution Fix) ---
def validate_decision(ml_prob, df_slice, strategy_results, params):
    m = map_columns(df_slice)
    tech_signal = strategy_results.get('signal', 0)
    r = df_slice.iloc[-1]
    
    # 1. Consensus Gate
    decision = "WAIT"
    if ml_prob >= params.get('long_threshold', 0.65) and tech_signal == 1: decision = "LONG"
    elif ml_prob <= params.get('short_threshold', 0.35) and tech_signal == -1: decision = "SHORT"

    if decision == "WAIT": return {"decision": "WAIT", "reason": "Signal Conflict"}

    # 2. Volatility Veto
    if 'bbu' in m and 'bbl' in m:
        bandwidth = (r[m['bbu']] - r[m['bbl']]) / r['close']
        if bandwidth < 0.0035: return {"decision": "WAIT", "reason": "VETO: Squeeze"}

    # 3. Trend Regime Veto
    if 'adx' in m and r[m['adx']] < params.get('minAdxLevel', 20):
        return {"decision": "WAIT", "reason": "VETO: Low ADX (Chop)"}

    return {"decision": decision, "reason": strategy_results.get('reason'), "prob": ml_prob}
