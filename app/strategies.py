import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("StrategyEngine")

# --- 🟢 1. FULL STRATEGY REGISTRY ---
STRATEGY_REGISTRY = {
    "sma_crossover": {
        "label": "SMA Crossover", "type": "trend", "needs": ["fast_sma", "slow_sma"],
        "logic": lambda r, p, opt: (1, "Golden Cross") if r.get('fast_sma') > r.get('slow_sma') and p.get('fast_sma') <= p.get('slow_sma') else (-1, "Death Cross") if r.get('fast_sma') < r.get('slow_sma') and p.get('fast_sma') >= p.get('slow_sma') else (0, "neutral")
    },
    "macd_crossover": {
        "label": "MACD Momentum", "type": "trend", "needs": ["MACD", "MACDs"],
        "logic": lambda r, p, opt: (1, "MACD Bullish") if r.get('MACD') > r.get('MACDs') and p.get('MACD') <= p.get('MACDs') else (-1, "MACD Bearish") if r.get('MACD') < r.get('MACDs') and p.get('MACD') >= p.get('MACDs') else (0, "neutral")
    },
    "rsi_threshold": {
        "label": "RSI Threshold", "type": "reversion", "needs": ["rsi"],
        "logic": lambda r, p, opt: (1, "Oversold") if r.get('rsi') < opt.get("oversold_level", 30) else (-1, "Overbought") if r.get('rsi') > opt.get("overbought_level", 70) else (0, "neutral")
    },
    "rsi_divergence": {
        "label": "RSI Divergence", "type": "reversion", "needs": ["rsi", "low", "high"],
        "logic": lambda r, p, opt: (1, "Bullish Div") if r.get('low') < p.get('low') and r.get('rsi') > p.get('rsi') else (-1, "Bearish Div") if r.get('high') > p.get('high') and r.get('rsi') < p.get('rsi') else (0, "neutral")
    },
    "bollinger_bands": {
        "label": "BB Reversion", "type": "reversion", "needs": ["BBL", "BBU", "close"],
        "logic": lambda r, p, opt: (1, "BB Lower") if r.get('close') < r.get('BBL') else (-1, "BB Upper") if r.get('close') > r.get('BBU') else (0, "neutral")
    },
    "stochastic_cross": {
        "label": "Stoch Cross", "type": "reversion", "needs": ["STOCHk", "STOCHd"],
        "logic": lambda r, p, opt: (1, "Stoch Bull") if r.get('STOCHk') > r.get('STOCHd') and p.get('STOCHk') <= p.get('STOCHd') else (-1, "Stoch Bear") if r.get('STOCHk') < r.get('STOCHd') and p.get('STOCHk') >= r.get('STOCHd') else (0, "neutral")
    },
    "atr_breakout": {
        "label": "ATR Breakout", "type": "trend", "needs": ["close", "high", "low", "ATR"],
        "logic": lambda r, p, opt: (1, "Vol Up") if r.get('close') > (p.get('high') + r.get('ATR') * 0.5) else (-1, "Vol Down") if r.get('close') < (p.get('low') - r.get('ATR') * 0.5) else (0, "neutral")
    },
    "ichimoku_cloud": {
        "label": "Ichimoku", "type": "trend", "needs": ["ISA", "ISB", "close"],
        "logic": lambda r, p, opt: (1, "Above Cloud") if r.get('close') > max(r.get('ISA'), r.get('ISB')) else (-1, "Below Cloud") if r.get('close') < min(r.get('ISA'), r.get('ISB')) else (0, "neutral")
    },
    "psar_signal": {
        "label": "Parabolic SAR", "type": "trend", "needs": ["PSAR", "close"],
        "logic": lambda r, p, opt: (1, "PSAR Long") if r.get('close') > r.get('PSAR') and p.get('close') <= p.get('PSAR') else (-1, "PSAR Short") if r.get('close') < r.get('PSAR') and p.get('close') >= p.get('PSAR') else (0, "neutral")
    },
    "obv_trend": {
        "label": "OBV Volume", "type": "trend", "needs": ["OBV", "OBV_MA"],
        "logic": lambda r, p, opt: (1, "OBV Acc") if r.get('OBV') > r.get('OBV_MA') else (-1, "OBV Dist") if r.get('OBV') < r.get('OBV_MA') else (0, "neutral")
    }
}

# --- 🟢 2. UTILITY: log_requirements ---
def log_requirements(logger, strategy_code):
    strat = STRATEGY_REGISTRY.get(strategy_code)
    if strat:
        logger.info(f"🔎 REQUIREMENTS | {strat['label']} needs → {', '.join(strat['needs'])}")
    else:
        logger.warning(f"⚠️ Strategy {strategy_code} not found.")

# --- 🟢 3. SIGNAL COMPUTATION ---
def compute_signal(df_slice, strategy_conf):
    code = strategy_conf.get("code")
    params = strategy_conf.get("params", {})
    strat = STRATEGY_REGISTRY.get(code)
    
    if not strat or len(df_slice) < 2: 
        return {"signal": 0, "reason": "Inactive/No Data"}
    
    row, prev = df_slice.iloc[-1], df_slice.iloc[-2]
    mapped_row, mapped_prev = {}, {}
    
    for need in strat["needs"]:
        match = [c for c in df_slice.columns if need.lower() in c.lower()]
        if not match: 
            return {"signal": 0, "reason": f"Missing {need}", "type": strat.get("type", "trend")}
        
        col = match[0]
        mapped_row[need] = row.get(col)
        mapped_prev[need] = prev.get(col)
    
    for k in ['close', 'open', 'high', 'low']: 
        mapped_row[k] = row.get(k, 0)
        mapped_prev[k] = prev.get(k, 0)
    
    try:
        sig, reason = strat["logic"](mapped_row, mapped_prev, params)
        return {"signal": sig, "reason": reason, "code": code, "type": strat["type"]}
    except Exception as e:
        return {"signal": 0, "reason": f"Logic Error: {str(e)}", "type": strat["type"]}

# --- 🟢 4. SIGNAL AGGREGATION ---
def check_trend_stall(df_slice):
    """EXPERT EXIT: Detects momentum roll-over using ADX Slope."""
    if len(df_slice) < 4: return False
    adx_cols = [c for c in df_slice.columns if 'adx' in c.lower()]
    if not adx_cols: return False
    adx = df_slice[adx_cols[0]]
    current_adx = adx.iloc[-1]
    prior_adx = adx.iloc[-3]
    slope = (current_adx - prior_adx) / 2
    return current_adx > 30 and slope < -0.5

def aggregate_signals(results, df_slice=None, adx_val=0, combination_rule='OR', **kwargs):
    """AGGREGATOR v100.0: Consolidated Squeeze Shield & Volume Validator"""
    min_adx = float(kwargs.get('minAdxLevel', 15))
    
    # 🛡️ 1. SQUEEZE SHIELD (Filter Sideways Noise)
    # Calibrated to 0.3% to avoid suffocating the bot
    if df_slice is not None:
        bb_u = df_slice.get('BBU', df_slice.get('BBU_20_2.0', None))
        bb_l = df_slice.get('BBL', df_slice.get('BBL_20_2.0', None))
        if bb_u is not None and bb_l is not None:
            width = (bb_u.iloc[-1] - bb_l.iloc[-1]) / df_slice['close'].iloc[-1]
            if width < 0.003: 
                return {"decision": "WAIT", "reason": "SQUEEZE: Range too narrow for fees"}

    # 🛡️ 2. THE DYNAMIC STALL EXIT
    if df_slice is not None and check_trend_stall(df_slice):
        return {"decision": "EXIT", "reason": "Trend Momentum Stall (ADX rolling over)"}

    # 🛡️ 3. THE VOLUME VALIDATOR
    # Calibrated to 1.02 (2% above average)
    vol_valid = True
    if df_slice is not None and 'volume' in df_slice.columns:
        vol_ma = df_slice['volume'].rolling(20).mean().iloc[-1]
        vol_valid = df_slice['volume'].iloc[-1] > (vol_ma * 1.02)

    # 🛡️ 4. SIGNAL FILTERING & VETO
    valid = []
    for r in results:
        if r['signal'] == 0: continue
        # Apply filters to Trend-type strategies
        if r['type'] == 'trend':
            if adx_val < min_adx: continue
            if not vol_valid: continue
        valid.append(r)
    
    if not valid:
        return {"decision": "WAIT", "reason": "No valid signals or Vetoed by Filters"}
    
    # Consenus Logic
    if combination_rule == 'AND' and len(valid) < len(results):
        return {"decision": "WAIT", "reason": f"AND Consensus failed"}
        
    net = sum(r['signal'] for r in valid)
    direction = "LONG" if net > 0 else "SHORT" if net < 0 else "WAIT"
    
    return {
        "decision": direction, 
        "reason": f"Aggregated {len(valid)} signals via {combination_rule}",
        "active_codes": [r['code'] for r in valid]
    }
