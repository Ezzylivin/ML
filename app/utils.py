import math
import copy
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Configure a module-level logger
logger = logging.getLogger("Utils")

def recursive_clean(obj):
    """
    Recursively cleans objects for JSON serialization by:
    - Replacing NaN/Inf values with 0.0 for floats.
    - Converting Numpy types (int64, float32) to standard Python int/float.
    - Handling dicts, lists, tuples, sets, and Pandas Series/Arrays recursively.
    """
    if obj is None:
        return None
    
    # Handle Floats (Standard and Numpy)
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return float(obj)
    
    # Handle Integers (Standard and Numpy)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    
    # Handle Dictionaries
    if isinstance(obj, dict):
        return {k: recursive_clean(v) for k, v in obj.items()}
    
    # Handle Iterables (List, Tuple, Set)
    if isinstance(obj, (list, tuple, set)):
        return [recursive_clean(i) for i in obj]
    
    # Handle Pandas/Numpy Arrays
    if isinstance(obj, (np.ndarray, pd.Series)):
        return recursive_clean(obj.tolist())

    return obj


def clean_float(val):
    """
    Attempts to convert a value into a safe standard float:
    - Returns 0.0 for invalid values or None.
    - Replaces NaN/Inf with 0.0.
    """
    try:
        if val is None:
            return 0.0
        # Check for strings that might be empty or invalid
        if isinstance(val, str):
            val = val.strip()
            if not val: return 0.0
            
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return 0.0
        return f_val
    except (ValueError, TypeError):
        return 0.0


def normalize_positions(positions):
    """
    Standardizes positions for the Live Bot HUD.
    """
    if not positions:
        return []

    safe_positions = copy.deepcopy(positions)
    out = []
    
    for p in safe_positions:
        out.append({
            "side": p.get("side", "long"),
            "qty": clean_float(p.get("qty")),
            "entry": clean_float(p.get("entry")),
            "time": p.get("time"),
            "stop_loss": clean_float(p.get("stop_loss", 0)),
            "peak": clean_float(p.get("peak", p.get("entry"))),
        })
    return out


def generate_chart_markers(trades, active_positions):
    """
    🟢 UPDATED: Generates markers that are guaranteed to align with 
    the integer Unix timestamps in api2.py.
    """
    markers = []
    
    # 1. Process Historical Trades
    if isinstance(trades, list):
        for t in trades:
            # 🔴 CRITICAL: Convert entryTime ISO string to Unix Integer
            try:
                raw_time = t.get("entryTime") or t.get("time")
                if isinstance(raw_time, str):
                    dt = datetime.fromisoformat(raw_time.replace('Z', '+00:00'))
                    ts = int(dt.timestamp())
                else:
                    ts = int(raw_time)
                
                markers.append({
                    "time": ts,
                    "position": "belowBar" if t.get("side") == "long" else "aboveBar",
                    "color": "#10b981" if t.get("side") == "long" else "#f59e0b",
                    "shape": "arrowUp" if t.get("side") == "long" else "arrowDown",
                    "text": t.get("label", "E") # Use PnL label from api2.py
                })
            except Exception as e:
                continue

    # 2. Process Active Positions
    if isinstance(active_positions, list):
        for p in active_positions:
            try:
                raw_time = p.get("time")
                if isinstance(raw_time, str):
                    dt = datetime.fromisoformat(raw_time.replace('Z', '+00:00'))
                    ts = int(dt.timestamp())
                else:
                    ts = int(raw_time)

                markers.append({
                    "time": ts,
                    "position": "belowBar" if p.get("side") == "long" else "aboveBar",
                    "color": "#3b82f6", # Blue for live
                    "shape": "circle",
                    "text": "LIVE"
                })
            except:
                continue

    # Sort markers by time for Lightweight Charts compatibility
    return sorted(markers, key=lambda x: x["time"])
