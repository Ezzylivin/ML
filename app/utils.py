import math
import copy
import logging
import numpy as np
import pandas as pd

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

    # Log a warning for unsupported object types but allow flow to continue
    # (Commented out to reduce noise, enable if debugging serialization issues)
    # logger.warning(f"Unsupported type {type(obj)} in recursive_clean. Returning as is.")
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
        if isinstance(val, (float, np.floating)):
            if math.isnan(val) or math.isinf(val):
                return 0.0
            return float(val)
        return float(val)
    except (ValueError, TypeError):
        # logger.warning(f"Failed to convert value '{val}' to float.")
        return 0.0


def normalize_positions(positions):
    """
    Normalizes a list of position dictionaries for API responses.
    Ensures that fields like 'peak', 'trough', and 'entry' are valid floats
    so the frontend chart doesn't crash.
    """
    if not positions:
        return []

    # Deep copy to prevent mutating the actual trading state
    safe_positions = copy.deepcopy(positions)
    out = []
    
    for p in safe_positions:
        out.append({
            "side": p.get("side", "long"),
            "qty": clean_float(p.get("qty")),
            "entry": clean_float(p.get("entry")),
            "time": p.get("time"),
            # Optional fields that might exist in advanced managers
            "peak": clean_float(p.get("peak", p.get("entry"))),
            "trough": clean_float(p.get("trough", p.get("entry"))),
            "entry_atr": clean_float(p.get("entry_atr", 0)),
            "stop_loss": clean_float(p.get("stop_loss", 0))
        })
    return out


def generate_chart_markers(trades, active_positions):
    """
    Generates visual markers for charting libraries (like Lightweight Charts).
    - Trades create 'ENTRY' and 'EXIT' markers.
    - Active positions create 'LIVE POS' markers.
    """
    if trades is not None and not isinstance(trades, list):
        logger.warning(f"Invalid type for trades: {type(trades)}. Expected list.")
        return []

    if active_positions is not None and not isinstance(active_positions, list):
        logger.warning(f"Invalid type for active_positions: {type(active_positions)}. Expected list.")
        return []

    # Limit to last 20 trades to prevent chart clutter
    safe_trades = copy.deepcopy(trades)[-20:] if trades else []
    safe_active = copy.deepcopy(active_positions) if active_positions else []
    markers = []

    # 1. Process Historical Trades
    for t in safe_trades:
        # Entry Marker
        markers.append({
            "time": t.get("entryTime") or t.get("time"), # Fallback for different naming conventions
            "position": "belowBar" if t.get("side") == "long" else "aboveBar",
            "color": "#2196F3", # Blue
            "shape": "arrowUp" if t.get("side") == "long" else "arrowDown",
            "text": f"ENTRY {t.get('side', '').upper()}"
        })
        
        # Exit Marker (if trade has exit details)
        if t.get("exitTime"):
            markers.append({
                "time": t.get("exitTime"),
                "position": "aboveBar" if t.get("side") == "long" else "belowBar",
                "color": "#E91E63", # Pink/Red
                "shape": "arrowDown" if t.get("side") == "long" else "arrowUp",
                "text": f"EXIT ({clean_float(t.get('profit', 0)):.2f})"
            })

    # 2. Process Active Positions
    for p in safe_active:
        markers.append({
            "time": p.get("time"),
            "position": "belowBar" if p.get("side") == "long" else "aboveBar",
            "color": "#00E676", # Green
            "shape": "arrowUp" if p.get("side") == "long" else "arrowDown",
            "text": "LIVE POS"
        })

    # Sort markers by time to ensure correct rendering order
    try:
        markers.sort(key=lambda x: x.get("time", ""))
    except Exception as e:
        logger.warning(f"Error sorting markers: {e}")

    return markers
