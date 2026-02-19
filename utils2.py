import math
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# Configure a module-level logger
logger = logging.getLogger("Utils")

def recursive_clean(obj):
    """
    ENGINE v7.5: JSON-Safe Recursive Sanitizer.
    Orchestrates the conversion of ML-heavy objects into standard Python types.
    Ensures NaN/Inf/Numpy types are UI-ready.
    """
    if obj is None:
        return None
    
    # 1. Handle Numeric Types (Critical for Stacking Probabilities)
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return float(obj)
    
    if isinstance(obj, (int, np.integer)):
        return int(obj)

    # 2. Handle Booleans (Numpy bools often break standard JSON encoders)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    
    # 3. Handle Dictionaries (Deep cleaning for nested strategy params)
    if isinstance(obj, dict):
        return {str(k): recursive_clean(v) for k, v in obj.items()}
    
    # 4. Handle Iterables (List, Tuple, Set, Numpy Arrays, Pandas Series)
    if isinstance(obj, (list, tuple, set, np.ndarray, pd.Series)):
        return [recursive_clean(i) for i in obj]

    # 5. Handle Timestamps (Standardize to ISO for Frontend Sync)
    if isinstance(obj, (datetime, pd.Timestamp)):
        # Ensure UTC and ISO format
        if hasattr(obj, 'tzinfo') and obj.tzinfo is None:
            obj = obj.replace(tzinfo=timezone.utc)
        return obj.isoformat()

    # 6. Fallback for custom objects (like classes)
    if hasattr(obj, '__dict__'):
        return recursive_clean(obj.__dict__)

    return str(obj) if not isinstance(obj, str) else obj

def normalize_positions(positions):
    """
    HUD ADAPTER: Standardizes live position objects for the Frontend.
    Handles the new v7.5 Probabilistic conviction metadata.
    """
    if not positions:
        return []

    out = []
    for p in positions:
        try:
            # Handle both dictionary and class-based position storage
            entry = clean_float(p.get("entry") if isinstance(p, dict) else getattr(p, 'entry', 0))
            side = str(p.get("side") if isinstance(p, dict) else getattr(p, 'side', 'long')).upper()
            qty = clean_float(p.get("size") if isinstance(p, dict) else getattr(p, 'size', 0))
            
            # New: Capture conviction metadata from the Stacking Judge
            conviction = clean_float(p.get("conviction") if isinstance(p, dict) else getattr(p, 'conviction', 0.5))
            
            out.append({
                "side": side,
                "qty": qty,
                "entry": entry,
                "time": p.get("time") if isinstance(p, dict) else getattr(p, 'time', ""),
                "stop_loss": clean_float(p.get("stop_loss", 0)),
                "conviction": f"{conviction:.2%}", # Formatted for UI display
                "current_pnl_pct": 0.0 # UI or Live Bot will update this
            })
        except Exception as e:
            logger.error(f"Failed to normalize position: {e}")
            continue
            
    return out

def generate_chart_markers(trades, active_positions):
    """
    ALIGNMENT ENGINE: Syncs trade history with Lightweight Charts.
    Formats markers for 'aboveBar' and 'belowBar' annotations.
    """
    markers = []
    
    # 1. Historical Trades (Exit Markers)
    if isinstance(trades, list):
        for t in trades:
            try:
                # Lightweight Charts requires integer Unix timestamps
                raw_time = t.get("exit_time") or t.get("time")
                dt = pd.to_datetime(raw_time)
                ts = int(dt.timestamp())
                
                side = str(t.get("side")).lower()
                profit = clean_float(t.get("profit", 0))
                
                markers.append({
                    "time": ts,
                    "position": "aboveBar" if side == "long" else "belowBar",
                    "color": "#f87171" if profit < 0 else "#34d399",
                    "shape": "arrowDown" if side == "long" else "arrowUp",
                    "text": f"{t.get('reason', 'Exit')} ({profit:+.2f})"
                })
            except: continue

    # 2. Live Positions (Entry Markers)
    if isinstance(active_positions, list):
        for p in active_positions:
            try:
                dt = pd.to_datetime(p.get("time"))
                ts = int(dt.timestamp())
                side = str(p.get("side")).upper()
                
                markers.append({
                    "time": ts,
                    "position": "belowBar" if "LONG" in side else "aboveBar",
                    "color": "#3b82f6",
                    "shape": "circle",
                    "text": f"ENTRY ({p.get('conviction', 'N/A')})"
                })
            except: continue

    return sorted(markers, key=lambda x: x["time"])

def clean_float(val):
    """Deep convert to float with strict NaN/Inf veto."""
    try:
        if val is None: return 0.0
        f_val = float(val)
        return f_val if not (math.isnan(f_val) or math.isinf(f_val)) else 0.0
    except: return 0.0
(venv) root@intelligent-mendel:~/Project/ML# 
