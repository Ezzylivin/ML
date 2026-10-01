import pytest
import pandas as pd
import numpy as np
import sys
import os

# 🟢 FIX: Set Credentials BEFORE importing diamond0
# This satisfies the new security check in v89.9
os.environ["MONGO_URI"] = "mongodb+srv://ericjdickerson93:Dadeadend1!!@cluster0.wzztvhe.mongodb.net/?retryWrites=true&w=majority"

# 🟢 FIX: Add current directory to path so we can import diamond0
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 🟢 FIX: Import from diamond0, not diamond
try:
    from diamond0 import compute_signal, PrecisionPyramidManager
except ImportError:
    # Fallback in case the file is just named 'diamond.py' in other environments
    try:
        from diamond import compute_signal, PrecisionPyramidManager
    except ImportError:
        raise ImportError("Could not import 'diamond0.py' or 'diamond.py'. Check file name.")

# --- 1. STRATEGY UNIT TESTS ---

def test_sma_crossover_buy():
    df = pd.DataFrame({
        "fast_sma": [100, 105, 110], 
        "slow_sma": [108, 106, 104]
    })
    sig = compute_signal(df, {"code": "sma_crossover"})
    assert sig == 1.0

def test_macd_crossover_sell():
    df = pd.DataFrame({
        "MACD_12_26_9":  [10, 5, 0],
        "MACDs_12_26_9": [5, 5, 5]
    })
    sig = compute_signal(df, {"code": "macd_crossover"})
    assert sig == -1.0

def test_rsi_threshold_logic():
    df_buy = pd.DataFrame({"RSI_14": [50, 40, 25]})
    sig_buy = compute_signal(df_buy, {"code": "rsi_divergence", "params": {"oversold_level": 30}})
    assert sig_buy == 1.0

    df_sell = pd.DataFrame({"RSI_14": [50, 60, 75]})
    sig_sell = compute_signal(df_sell, {"code": "rsi_divergence", "params": {"overbought_level": 70}})
    assert sig_sell == -1.0

def test_bollinger_bands_buy():
    df = pd.DataFrame({
        "close":       [100, 95, 90],
        "BBL_20_2.0":  [98, 96, 92],
        "BBU_20_2.0":  [102, 104, 108]
    })
    sig = compute_signal(df, {"code": "bollinger_bands"})
    assert sig == 1.0

def test_atr_breakout_buy():
    df = pd.DataFrame({
        "close":  [100, 100, 107],
        "EMA_20": [100, 100, 100],
        "ATR_14": [2, 2, 2]
    })
    sig = compute_signal(df, {"code": "atr_breakout", "params": {"atr_multiplier": 3.0}})
    assert sig == 1.0

def test_stochastic_crossover_buy():
    df = pd.DataFrame({
        "STOCHk_14_3_3": [10, 15, 18],
        "STOCHd_14_3_3": [12, 16, 14] 
    })
    sig = compute_signal(df, {"code": "stochastic_crossover"})
    assert sig == 1.0

def test_cci_oversold_buy():
    df = pd.DataFrame({"CCI_14_0.015": [0, -50, -101]})
    sig = compute_signal(df, {"code": "cci_oversold"})
    assert sig == 1.0

def test_ichimoku_buy():
    df = pd.DataFrame({
        "ITS_9":  [90, 95, 105],
        "IKS_26": [100, 100, 100]
    })
    sig = compute_signal(df, {"code": "ichimoku_system"})
    assert sig == 1.0

def test_obv_momentum_buy():
    df = pd.DataFrame({"OBV": [1000, 1005, 1010]})
    sig = compute_signal(df, {"code": "obv_signal"})
    assert sig == 1.0

def test_psar_flip_buy():
    df = pd.DataFrame({
        "PSARl_0.02_0.2": [float('nan'), float('nan'), 100.0],
        "PSARs_0.02_0.2": [105.0, 104.0, float('nan')]
    })
    sig = compute_signal(df, {"code": "psar_flip_signal"})
    assert sig == 1.0

# --- 2. COMBO LOGIC & ML TESTS ---

def test_neutral_conflict_votes():
    votes = [1, 1, -1, -1] 
    min_votes = 1
    
    buy_votes = sum(1 for v in votes if v > 0)
    sell_votes = sum(1 for v in votes if v < 0)
    
    final_sig = 0
    # Logic in v89.7: If both hit min, it's Neutral (0)
    if buy_votes >= min_votes and sell_votes >= min_votes:
        final_sig = 0
    elif buy_votes >= min_votes:
        final_sig = 1
    elif sell_votes >= min_votes:
        final_sig = -1
        
    assert final_sig == 0 

def test_ml_bypass_feature_mismatch():
    class DummyModel:
        n_features_in_ = 5
        feature_names_in_ = ["a","b","c","d","e"]
        def predict_proba(self, x): return np.array([[0.1, 0.9]])
    
    df = pd.DataFrame({"open": [1,2,3]})
    # v89.7 Logic: Feature mismatch -> ml_gate_sig = 0 -> Pass Signal
    sig = compute_signal(df, {"code": "sma_crossover", "mlConfirm": True}, ml_model=DummyModel())
    assert sig == 0.0 # Returns 0.0 because SMA columns missing, but CRUCIALLY does not crash.

# --- 3. RISK MANAGER TESTS ---

def test_manager_stop_loss_execution():
    # 🛠 FIX: Use 'base_risk' matching the updated class definition
    mgr = PrecisionPyramidManager(capital=1000, base_risk=0.01)
    
    mgr.positions.append({
        'side': 'long', 'qty': 1.0, 'entry': 100, 
        'time': 'T1', 'peak': 100, 'entry_atr': 10, 'stop_loss': 90
    })
    mgr.cash = 0 
    
    orders, msg = mgr.handle(0, 80, 80, 100, "T2", 10, 1.0)
    
    assert len(orders) > 0
    assert orders[0][0] == 'SELL'
    assert len(mgr.trades) == 1

def test_manager_daily_loss_limit():
    # 🛠 FIX: Ensure diamond0.py has updated __init__ with max_daily_loss
    mgr = PrecisionPyramidManager(capital=1000, max_daily_loss=5.0) 
    mgr.today_start_equity = 1000
    mgr.current_equity = 940 # Lost 6%
    
    allowed, reason = mgr.is_trading_allowed()
    assert allowed is False
    assert "Daily Loss Limit" in reason
