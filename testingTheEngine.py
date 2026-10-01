import asyncio
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
import logging

# 🟢 IMPORT YOUR ACTUAL BACKTESTER
from app.backtest import execute_backtest, get_strategy_signal, sanitize
from app.manager import PrecisionPyramidManager

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SyntheticTest")

# --- 1. SYNTHETIC DATA GENERATOR ---
def generate_synthetic_data(data_length=200, trend='up'):
    timestamps = [datetime(2023, 1, 1, 0) + timedelta(hours=i) for i in range(data_length)]
    
    if trend == 'up':
        # Clean 45-degree angle up
        prices = np.linspace(100, 200, data_length)
    elif trend == 'down':
        # Clean 45-degree angle down
        prices = np.linspace(200, 100, data_length)
    elif trend == 'flat':
        # Perfect flat line
        prices = np.full(data_length, 150.0)
    else:
        raise ValueError("Trend must be 'up', 'down', or 'flat'")

    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,       
        'high': prices + 0.5, 
        'low': prices - 0.5,  
        'close': prices,      
        'volume': 1000
    })
    
    # Needs datetime index for pandas_ta
    df.set_index('timestamp', inplace=True)
    return df

# --- 2. ENGINE SHIM (Injects Data into Logic) ---
async def run_engine_with_dataframe(df, strategies):
    """
    Acts like execute_backtest but skips the CCXT fetch 
    and uses the provided DataFrame directly.
    """
    
    # A. Calculate Indicators (Standard Set)
    # We must match what backtest.py does
    df.ta.sma(length=10, append=True, col_names=("fast_sma",))
    df.ta.sma(length=50, append=True, col_names=("slow_sma",))
    df.ta.macd(append=True)
    df.ta.rsi(length=14, append=True, col_names=("rsi",))
    df.ta.atr(length=14, append=True, col_names=("atr",))

    # B. Initialize Manager
    mgr = PrecisionPyramidManager(
        capital=1000.0,
        symbol="SYNTH-USD",
        risk_mode='static',
        base_risk=0.05, # High risk for visibility
        tradeDirection='both' # Allow both sides
    )

    curve = []
    active_signal = 0 
    signals_log = []

    # C. Simulation Loop (Standard Logic)
    for i in range(50, len(df)):
        current_row = df.iloc[i]
        previous_row = df.iloc[i-1]
        current_time = current_row.name.isoformat()

        # 1. EXECUTE (at Open) using Signal from T-1
        mgr.handle(
            signal=active_signal, 
            price=current_row['open'], 
            time=current_time,
            high=current_row['high'], 
            low=current_row['low'], 
            atr=previous_row.get('atr', 0)
        )

        # 2. CALCULATE SIGNAL (using Close of T-1)
        votes = [get_strategy_signal(previous_row, s) for s in strategies]
        
        # Simple Majority Rule
        new_signal = 0
        if votes:
            if votes.count(1) > votes.count(-1): new_signal = 1
            elif votes.count(-1) > votes.count(1): new_signal = -1
        
        active_signal = new_signal
        signals_log.append(new_signal)
        
        # 3. RECORD
        bal = mgr.step_equity(current_row['close'])
        curve.append({'timestamp': current_time, 'balance': bal})

    # D. Force Close
    if mgr.positions:
        last_price = df.iloc[-1]['close']
        mgr.handle(signal=-1 if mgr.positions[0]['side']=='long' else 1, price=last_price, time="END")

    final_balance = curve[-1]['balance'] if curve else 1000.0
    total_return = (final_balance - 1000.0) / 1000.0 * 100
    
    return {
        "metrics": {"totalReturn": total_return, "finalBalance": final_balance},
        "trades": getattr(mgr, 'trades', []),
        "signals": signals_log
    }

# --- 3. TEST SUITE ---
async def test_synthetic_data():
    logger.info("🧪 STARTING SYNTHETIC DATA TESTS...")
    
    strategy_config = [{
        "code": "sma_crossover", 
        "params": {"sma_fast_period": 10, "sma_slow_period": 50}
    }]

    # --- TEST 1: UPTREND ---
    logger.info("\n📈 Running UPTREND Test...")
    up_data = generate_synthetic_data(data_length=200, trend='up')
    res_up = await run_engine_with_dataframe(up_data, strategy_config)
    
    trades_up = len(res_up['trades'])
    roi_up = res_up['metrics']['totalReturn']
    
    print(f"   Trades: {trades_up} | Return: {roi_up:.2f}%")
    
    # Assertions
    if trades_up > 0: print("   ✅ Engine correctly entered trades.")
    else: print("   ❌ FAILURE: No trades in perfect uptrend.")
    
    if roi_up > 0: print("   ✅ Engine made profit in uptrend.")
    else: print("   ❌ FAILURE: Engine lost money in perfect uptrend.")


    # --- TEST 2: DOWNTREND ---
    logger.info("\n📉 Running DOWNTREND Test...")
    down_data = generate_synthetic_data(data_length=200, trend='down')
    res_down = await run_engine_with_dataframe(down_data, strategy_config)
    
    trades_down = len(res_down['trades'])
    roi_down = res_down['metrics']['totalReturn']
    
    print(f"   Trades: {trades_down} | Return: {roi_down:.2f}%")
    
    # Assertions
    if trades_down > 0: print("   ✅ Engine correctly entered trades.")
    else: print("   ❌ FAILURE: No trades in perfect downtrend.")
    
    # Note: If shorting is enabled, ROI should be POSITIVE even in downtrend
    if roi_down > 0: print("   ✅ Engine made profit shorting downtrend.")
    else: print("   ⚠️ WARNING: ROI negative (Did it short? Or buy the dip?)")


    # --- TEST 3: FLAT ---
    logger.info("\n➖ Running FLAT Test...")
    flat_data = generate_synthetic_data(data_length=200, trend='flat')
    res_flat = await run_engine_with_dataframe(flat_data, strategy_config)
    
    trades_flat = len(res_flat['trades'])
    print(f"   Trades: {trades_flat}")
    
    if trades_flat == 0: print("   ✅ Engine correctly stayed out of flat market.")
    else: print("   ❌ FAILURE: Engine traded noise.")

if __name__ == "__main__":
    asyncio.run(test_synthetic_data())
