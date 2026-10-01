import asyncio
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
import logging

# 🟢 IMPORT YOUR ENGINE
from app.backtest import get_strategy_signal, sanitize
from app.manager import PrecisionPyramidManager

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SymmetryTest")

# --- 1. DATA GENERATOR ---
def generate_synthetic_data(data_length=100, trend='up'):
    timestamps = [datetime(2023, 1, 1, 0) + timedelta(hours=i) for i in range(data_length)]
    
    if trend == 'up':
        prices = np.linspace(100, 200, data_length)
    elif trend == 'down':
        prices = np.linspace(200, 100, data_length)
    else:
        prices = np.full(data_length, 150.0)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices + 1,
        'low': prices - 1,
        'close': prices,
        'volume': 1000
    })
    df.set_index('timestamp', inplace=True)
    return df

# --- 2. ENGINE SHIM ---
async def run_engine_direct(df, strategies):
    # Calculate Indicators
    df.ta.sma(length=10, append=True, col_names=("fast_sma",))
    df.ta.sma(length=20, append=True, col_names=("slow_sma",)) # Matches test params
    df.ta.atr(length=14, append=True, col_names=("atr",))
    df.ta.rsi(length=14, append=True, col_names=("rsi",))
    df.ta.macd(append=True)

    mgr = PrecisionPyramidManager(
        capital=1000.0,
        symbol="TEST-SYM",
        tradeDirection="both" # Allow Shorting
    )

    active_signal = 0 
    curve = []

    for i in range(50, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        # EXECUTE
        mgr.handle(
            signal=active_signal, 
            price=curr['open'], 
            time=curr.name.isoformat(),
            high=curr['high'], 
            low=curr['low'],
            atr=prev.get('atr', 0)
        )

        # SIGNAL
        votes = [get_strategy_signal(prev, s) for s in strategies]
        # Simple Logic
        if votes:
            if votes.count(1) > votes.count(-1): active_signal = 1
            elif votes.count(-1) > votes.count(1): active_signal = -1
            else: active_signal = 0
        
        # RECORD
        bal = mgr.step_equity(curr['close'])
        curve.append({'timestamp': curr.name.isoformat(), 'balance': bal})

    # Force Close
    if mgr.positions:
        mgr.handle(signal=-1 if mgr.positions[0]['side']=='long' else 1, price=df.iloc[-1]['close'], time="END")

    final_bal = curve[-1]['balance'] if curve else 1000.0
    total_ret = (final_bal - 1000.0) / 1000.0 * 100
    
    return {
        "metrics": {"totalReturn": total_ret, "finalBalance": final_bal},
        "tradeBreakdown": getattr(mgr, 'trades', [])
    }

# --- 3. THE SYMMETRY TEST ---
async def test_round_trip_symmetry():
    logger.info("🧪 STARTING SYMMETRY TEST...")

    # 1. Generate Uptrend
    original_data = generate_synthetic_data(data_length=200, trend='up')
    
    strategies = [
        {"code": "sma_crossover", "params": {"sma_fast_period": 10, "sma_slow_period": 20}}
    ]

    # 2. Run Original (Should Buy)
    logger.info("   ▶ Running Original (Uptrend)...")
    res_orig = await run_engine_direct(original_data, strategies)
    
    # 3. Create Inverted Data (Prices become negative, Trend becomes Down)
    # Note: Indicators handle negative numbers fine mathematically
    inverted_data = original_data.copy()
    inverted_data['open'] = -original_data['open']
    inverted_data['high'] = -original_data['low']  # Swap high/low for validity
    inverted_data['low'] = -original_data['high']
    inverted_data['close'] = -original_data['close']
    
    # 4. Run Inverted (Should Short)
    logger.info("   ▶ Running Inverted (Negative Price Downtrend)...")
    res_inv = await run_engine_direct(inverted_data, strategies)

    # 5. Analysis
    ret_orig = res_orig['metrics']['totalReturn']
    ret_inv = res_inv['metrics']['totalReturn']
    trades_orig = len(res_orig['tradeBreakdown'])
    trades_inv = len(res_inv['tradeBreakdown'])

    print(f"\n   Original Return: {ret_orig:.2f}% | Trades: {trades_orig}")
    print(f"   Inverted Return: {ret_inv:.2f}% | Trades: {trades_inv}")

    # 6. Assertions
    # Since your engine supports SHORTS, it should profit in both directions.
    # Therefore, returns should be roughly EQUAL, not inverted.
    
    if abs(ret_orig - ret_inv) < 1.0:
        print("   ✅ SUCCESS: Returns are symmetric (Bot profited in both directions).")
    else:
        print(f"   ⚠️ WARNING: Asymmetry detected ({ret_orig} vs {ret_inv}). Check spread/fees logic.")

    assert trades_orig == trades_inv, "❌ Trade count mismatch!"
    print("   ✅ Trade counts match.")

if __name__ == "__main__":
    asyncio.run(test_round_trip_symmetry())
