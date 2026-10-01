import ccxt
import pandas as pd
import pandas_ta as ta
import time

def run_parity_check():
    print("🔹 Fetching live data from Coinbase...")
    exchange = ccxt.coinbase()
    # Fetch enough data to cover the moving average periods
    ohlcv = exchange.fetch_ohlcv('BTC/USD', '1h', limit=100)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('datetime', inplace=True)
    
    # 1. Replicate Indicators exactly
    # Strategy: SMA 10 crossing SMA 50
    df.ta.sma(length=10, append=True, col_names=("fast_sma",))
    df.ta.sma(length=50, append=True, col_names=("slow_sma",))
    
    print("\n🔎 GENERATING REFERENCE SIGNALS (Strategy: SMA 10/50 Crossover)")
    print("-" * 80)
    print(f"{'Decision Time (Close)':<25} | {'Signal':<10} | {'Action Time (Open)':<25} | {'Action Price'}")
    print("-" * 80)
    
    prev_sig = 0
    trades_found = 0

    # Start loop after indicators are valid
    for i in range(50, len(df)):
        # Data for Decision (Candle i-1)
        # This effectively simulates the "Just Closed" candle
        prev_candle = df.iloc[i-1]
        
        # Data for Execution (Candle i)
        # This is the "Current Open" candle
        curr_candle = df.iloc[i]
        
        # Signal Logic
        fast = prev_candle['fast_sma']
        slow = prev_candle['slow_sma']
        
        current_sig = 0
        if fast > slow: current_sig = 1
        elif fast < slow: current_sig = -1
        
        # Did the signal CHANGE?
        if current_sig != prev_sig and current_sig != 0:
            action = "BUY" if current_sig == 1 else "SELL"
            
            # Formatted Output
            decision_time = prev_candle.name.strftime('%Y-%m-%d %H:%M')
            action_time = curr_candle.name.strftime('%Y-%m-%d %H:%M')
            price = f"${curr_candle['open']:,.2f}"
            
            print(f"{decision_time:<25} | {action:<10} | {action_time:<25} | {price}")
            trades_found += 1
            
        prev_sig = current_sig

    print("-" * 80)
    print(f"✅ Found {trades_found} hypothetical trades in the last 100 hours.")
    print("Compare these timestamps and prices EXACTLY with your Dashboard 'Trade Breakdown'.")

if __name__ == "__main__":
    run_parity_check()
