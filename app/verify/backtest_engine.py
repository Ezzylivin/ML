import os
import pandas as pd
import numpy as np
import tensorflow as tf
from app.predictors.model_factory import ModelFactory
from app.config2 import DATA_DIR

# 🟢 OPTIMIZATION: Silence the retracing warnings
import logging
tf.get_logger().setLevel(logging.ERROR)

def run_backtest(symbol="BTC-USD", threshold=0.55):
    print(f"\n📊 --- BACKTESTING COUNCIL: {symbol} ---")
    
    judge = ModelFactory.load_model("stacking", symbol=symbol)
    if not judge: return None

    df = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}-1h.csv"))
    
    balance = 1000.0
    position = 0
    fee_rate = 0.0006
    trades = []
    
    start_idx = 250
    total_steps = len(df) - start_idx
    print(f"  ⏳ Testing on {total_steps} bars...")

    # 🚀 OPTIMIZATION: Wrap prediction in a compiled function if possible
    # or use the __call__ method for Transformers to avoid retracing
    
    for i in range(start_idx, len(df) - 1):
        if i % 100 == 0:
            print(f"    Progress: {((i-start_idx)/total_steps)*100:.1f}%...")

        state_df = df.iloc[:i+1]
        current_price = df.iloc[i]['close']
        
        # 🎯 The "Judge" calls the Experts (including Transformer)
        prob = judge.predict_direction(state_df)
        
        if prob > threshold and position == 0:
            position = 1
            entry_price = current_price
            balance -= balance * fee_rate
            trades.append(balance)
            
        elif prob < (1 - (threshold - 0.5)) and position == 1:
            position = 0
            balance *= (current_price / entry_price)
            balance -= balance * fee_rate
            trades.append(balance)

    roi = ((balance - 1000.0) / 1000.0) * 100
    print(f"  💰 Final Balance: ${balance:.2f} | ROI: {roi:+.2f}%")
    return roi

if __name__ == "__main__":
    active_council = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "SUI-USD", "PEPE-USD"]
    for s in active_council:
        run_backtest(s)
