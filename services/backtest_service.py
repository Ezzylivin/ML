import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_squared_error
import math

def run_backtest(
    df: pd.DataFrame,
    initial_balance: float = 1000.0,
    fee: float = 0.001
) -> Dict[str, Any]:
    """
    Run a more realistic backtest using the 'predicted_target' column.

    Improvements:
    - Fixes lookahead bias by using the next candle's 'open' price for trades.
    - Correctly calculates win rate and other metrics by tracking round-trip trades.
    - Adds more detailed performance metrics like Profit Factor and total trades.
    """
    if "predicted_target" not in df.columns:
        raise ValueError("Missing 'predicted_target' in DataFrame.")
    if "open" not in df.columns:
        raise ValueError("DataFrame must contain 'open' column for realistic backtesting.")

    balance = initial_balance
    position_size = 0.0
    entry_price = 0.0
    equity_curve = [initial_balance] # Start with initial balance
    trades = []
    
    # --- Main trading loop ---
    # We loop to len(df) - 1 because we need the next candle's open to trade.
    for i in range(len(df) - 1):
        # The signal is based on the data from the current candle 'i'.
        signal = np.sign(df["predicted_target"].iloc[i])
        
        # The trade is executed at the open of the *next* candle 'i+1'.
        trade_price = df["open"].iloc[i+1]

        # --- Execute Trades ---
        # Buy Signal: If we get a buy signal and are not in a position.
        if signal > 0 and balance > 0:
            position_size = balance / trade_price
            entry_price = trade_price
            balance = 0
            # Record the buy trade
            trades.append({
                "action": "buy",
                "price": trade_price,
                "time": str(df.index[i+1]),
                "size": position_size
            })

        # Sell Signal: If we get a sell signal and are currently in a position.
        elif signal < 0 and position_size > 0:
            sell_value = position_size * trade_price
            balance = sell_value * (1 - fee)
            # Record the sell trade with profit/loss for this round trip
            pnl_pct = (trade_price / entry_price - 1) * 100
            trades.append({
                "action": "sell",
                "price": trade_price,
                "time": str(df.index[i+1]),
                "size": position_size,
                "pnl_pct": pnl_pct
            })
            position_size = 0
            entry_price = 0

        # --- Update Equity Curve ---
        # Calculate current equity at the close of candle 'i'.
        current_price = df["close"].iloc[i]
        equity = balance + position_size * current_price
        equity_curve.append(equity)

    # --- Calculate Final Metrics ---
    equity_series = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
    
    final_balance = equity_series.iloc[-1]
    total_profit_pct = (final_balance / initial_balance - 1) * 100
    
    # Calculate Drawdown
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_drawdown_pct = abs(drawdown.min() * 100)
    
    # --- Calculate Trade Statistics from round trips ---
    sell_trades = [t for t in trades if t['action'] == 'sell']
    total_trades = len(sell_trades)
    
    if total_trades > 0:
        winning_trades = [t for t in sell_trades if t['pnl_pct'] > 0]
        losing_trades = [t for t in sell_trades if t['pnl_pct'] <= 0]
        
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        
        gross_profit = sum(t['price'] * t['size'] - trades[trades.index(t)-1]['price'] * trades[trades.index(t)-1]['size'] for t in winning_trades)
        gross_loss = abs(sum(t['price'] * t['size'] - trades[trades.index(t)-1]['price'] * trades[trades.index(t)-1]['size'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        win_rate = 0
        profit_factor = 0

    # --- ML Metrics (optional) ---
    ml_metrics = None
    if "target" in df.columns:
        try:
            rmse = math.sqrt(mean_squared_error(df["target"], df["predicted_target"]))
            ml_metrics = {"rmse_overall": rmse}
        except Exception:
            pass

    return {
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "total_profit_pct": total_profit_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "trades": trades,
        "equity_curve": [{"time": str(t), "equity": v} for t, v in equity_series.items()],
        "ml_metrics": ml_metrics
    }
