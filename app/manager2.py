import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone

class PrecisionPyramidManager:
    def __init__(self, capital, symbol="BTC-USD", risk_mode='dynamic', base_risk=0.01, logger=None, **kwargs):
        self.initial_capital = float(capital)
        self.current_equity = float(capital)
        self.symbol = symbol
        self.risk_mode = risk_mode 
        self.base_risk = float(base_risk)
        self.commission = float(kwargs.get('commission', 0.006)) 
        self.slippage = float(kwargs.get('slippage', 0.001))   
        self.long_threshold = float(kwargs.get('long_threshold', 0.65))
        self.short_threshold = float(kwargs.get('short_threshold', 0.35))
        self.tsl_mult = float(kwargs.get('tslAtrMult', 3.0)) 
        self.positions, self.trades, self.equity_curve = [], [], []
        self.equity_curve.append({"time": datetime.now(timezone.utc).isoformat(), "balance": round(self.current_equity, 2)})

    def handle(self, signal, price, time, signal_prob=0.5, **kwargs):
        atr = float(kwargs.get('atr', 0))
        if signal != 0 and not self.positions:
            side = "long" if signal == 1 else "short"
            entry_price = price * (1 + self.slippage) if side == "long" else price * (1 - self.slippage)
            sl_dist = atr * self.tsl_mult if atr > 0 else (entry_price * 0.02)
            dollar_risk = self.current_equity * self.base_risk
            qty = round(dollar_risk / sl_dist, 8) if sl_dist > 0 else 0
            final_qty = min(qty, (self.current_equity * 0.90) / entry_price)
            if final_qty > 0:
                self.current_equity -= (entry_price * final_qty * self.commission)
                self.positions.append({
                    "entry": entry_price, "size": final_qty, "side": side, "time": time, 
                    "stop_loss": entry_price - sl_dist if side == "long" else entry_price + sl_dist,
                    "best_price": entry_price, "atr_dist": sl_dist,
                    "take_profit": entry_price + (sl_dist * 5) if side == "long" else entry_price - (sl_dist * 5)
                })

    def check_exit(self, high, low, time):
        if not self.positions: return
        pos = self.positions[0]
        side, exit_trigger, exit_price, reason = pos['side'], False, 0, ""
        if side == "long":
            if high > pos['best_price']:
                pos['best_price'] = high
                new_sl = high - pos['atr_dist']
                if new_sl > pos['stop_loss']: pos['stop_loss'] = new_sl
            if low <= pos['stop_loss']: exit_trigger, exit_price, reason = True, pos['stop_loss'], "Trailing Stop"
            elif high >= pos['take_profit']: exit_trigger, exit_price, reason = True, pos['take_profit'], "Take Profit"
        else:
            if low < pos['best_price']:
                pos['best_price'] = low
                new_sl = low + pos['atr_dist']
                if new_sl < pos['stop_loss']: pos['stop_loss'] = new_sl
            if high >= pos['stop_loss']: exit_trigger, exit_price, reason = True, pos['stop_loss'], "Trailing Stop"
            elif low <= pos['take_profit']: exit_trigger, exit_price, reason = True, pos['take_profit'], "Take Profit"
        if exit_trigger: self._close_position(exit_price, time, reason)

    def _close_position(self, price, time, reason):
        if not self.positions: return
        pos = self.positions.pop(0)
        exit_price = price * (1 - self.slippage) if pos['side'] == 'long' else price * (1 + self.slippage)
        pnl = (exit_price - pos['entry']) * pos['size'] if pos['side'] == 'long' else (pos['entry'] - exit_price) * pos['size']
        net_pnl = pnl - (exit_price * pos['size'] * self.commission)
        self.current_equity += net_pnl
        self.trades.append({"entry_time": pos['time'], "exit_time": time, "side": pos['side'], "entry": pos['entry'], "exit": exit_price, "profit": round(net_pnl, 2), "reason": reason, "net_roi": (net_pnl / self.initial_capital) * 100})

    def step_equity(self, price, timestamp=None):
        unrealized = 0
        if self.positions:
            p = self.positions[0]
            unrealized = (price - p['entry']) * p['size'] if p['side'] == 'long' else (p['entry'] - price) * p['size']
        if timestamp: self.equity_curve.append({"time": str(timestamp), "balance": round(float(self.current_equity + unrealized), 2)})
        return self.current_equity + unrealized

# Refined snippet for app/manager2.py

    def get_results(self):
        if not self.trades:
            return {
                "metrics": {"roi": 0, "maxDrawdown": 0, "totalTrades": 0, "winRate": 0, "calmarRatio": 0},
                "trades": [],
                "equityCurve": self.equity_curve,
                "status": "success"
            }

        profits = [t['profit'] for t in self.trades]
        roi = ((self.current_equity - self.initial_capital) / self.initial_capital) * 100
        df_eq = pd.Series([c['balance'] for c in self.equity_curve])
        
        # Calculate Max Drawdown safely
        cum_max = df_eq.cummax()
        drawdown = (cum_max - df_eq) / cum_max
        max_dd = drawdown.max() * 100
        
        # Calculate Calmar Ratio for the Evolver
        # Calmar = Annualized Return / Max Drawdown
        # We use a simplified version for the fitness score:
        calmar = (roi / max_dd) if max_dd > 0 else (roi / 0.1)

        return {
            "metrics": {
                "roi": round(roi, 2), 
                "maxDrawdown": round(max_dd, 2), 
                "totalTrades": len(self.trades), 
                "winRate": round((len([p for p in profits if p > 0]) / len(self.trades) * 100), 2),
                "calmarRatio": round(calmar, 2)
            }, 
            "trades": self.trades, 
            "equityCurve": self.equity_curve, 
            "status": "success"
        }
