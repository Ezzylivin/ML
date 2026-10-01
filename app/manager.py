import logging
import pandas as pd
import numpy as np
from datetime import datetime

class PrecisionPyramidManager:
    def __init__(self, capital, symbol="BTC-USD", risk_mode='dynamic', base_risk=0.01, logger=None, **kwargs):
        """
        PRECISION PYRAMID MANAGER v100.0
        Core Logic: Inverse Volatility Sizing & Panic Regime Scaling
        """
        self.initial_capital = float(capital)
        self.current_equity = float(capital)
        self.symbol = symbol
        self.risk_mode = risk_mode 
        self.base_risk = float(base_risk) # e.g., 0.01 = 1% risk per trade
        
        # COINBASE FEE HARDENING
        self.commission = float(kwargs.get('commission', 0.006))
        self.slippage = float(kwargs.get('slippage', 0.001))
        
        # REGIME BOUNDARIES
        self.regime_params = {
            "adx_threshold": float(kwargs.get('minAdxLevel', 15)),
            "panic_vol_threshold": 3.0, # ATR as % of Price
        }
        
        self.positions = [] 
        self.trades = []
        self.equity_curve = []
        self.peak_equity = float(capital)
        self.logger = logger or logging.getLogger("Manager")

        self.equity_curve.append({
            "time": datetime.utcnow().isoformat(),
            "balance": round(self.current_equity, 2)
        })

    def handle(self, signal, price, time, **kwargs):
        """
        Dynamic Entry Logic: Calculates position size based on current ATR.
        Formula: Qty = (Equity * Risk%) / (ATR * Multiplier)
        """
        atr = float(kwargs.get('atr', 0))
        adx = float(kwargs.get('adx', 0))
        tsl_mult = float(kwargs.get('tslAtrMult', 3.0))
        min_adx = float(kwargs.get('minAdxLevel', self.regime_params["adx_threshold"]))

        if signal != 0 and not self.positions:
            # TREND FILTER
            if adx < min_adx: return 

            side = "long" if signal == 1 else "short"
            entry_price = price * (1 + self.slippage) if side == "long" else price * (1 - self.slippage)
            
            # --- DYNAMIC POSITION SIZING ---
            # Distance from entry to stop in price units
            sl_dist = atr * tsl_mult if atr > 0 else (entry_price * 0.02)
            
            # Risk Adjustment for Panic Regimes
            risk_multiplier = 1.0
            volatility_pct = (atr / entry_price) * 100 if entry_price > 0 else 0
            if volatility_pct > self.regime_params["panic_vol_threshold"]: 
                risk_multiplier = 0.5 
                self.logger.info(f"⚠️ [PANIC] Volatility {volatility_pct:.2f}% | Scaling risk to 50%")

            # Mathematical Position Sizing
            cash_to_risk = self.current_equity * (self.base_risk * risk_multiplier)
            raw_qty = cash_to_risk / sl_dist if sl_dist > 0 else 0
            
            # Apply 95% Buyer Power Cap to account for fees
            max_qty = (self.current_equity * 0.95) / entry_price
            qty = round(min(raw_qty, max_qty), 6)

            if qty > 0:
                # Deduct Entry Fee
                self.current_equity -= (entry_price * qty * self.commission)
                sl_price = entry_price - sl_dist if side == "long" else entry_price + sl_dist
                
                self.positions.append({
                    "entry": entry_price, "size": qty, "side": side, "time": time, 
                    "stop_loss": sl_price, "risk_multiplier": risk_multiplier
                })
                print(f"✅ [ENTRY] {side.upper()} {qty} @ {entry_price:.2f} | Risk Applied: {self.base_risk * risk_multiplier * 100:.2f}%")

    def check_exit(self, high, low, time):
        if not self.positions: return False
        pos = self.positions[0]
        if pos['side'] == 'long' and low <= pos['stop_loss']:
            self._close_position(pos['stop_loss'], time, "Stop Loss")
            return True
        elif pos['side'] == 'short' and high >= pos['stop_loss']:
            self._close_position(pos['stop_loss'], time, "Stop Loss")
            return True
        return False

    def _close_position(self, price, time, reason):
        if not self.positions: return
        pos = self.positions.pop(0)
        exit_price = price * (1 - self.slippage) if pos['side'] == 'long' else price * (1 + self.slippage)
        raw_pnl = (exit_price - pos['entry']) * pos['size'] if pos['side'] == 'long' else (pos['entry'] - exit_price) * pos['size']
        
        # Deduct Exit Fee
        fee = (exit_price * pos['size'] * self.commission)
        net_pnl = raw_pnl - fee
        
        self.current_equity += net_pnl
        self.trades.append({
            "entry_price": round(float(pos['entry']), 2), "exit_price": round(float(exit_price), 2),
            "profit": round(float(net_pnl), 2), "side": pos['side'], "reason": reason
        })
        print(f"🚩 [EXIT] {pos['side'].upper()} @ {exit_price:.2f} | PnL: {net_pnl:.2f} | Reason: {reason}")

    def step_equity(self, current_price, timestamp=None):
        unrealized = 0
        if self.positions:
            p = self.positions[0]
            unrealized = (current_price - p['entry']) * p['size'] if p['side'] == 'long' else (p['entry'] - current_price) * p['size']
        total_eq = self.current_equity + unrealized
        if total_eq > self.peak_equity: self.peak_equity = total_eq
        if timestamp: self.equity_curve.append({"time": str(timestamp), "balance": round(float(total_eq), 2)})
        return total_eq

    def get_results(self):
        equity_series = pd.Series([c['balance'] for c in self.equity_curve])
        profits = [t.get('profit', 0) for t in self.trades]
        wins, losses = [p for p in profits if p > 0], [p for p in profits if p < 0]
        roi = ((self.current_equity - self.initial_capital) / self.initial_capital) * 100
        
        max_dd = 0
        if not equity_series.empty:
            rolling_max = equity_series.cummax()
            max_dd = ((rolling_max - equity_series) / rolling_max).max() * 100

        return {
            "metrics": {
                "roi": round(float(roi), 2), "totalTrades": len(self.trades),
                "winRate": round((len(wins)/len(self.trades)*100), 2) if self.trades else 0,
                "profitFactor": round(sum(wins)/abs(sum(losses)), 2) if losses else 0,
                "maxDrawdown": round(float(max_dd), 2)
            },
            "trades": self.trades, "equityCurve": self.equity_curve
        }
