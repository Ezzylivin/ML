from datetime import datetime, timezone
import math

class PrecisionPyramidManager:
    def __init__(self, capital, fee=0.0006, max_layers=3, base_risk=0.01, risk_mode="static", max_daily_loss=5.0, max_trades_per_day=20):
        self.initial_capital = float(capital)
        self.cash = float(capital)
        self.current_equity = float(capital)
        self.fee = fee
        self.max_layers = max_layers
        self.base_risk = base_risk
        self.risk_mode = risk_mode
        self.max_daily_loss = max_daily_loss
        self.max_trades_per_day = max_trades_per_day
        
        self.positions = []
        self.trades = []  # Holds completed trades.
        self.stop_threshold = 1e-2 

        # Daily tracking reset variables
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.last_trade_date = None

    def _reset_daily_metrics_if_new_day(self, current_time_str):
        """Resets daily counters if the date has changed."""
        try:
            # Handle ISO format string or datetime object
            if isinstance(current_time_str, str):
                current_date = datetime.fromisoformat(current_time_str).date()
            else:
                current_date = current_time_str.date()

            if self.last_trade_date != current_date:
                self.daily_pnl = 0.0
                self.trades_today = 0
                self.last_trade_date = current_date
        except ValueError:
            pass # Keep going if date parsing fails

    def handle(self, signal, price, low, high, time, atr=0, tsl_mult=3.0):
        """
        Main logic loop called on every tick/candle.
        1. Checks stop losses on existing positions.
        2. Executes new trades based on signal.
        """
        self._reset_daily_metrics_if_new_day(time)
        messages = "OK"
        executed_orders = []

        # --- 1. Check Stops & Take Profits on Existing Positions ---
        active_positions = []
        for pos in self.positions:
            # Determine exit condition
            exit_price = None
            pnl = 0.0
            
            if pos['side'] == 'long':
                if low <= pos['stop_loss']:
                    exit_price = pos['stop_loss']
                # (Optional) Implement Trailing Stop updates here if needed
            elif pos['side'] == 'short':
                if high >= pos['stop_loss']:
                    exit_price = pos['stop_loss']

            if exit_price:
                # Execute Exit
                if pos['side'] == 'long':
                    pnl = (exit_price - pos['entry']) * pos['qty']
                else:
                    pnl = (pos['entry'] - exit_price) * pos['qty']
                
                # Apply Fees (Exit)
                fee_cost = (exit_price * pos['qty']) * self.fee
                pnl -= fee_cost
                self.cash += (exit_price * pos['qty']) if pos['side'] == 'long' else (2 * pos['entry'] * pos['qty']) - (exit_price * pos['qty']) # Simplified cash return logic
                self.cash -= fee_cost # Deduct fee from cash

                self.trades.append({
                    'entry_time': pos['time'],
                    'exit_time': time,
                    'side': pos['side'],
                    'entry_price': pos['entry'],
                    'exit_price': exit_price,
                    'qty': pos['qty'],
                    'pnl': pnl,
                    'reason': 'Stop Loss'
                })
                
                self.daily_pnl += pnl
                executed_orders.append(f"STOP {pos['side']} @ {exit_price}")
            else:
                active_positions.append(pos)
        
        self.positions = active_positions

        # --- 2. Check Risk Limits ---
        # If daily loss limit reached, stop trading for the day
        equity_drop_percent = (self.initial_capital - self.current_equity) / self.initial_capital * 100
        if self.daily_pnl < -self.max_daily_loss or equity_drop_percent > 20.0: 
             return executed_orders, "Risk Limit Hit"

        if self.trades_today >= self.max_trades_per_day:
             return executed_orders, "Max Trades Hit"

        # --- 3. Execute Signal ---
        if signal == 0:
            return executed_orders, messages

        # Logic: If Long Signal and no positions (or pyramiding allowed)
        if signal == 1: 
            # Simple logic: Only one position at a time for this base version
            if not self.positions: 
                # Position Sizing
                risk_amt = self.current_equity * self.base_risk
                # Stop distance based on ATR if available, else 2%
                dist = (atr * tsl_mult) if atr > 0 else (price * 0.02)
                
                # Ensure distance isn't zero
                if dist == 0: dist = price * 0.01

                qty = risk_amt / dist
                cost = qty * price
                
                # Check if we have enough cash
                if cost > self.cash:
                    qty = (self.cash * 0.98) / price # 98% of cash to leave room for fees
                    cost = qty * price

                if qty > 0:
                    fee_val = cost * self.fee
                    self.cash -= (cost + fee_val)
                    
                    stop_price = price - dist
                    
                    new_pos = {
                        'side': 'long',
                        'qty': qty,
                        'entry': price,
                        'stop_loss': stop_price,
                        'time': time
                    }
                    self.positions.append(new_pos)
                    self.trades_today += 1
                    executed_orders.append(f"BUY {qty:.4f} @ {price}")

        elif signal == -1:
            # Close Longs if any (Flip logic or just exit)
            if self.positions:
                 # Close all positions logic (Market Sell)
                 for p in self.positions:
                     exit_price = price
                     pnl = (exit_price - p['entry']) * p['qty']
                     fee_cost = (exit_price * p['qty']) * self.fee
                     pnl -= fee_cost
                     
                     self.cash += (exit_price * p['qty'])
                     self.cash -= fee_cost
                     
                     self.trades.append({
                        'entry_time': p['time'],
                        'exit_time': time,
                        'side': 'long',
                        'entry_price': p['entry'],
                        'exit_price': exit_price,
                        'qty': p['qty'],
                        'pnl': pnl,
                        'reason': 'Signal Close'
                    })
                     self.daily_pnl += pnl
                     executed_orders.append(f"SELL {p['qty']:.4f} @ {price}")
                 
                 self.positions = [] # Clear positions

        return executed_orders, messages

    def step_equity(self, current_price):
        """Updates the floating equity based on current market price."""
        floating_pnl = 0.0
        for p in self.positions:
            if p['side'] == 'long':
                val = p['qty'] * current_price
                cost = p['qty'] * p['entry']
                floating_pnl += (val - cost)
            # Add short logic if needed
            
        self.current_equity = self.cash + sum(p['qty'] * current_price for p in self.positions if p['side']=='long') # Simplified for long-only portfolio value
        
        # A more accurate equity calculation for mixed/margin accounts:
        # self.current_equity = self.cash + margin_collateral + floating_pnl
        
        return self.current_equity
