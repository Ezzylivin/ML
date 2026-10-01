# File: ml.py
# 🚀 UPGRADE: v71.5 - "The Production Hardened Sovereign"
import os
import json
import pandas as pd
import numpy as np
import joblib
import logging
import traceback
import math
import ccxt
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import pandas_ta as ta
import warnings
from datetime import datetime, timezone

# --- 1. CONFIGURATION ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/root/Project/ML"
MODEL_DIR = os.path.join(PROJECT_ROOT, "app/models")
DEFAULT_TAKER_FEE = 0.0006 
SLIPPAGE_PCT = 0.0001

app = FastAPI(title="Production Hardened Strategy Server v71.5")

# --- 2. DYNAMIC STRATEGY REGISTRY ---
def compute_signal(df, code, p, ml_model=None, ml_thresh=0.5):
    try:
        def get_v(k, d): return float(p.get(k, d))
        
        # 🧠 STATISTICAL UPGRADE: 5-Bar Rolling ML Window
        # Note: Window is kept short to minimize lag while improving pattern stability.
        if ml_model is not None:
            window = df.tail(5).select_dtypes(include=[np.number])
            if len(window) >= 5:
                probs = ml_model.predict_proba(window)
                p_long = np.mean(probs[:, 1]) 
                if p_long > ml_thresh: return 1
                if p_long < (1 - ml_thresh): return -1
                return 0

        if code == 'sma_crossover':
            f, s = int(get_v('sma_fast_period', 10)), int(get_v('sma_slow_period', 50))
            fast, slow = df.ta.sma(length=f), df.ta.sma(length=s)
            return np.where((fast > slow) & (fast.shift(1) <= slow.shift(1)), 1,
                  np.where((fast < slow) & (fast.shift(1) >= slow.shift(1)), -1, 0))

        if code == 'macd_crossover':
            macd = df.ta.macd()
            m, s = macd.iloc[:, 0], macd.iloc[:, 2]
            return np.where((m > s) & (m.shift(1) <= s.shift(1)), 1,
                   np.where((m < s) & (m.shift(1) >= s.shift(1)), -1, 0))

        if code == 'ichimoku_system':
            ichi = df.ta.ichimoku()[0]
            tenkan, kijun = ichi.iloc[:, 0], ichi.iloc[:, 1]
            return np.where((tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1)), 1,
                   np.where((tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1)), -1, 0))

        if code == 'psar_flip_signal':
            psar = df.ta.psar()
            pl, ps = psar.iloc[:, 0], psar.iloc[:, 1]
            return np.where(pl.notna() & pl.shift(1).isna(), 1,
                   np.where(ps.notna() & ps.shift(1).isna(), -1, 0))

        if code == 'atr_breakout':
            l, mult = int(get_v('atr_length', 14)), get_v('atr_multiplier', 3.0)
            atr, ema = df.ta.atr(length=l), df.ta.ema(length=20)
            u, lo = ema + (atr * mult), ema - (atr * mult)
            return np.where((df['close'] > u) & (df['close'].shift(1) <= u.shift(1)), 1,
                   np.where((df['close'] < lo) & (df['close'].shift(1) >= lo.shift(1)), -1, 0))

    except Exception as e:
        logger.error(f"Registry Logic Error [{code}]: {e}")
    return np.zeros(len(df))

# --- 3. EXECUTION ENGINE (RESILIENT RISK) ---
class PyramidManager:
    def __init__(self, capital, fee, max_layers=3, base_risk=0.01):
        self.initial_capital = float(capital)
        self.available_cash = float(capital)
        self.fee = fee
        self.max_layers = max_layers
        self.base_risk = base_risk 
        self.positions = [] 
        self.trades = []
        self.peak_equity = float(capital)
        self.drawdown_series = []

    def get_dynamic_risk(self, atr, price):
        # 🛡️ CAVEAT: Simple 2x ATR heuristic. 
        # Future enhancement: Integrate VIX or Volatility-of-Volatility index.
        if atr == 0 or price == 0: return self.base_risk
        volatility_ratio = (atr / price) * 100
        if volatility_ratio > 2.0: return self.base_risk * 0.5
        return self.base_risk

    def handle(self, signal, price, time, atr=0, tsl_mult=3.5, rolling_std=0):
        if signal == 1 and len(self.positions) < self.max_layers:
            # 🛡️ CAVEAT: Rolling 30-bar STD may understate long-term structural volatility.
            risk_dist = (atr * tsl_mult) if atr > 0 else (rolling_std * 2.0)
            if risk_dist == 0: risk_dist = price * 0.05 

            dynamic_risk = self.get_dynamic_risk(atr, price)
            risk_amount = self.initial_capital * dynamic_risk
            
            qty = risk_amount / risk_dist
            cost = qty * price
            
            if self.available_cash >= cost:
                self.positions.append({
                    'qty': qty, 'entry': price, 'time': time, 
                    'peak': price, 'partial_hit': False, 'risk_at_entry': risk_dist
                })
                self.available_cash -= cost

        elif signal == -1 and self.positions:
            self.close_all(price, time, "SIGNAL")
            
        if self.positions:
            self.manage_active_risk(price, atr, tsl_mult, time, rolling_std)

    def manage_active_risk(self, price, atr, tsl_mult, time, rolling_std):
        exit_dist = (atr * tsl_mult) if atr > 0 else (rolling_std * 2.0)
        for i, pos in enumerate(self.positions):
            if not pos['partial_hit']:
                pnl_per_share = price - pos['entry']
                if pnl_per_share >= (pos['risk_at_entry'] * 2.0):
                    sell_qty = pos['qty'] * 0.5
                    self.available_cash += (sell_qty * price * (1 - self.fee))
                    self.positions[i]['qty'] -= sell_qty
                    self.positions[i]['partial_hit'] = True
                    logger.info(f"✂️ Partial Exit (50%) | {time}")

            pos['peak'] = max(pos['peak'], price)
            if price <= (pos['peak'] - exit_dist):
                self.close_all(price, time, "TSL")
                break

    def close_all(self, price, time, reason):
        for pos in self.positions:
            raw_pnl = (price * (1 - self.fee) - pos['entry']) * pos['qty']
            self.available_cash += (pos['entry'] * pos['qty']) + raw_pnl
            self.trades.append({'entryTime': pos['time'], 'exitTime': time, 'profit': raw_pnl, 'type': reason})
        self.positions = []

    def get_equity(self, price):
        current_equity = self.available_cash + sum(p['qty'] * price for p in self.positions)
        self.peak_equity = max(self.peak_equity, current_equity)
        self.drawdown_series.append((self.peak_equity - current_equity) / self.peak_equity)
        return current_equity

# --- 4. BACKTEST & ANALYTICS ---
@app.post('/api/ml/run-combo-backtest')
async def handle_run(request: Request):
    try:
        body = await request.json()
        exchange = ccxt.coinbase()
        ohlcv = exchange.fetch_ohlcv(body['symbol'].replace('-','/'), body['timeframe'], limit=1000)
        df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
        df['datetime'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        df.set_index('datetime', inplace=True)

        ml_model = None
        if body.get('mlMode') == 'on' and body.get('mlModel'):
            path = os.path.join(MODEL_DIR, f"{body['mlModel']}.joblib")
            if os.path.exists(path): ml_model = joblib.load(path)

        strategies = body.get('strategies', [])
        signals_matrix = pd.DataFrame(index=df.index)
        for i, s in enumerate(strategies):
            signals_matrix[f's_{i}'] = compute_signal(df, s['code'], s.get('params', {}), ml_model, body.get('mlThreshold', 0.5))

        combined = np.where(signals_matrix.sum(axis=1) > 0, 1, np.where(signals_matrix.sum(axis=1) < 0, -1, 0))
        df['final_sig'] = pd.Series(combined, index=df.index).shift(1).fillna(0)

        mgr = PyramidManager(body['initialBalance'], DEFAULT_TAKER_FEE, body.get('maxPyramiding', 3))
        df.ta.atr(append=True)
        df['std_fallback'] = df['close'].rolling(30).std()
        
        curve_data = []
        for t, row in df.iterrows():
            mgr.handle(row['final_sig'], row['close'], t.isoformat(), row.get('ATR_14', 0), 3.5, row.get('std_fallback', 0))
            curve_data.append({'timestamp': t.isoformat(), 'balance': mgr.get_equity(row['close'])})

        # Sharpe Calculation from Equity Curve (Daily Proxy)
        equity_series = pd.Series([c['balance'] for c in curve_data])
        returns = equity_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if not returns.empty and returns.std() != 0 else 0
        total_ret = ((equity_series.iloc[-1] - body['initialBalance']) / body['initialBalance']) * 100
        max_dd = max(mgr.drawdown_series) * 100 if mgr.drawdown_series else 0

        return JSONResponse(content={
            "metrics": {
                "totalReturn": round(total_ret, 2),
                "maxDrawdown": round(max_dd, 2),
                "totalTrades": len(mgr.trades),
                "sharpe": round(sharpe, 2),
                "calmar": round(total_ret / max_dd, 2) if max_dd != 0 else 0
            },
            "equityCurve": curve_data,
            "tradeBreakdown": mgr.trades
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
