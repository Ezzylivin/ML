# File: ml.py
# 🚀 UPGRADE: v70.9.8 - "The Fully Traced Engine"
# 🛠 FEATURES: Strategy Registry, Risk Sizing, Winners Endpoint, and Comprehensive Tracing.

import os
import json
import pandas as pd
import numpy as np
import logging
import traceback
import joblib
import ccxt
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas_ta as ta
import warnings
from datetime import datetime

# --- 1. CONFIGURATION ---
warnings.filterwarnings('ignore')
# Enhanced logging format to include timestamps and line numbers
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/root/Project/ML"
MODEL_DIR = os.path.join(PROJECT_ROOT, "app/models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data/optimizer_results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI(title="Trading ML Engine v70.9.8")

# --- 2. DYNAMIC STRATEGY REGISTRY ---
def compute_signal(df, code, p):
    """Dynamically computes signals based on strategy code and params."""
    try:
        logger.info(f"🔍 Computing signals for strategy: {code}")
        
        if code == 'sma_crossover':
            f, s = int(p.get('sma_fast_period', 50)), int(p.get('sma_slow_period', 200))
            fast = df.ta.sma(length=f)
            slow = df.ta.sma(length=s)
            
            # Check for NaN (Warmup issue)
            if fast.isna().all() or slow.isna().all():
                logger.warning(f"⚠️ SMA {f}/{s} produced all NaNs. Increase data limit or lower periods.")
                
            sig = np.where((fast > slow) & (fast.shift(1) <= slow.shift(1)), 1, 
                   np.where((fast < slow) & (fast.shift(1) >= slow.shift(1)), -1, 0))
            
        elif code == 'macd_crossover':
            f, s, sig_len = int(p.get('macd_fast_period', 12)), int(p.get('macd_slow_period', 26)), int(p.get('macd_signal_period', 9))
            macd = df.ta.macd(fast=f, slow=s, signal=sig_len)
            sig = np.where((macd.iloc[:, 0] > macd.iloc[:, 2]) & (macd.iloc[:, 0].shift(1) <= macd.iloc[:, 2].shift(1)), 1,
                   np.where((macd.iloc[:, 0] < macd.iloc[:, 2]) & (macd.iloc[:, 0].shift(1) >= macd.iloc[:, 2].shift(1)), -1, 0))

        elif code == 'rsi_divergence':
            l = int(p.get('rsi_length', 14))
            os, ob = float(p.get('oversold_level', 30)), float(p.get('overbought_level', 70))
            rsi = df.ta.rsi(length=l)
            sig = np.where((rsi > os) & (rsi.shift(1) <= os), 1,
                   np.where((rsi < ob) & (rsi.shift(1) >= ob), -1, 0))

        elif code == 'bollinger_bands':
            l, std = int(p.get('bb_length', 20)), float(p.get('bb_std', 2))
            bb = df.ta.bbands(length=l, std=std)
            sig = np.where(df['close'] < bb.iloc[:, 0], 1,
                   np.where(df['close'] > bb.iloc[:, 2], -1, 0))
        else:
            logger.error(f"❌ Strategy code '{code}' not recognized in Registry.")
            return np.zeros(len(df))

        logger.info(f"📊 {code} Signal Stats -> Buys: {(sig==1).sum()}, Sells: {(sig==-1).sum()}")
        return sig

    except Exception as e:
        logger.error(f"💥 Logic Error for {code}: {str(e)}")
        return np.zeros(len(df))

# --- 3. EXECUTION ENGINE ---
class PyramidManager:
    def __init__(self, capital, fee, risk_pct):
        self.cash = float(capital)
        self.risk_pct = float(risk_pct) / 100.0
        self.fee = fee
        self.pos = None
        self.trades = []
        logger.info(f"🏗️ Manager Init | Cap: ${capital} | Fee: {fee} | Risk: {risk_pct}%")

    def handle(self, signal, price, time):
        # Entry
        if signal == 1 and not self.pos:
            allocation = self.cash * self.risk_pct
            if allocation < 5: # Minimum trade size check
                return
            qty = allocation / (price * (1 + self.fee))
            self.pos = {'qty': qty, 'entry': price, 'time': time, 'cost': allocation}
            self.cash -= allocation
            logger.info(f"🟢 ENTRY | Time: {time} | Price: {price} | Qty: {qty:.4f}")

        # Exit
        elif signal == -1 and self.pos:
            profit = (price * (1 - self.fee) - self.pos['entry']) * self.pos['qty']
            self.cash += (self.pos['cost'] + profit)
            self.trades.append({
                'entryTime': self.pos['time'], 
                'exitTime': time, 
                'profit': profit, 
                'exitPrice': price,
                'side': 'long'
            })
            logger.info(f"🔴 EXIT  | Time: {time} | Price: {price} | Profit: ${profit:.2f}")
            self.pos = None

    def get_equity(self, price):
        return self.cash + (self.pos['qty'] * price if self.pos else 0)

# --- 4. ENDPOINTS ---

@app.get("/api/data/symbols")
def get_symbols():
    return ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD"]

@app.get("/api/ml/available-models")
def list_models():
    models = [{"id": f.split('.')[0], "name": f.split('.')[0].replace('_',' ').title()} 
            for f in os.listdir(MODEL_DIR) if f.endswith(('.joblib', '.pkl'))]
    logger.info(f"📡 Listed {len(models)} available ML models.")
    return models

@app.get("/api/bot/winners")
def get_winners():
    winners = []
    if not os.path.exists(RESULTS_DIR):
        logger.warning(f"⚠️ Results directory missing: {RESULTS_DIR}")
        return []
    for f in os.listdir(RESULTS_DIR):
        if f.endswith(".json"):
            try:
                with open(os.path.join(RESULTS_DIR, f), 'r') as file:
                    data = json.load(file)
                    winners.append({"id": f, "name": f.replace('.json',''), "config": data})
            except Exception as e:
                logger.error(f"❌ Error reading strategy file {f}: {e}")
    logger.info(f"🏆 Loaded {len(winners)} strategies from Alpha vault.")
    return winners

@app.post('/api/backtest/run')
@app.post('/api/ml/run-combo-backtest')
async def handle_run(request: Request):
    try:
        body = await request.json()
        symbol = body.get('symbol', 'BTC-USD')
        tf = body.get('timeframe', '1h')
        
        logger.info(f"🚀 STARTING BACKTEST: {symbol} [{tf}]")
        
        # 1. Fetch History
        exchange = ccxt.coinbase({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol.replace('-','/'), tf, limit=1000)
        df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
        df['datetime'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        df.set_index('datetime', inplace=True)
        
        logger.info(f"📥 Fetched {len(df)} candles for {symbol}. First: {df.index[0]}, Last: {df.index[-1]}")

        # 2. Dynamic Signal Aggregation
        strategies = body.get('strategies', [])
        if not strategies:
            logger.warning("⚠️ No strategies provided in payload!")
            
        signals_matrix = pd.DataFrame(index=df.index)
        for i, s in enumerate(strategies):
            signals_matrix[f's_{i}'] = compute_signal(df, s['code'], s.get('params', {}))

        # 3. Decision Logic (Majority Voting)
        rule = body.get('comboConfig', {}).get('combinationRule', body.get('params', {}).get('hybridMode', 'OR'))
        logger.info(f"🧠 Combination Rule: {rule}")
        
        if rule == 'AND':
            combined = np.where((signals_matrix == 1).all(axis=1), 1, np.where((signals_matrix == -1).all(axis=1), -1, 0))
        else:
            net_sentiment = signals_matrix.sum(axis=1)
            combined = np.where(net_sentiment > 0, 1, np.where(net_sentiment < 0, -1, 0))

        # 4. Simulation
        df['final_sig'] = pd.Series(combined, index=df.index).shift(1).fillna(0)
        mgr = PyramidManager(body.get('initialBalance', 1000), body.get('fee', 0.001), body.get('riskPercentage', 10))
        
        curve = []
        for t, row in df.iterrows():
            mgr.handle(row['final_sig'], row['close'], t.isoformat())
            curve.append({'timestamp': t.isoformat(), 'balance': round(mgr.get_equity(row['close']), 2)})

        total_ret = ((mgr.get_equity(df.iloc[-1]['close']) - body.get('initialBalance', 1000)) / body.get('initialBalance', 1000)) * 100
        
        logger.info(f"🏁 BACKTEST COMPLETE | Return: {total_ret:.2f}% | Trades: {len(mgr.trades)}")

        res = {
            "metrics": {
                "totalReturn": round(total_ret, 4), 
                "finalBalance": round(mgr.cash if not mgr.pos else mgr.get_equity(df.iloc[-1]['close']), 2), 
                "totalTrades": len(mgr.trades)
            },
            "equityCurve": curve,
            "tradeBreakdown": mgr.trades,
            "candleData": [{"time": i.isoformat(), "open": r.open, "high": r.high, "low": r.low, "close": r.close} for i, r in df.iterrows()]
        }
        return JSONResponse(content={"combinedResult": res, **res})

    except Exception as e:
        logger.error(f"🔥 Critical Failure in handle_run: {traceback.format_exc()}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
