import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt 
from contextlib import asynccontextmanager

# 🟢 1. IMPORT SOCKET HELPERS
from app.services.socket_emitter import emit_log, emit_status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SystemAPI")

ACTIVE_BOTS = {} 

class BotStartRequest(BaseModel):
    userId: str
    config: Dict[str, Any]

class BotActionRequest(BaseModel):
    userId: str
    botId: Optional[str] = None
    capitalAllocation: Optional[float] = 1000.0

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="NEO-V7 Hybrid Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 🧠 1. SHARED STRATEGY BRAIN
# ==========================================
class StrategyBrain:
    @staticmethod
    def calculate_signals(df: pd.DataFrame, strategies: list):
        thoughts = []
        votes = 0
        current_price = df['close'].iloc[-1]
        
        for strat in strategies:
            code = strat.get('code')
            params = strat.get('params', {})
            try:
                if code == "rsi_threshold":
                    rsi = ta.rsi(df['close'], length=int(params.get('rsi_length', 14)))
                    if rsi is not None:
                        val = rsi.iloc[-1]
                        thoughts.append(f"RSI: {val:.2f}")
                        if val < params.get('oversold', 30): votes += 1
                        elif val > params.get('overbought', 70): votes -= 1
                elif code == "stoch":
                    stoch = ta.stoch(df['high'], df['low'], df['close'], k=int(params.get('k_period', 14)))
                    if stoch is not None:
                        k, d = stoch.iloc[-1, 0], stoch.iloc[-1, 1]
                        thoughts.append(f"Stoch: {k:.1f}/{d:.1f}")
                        if k < 20 and k > d: votes += 1
                        elif k > 80 and k < d: votes -= 1
                elif code == "bb_fade":
                    bb = ta.bbands(df['close'], length=20, std=2.0)
                    if bb is not None:
                        lower, upper = bb.iloc[-1, 0], bb.iloc[-1, 2]
                        thoughts.append(f"BB: {lower:.1f}|{upper:.1f}")
                        if current_price < lower: votes += 1
                        elif current_price > upper: votes -= -1
            except Exception:
                continue

        return (1 if votes > 0 else -1 if votes < 0 else 0), ", ".join(thoughts)

# ==========================================
# 🚀 2. ENGINE: LIVE TRADER (Finalized Synchronization)
# ==========================================
async def live_neural_heartbeat(user_id: str):
    """Proactive loop. Tracks PnL and broadcasts markers for UI charts."""
    while user_id in ACTIVE_BOTS and ACTIVE_BOTS[user_id]["status"] == "running":
        bot = ACTIVE_BOTS[user_id]
        try:
            symbol = bot['config'].get('symbol', 'BTC-USD')
            timeframe = bot['config'].get('timeframe', '1h')
            new_data = await fetch_live_candles_ccxt(symbol, timeframe, limit=50)
            
            if new_data:
                latest_price = new_data[-1]['close']
                bot['candles'] = new_data 
                df = pd.DataFrame(bot['candles'])
                
                # 📈 A. Real-Time Metrics Calculation
                unrealized_pnl = 0
                total_exposure = 0
                
                for pos in bot['positions']:
                    trade_pnl = (latest_price - pos['entry']) * pos['size']
                    unrealized_pnl += trade_pnl
                    total_exposure += (pos['size'] * latest_price)

                current_val = bot['balance'] + unrealized_pnl if bot['positions'] else bot['balance']
                
                # 📡 B. BROADCAST UPDATED METRICS + MARKERS
                emit_status(user_id, {
                    "status": "running",
                    "startedAt": bot.get("startedAt"), # 🟢 Fixes Uptime
                    "initialCapital": bot["initial_capital"], # 🟢 Prevents $1000 Reset
                    "currentBalance": round(current_val, 2),
                    "unrealizedPnl": round(unrealized_pnl, 2),
                    "dailyProfit": round(bot.get("daily_profit", 0), 2),
                    "exposure": round((total_exposure / bot['initial_capital']) * 100, 2),
                    "activePositions": bot['positions'],
                    "tradeMarkers": bot.get("trade_history", []), # 🟢 Real-time Chart Arrows
                    "candles": bot['candles'][-20:] 
                })

                # 🧠 C. Trading Logic
                strats = bot['config'].get('strategies', [])
                signal, thoughts = StrategyBrain.calculate_signals(df, strats)
                
                # Execution
                if signal == 1 and not bot['positions']:
                    size = bot['balance'] / latest_price
                    new_pos = {"type": "buy", "entry": latest_price, "size": size, "time": datetime.now().isoformat()}
                    bot['positions'].append(new_pos)
                    # 🟢 Add to history for chart markers
                    bot['trade_history'].append(new_pos) 
                    bot['balance'] = 0
                    emit_log(user_id, f"🟢 BUY @ {latest_price} | Capital at risk: ${round(size * latest_price, 2)}")
                
                elif signal == -1 and bot['positions']:
                    pos = bot['positions'].pop(0)
                    profit = (latest_price - pos['entry']) * pos['size']
                    bot['daily_profit'] = bot.get('daily_profit', 0) + profit
                    bot['balance'] = pos['size'] * latest_price
                    # 🟢 Add sell marker to history
                    bot['trade_history'].append({"type": "sell", "price": latest_price, "time": datetime.now().isoformat()})
                    emit_log(user_id, f"🔴 SELL @ {latest_price} | Realized PnL: ${round(profit, 2)}")
                    
            await asyncio.sleep(15) 

        except Exception as e:
            logger.error(f"Live Loop Error: {e}")
            await asyncio.sleep(10)

# --- UTILS & ENDPOINTS ---
async def fetch_live_candles_ccxt(symbol: str, timeframe: str, limit: int = 50):
    clean_symbol = symbol.replace('-', '/')
    async with ccxt.coinbase() as exchange:
        try:
            ohlcv = await exchange.fetch_ohlcv(clean_symbol, timeframe, limit=limit)
            return [{"timestamp": datetime.fromtimestamp(c[0]/1000, tz=timezone.utc).isoformat(),
                     "time": c[0]/1000, "open": c[1], "high": c[2], "low": c[3], "close": c[4]} for c in ohlcv]
        except: return []

@app.post("/api/bot/start")
async def start_bot(data: BotStartRequest, background_tasks: BackgroundTasks):
    user_id = data.userId.strip()
    capital = float(data.config.get("capitalAllocation", 1000))
    
    ACTIVE_BOTS[user_id] = {
        "status": "running",
        "config": data.config,
        "initial_capital": capital,
        "balance": capital,
        "positions": [],
        "trade_history": [], # 🟢 Stores markers for the chart
        "daily_profit": 0,
        "startedAt": datetime.now().isoformat(), # 🟢 Fixes Uptime
        "candles": [],
        "logs": []
    }
    emit_log(user_id, f"🚀 Bot Initialized with ${capital}")
    background_tasks.add_task(live_neural_heartbeat, user_id)
    return {"status": "running"}

@app.get("/api/bot/status")
async def get_bot_status(userId: str):
    userId = userId.strip()
    if userId not in ACTIVE_BOTS: return {"status": "stopped"}
    bot = ACTIVE_BOTS[userId]
    return {
        "status": bot["status"], 
        "currentBalance": bot["balance"],
        "activePositions": bot["positions"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
