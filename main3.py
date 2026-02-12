import asyncio
import logging
import os
import json
import glob
import random
import ccxt.async_support as ccxt 
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pandas as pd
import pandas_ta as ta

# 🟢 1. IMPORT SOCKET HELPERS
from app.services.socket_emitter import emit_log, emit_status

# 🟢 CONFIG & PATHS
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SystemAPI")

# 🧠 IN-MEMORY STATE
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
    """The unified logic core used by both Live and Backtest engines."""
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
                    rsi_series = ta.rsi(df['close'], length=int(params.get('rsi_length', 14)))
                    if rsi_series is not None:
                        val = rsi_series.iloc[-1]
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
                        elif current_price > upper: votes -= 1
            except Exception:
                continue

        final_signal = 1 if votes > 0 else -1 if votes < 0 else 0
        return final_signal, ", ".join(thoughts)

# ==========================================
# 📊 2. ENGINE A: BACKTESTER (Historical)
# ==========================================
class BacktestEngine:
    """Simulates trading on historical data bar-by-bar."""
    @staticmethod
    def run(historical_df: pd.DataFrame, config: Dict[str, Any]):
        balance = float(config.get("capitalAllocation", 1000))
        initial_balance = balance
        position = None
        equity_curve = []
        logs = [f"Backtest Initialized: ${balance}"]

        # Warm-up period for indicators
        warm_up = 20
        for i in range(warm_up, len(historical_df)):
            window = historical_df.iloc[:i+1]
            price = window['close'].iloc[-1]
            time_str = window['timestamp'].iloc[-1]

            # Use Shared Brain for 1:1 consistency with live
            signal, thoughts = StrategyBrain.calculate_signals(window, config.get('strategies', []))

            if signal == 1 and not position:
                position = {"entry_price": price, "size": balance / price}
                balance = 0
                logs.append(f"[{time_str}] 🟢 BUY at {price} | Indicators: {thoughts}")
            elif signal == -1 and position:
                balance = position['size'] * price
                position = None
                logs.append(f"[{time_str}] 🔴 SELL at {price} | Indicators: {thoughts}")

            equity_curve.append({
                "time": time_str, 
                "balance": balance if not position else position['size'] * price
            })

        final_bal = balance if not position else position['size'] * price
        return {
            "totalProfit": final_bal - initial_balance,
            "roi": ((final_bal / initial_balance) - 1) * 100,
            "logs": logs[-100:],
            "equityCurve": equity_curve
        }

# ==========================================
# 🚀 3. ENGINE B: LIVE TRADER (1m Heartbeat)
# ==========================================
async def live_neural_heartbeat(user_id: str):
    """Proactive 'Thinking' loop. Explains trend and reasoning every minute."""
    while user_id in ACTIVE_BOTS and ACTIVE_BOTS[user_id]["status"] == "running":
        bot = ACTIVE_BOTS[user_id]
        try:
            symbol = bot['config'].get('symbol', 'BTC-USD')
            timeframe = bot['config'].get('timeframe', '1h')
            new_data = await fetch_live_candles_ccxt(symbol, timeframe, limit=50)
            
            if new_data:
                latest = new_data[-1]
                bot['candles'] = new_data 
                df = pd.DataFrame(bot['candles'])
                
                # 📈 A. Trend Analysis (EMA 20)
                ema_20_series = ta.ema(df['close'], length=20)
                ema_20 = ema_20_series.iloc[-1] if ema_20_series is not None else latest['close']
                trend = "UPWARD" if latest['close'] > ema_20 else "DOWNWARD"
                volatility = "HIGH" if (latest['high'] - latest['low']) > (latest['close'] * 0.005) else "LOW"
                
                # 🧠 B. Neural Logic Processing
                strats = bot['config'].get('strategies', [])
                if not strats and bot['config'].get('comboConfig'):
                     strats = [{'code': c} for c in bot['config']['comboConfig'].get('strategyCodes', [])]

                signal, thoughts = StrategyBrain.calculate_signals(df, strats)
                
                # 🧐 C. Detailed Reasoning Generation
                logic_steps = []
                if signal == 0:
                    logic_steps.append(f"Market Trend: {trend} (Volatility: {volatility}).")
                    logic_steps.append(f"Internal Stream: {thoughts if thoughts else 'Analyzing market structure'}.")
                    logic_steps.append("Status: Neutral. Waiting for indicators to align for entry.")
                elif signal == 1:
                    logic_steps.append("Trend: BULLISH Alignment confirmed.")
                    logic_steps.append("Sentiment: Model is highly confident in LONG position.")
                elif signal == -1:
                    logic_steps.append("Trend: BEARISH Alignment confirmed.")
                    logic_steps.append("Sentiment: Model is highly confident in SHORT/EXIT position.")
                
                # D. Push to Neural Stream (Memory + Socket)
                for step in logic_steps:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    stream_msg = f"[{timestamp}] 🧠 {step}"
                    
                    # 1. Save to Memory
                    if not bot['logs'] or stream_msg.split("] ")[-1] not in bot['logs'][0]:
                        bot['logs'].insert(0, stream_msg)
                        
                        # 2. 🟢 EMIT SOCKET LOG
                        emit_log(user_id, stream_msg)

                # E. Order Execution
                price = latest['close']
                timestamp_str = datetime.now().strftime('%H:%M:%S')
                
                if signal == 1 and not bot['positions']:
                    bot['positions'].append({"type": "long", "entry": price, "size": bot['balance'] / price})
                    bot['balance'] = 0
                    
                    msg = f"[{timestamp_str}] 🟢 BUY order executed at {price}"
                    bot['logs'].insert(0, msg)
                    
                    # 🟢 EMIT BUY
                    emit_log(user_id, msg)
                    emit_status(user_id, {"status": "running", "currentBalance": bot['balance']})

                elif signal == -1 and bot['positions']:
                    pos = bot['positions'].pop(0)
                    bot['balance'] = pos['size'] * price
                    
                    msg = f"[{timestamp_str}] 🔴 SELL order executed at {price}"
                    bot['logs'].insert(0, msg)
                    
                    # 🟢 EMIT SELL
                    emit_log(user_id, msg)
                    emit_status(user_id, {"status": "running", "currentBalance": bot['balance']})
                    
            await asyncio.sleep(60) # 1-Minute Heartbeat

        except Exception as e:
            logger.error(f"Live Loop Error: {e}")
            await asyncio.sleep(10)

# --- 4. DATA UTILS ---
async def fetch_live_candles_ccxt(symbol: str, timeframe: str, limit: int = 50):
    clean_symbol = symbol.replace('-', '/')
    exchanges = [ccxt.coinbase(), ccxt.kraken()]
    for ex in exchanges:
        try:
            ohlcv = await ex.fetch_ohlcv(clean_symbol, timeframe, limit=limit)
            await ex.close()
            if ohlcv:
                return [{"timestamp": datetime.fromtimestamp(c[0]/1000, tz=timezone.utc).isoformat(),
                         "time": c[0]/1000, "open": c[1], "high": c[2], "low": c[3], "close": c[4]} for c in ohlcv]
        except:
            await ex.close()
    return []

async def initialize_bot_background(user_id: str, config: Dict[str, Any]):
    try:
        symbol, timeframe = config.get('symbol', 'BTC-USD'), config.get('timeframe', '1h')
        bot = ACTIVE_BOTS.get(user_id)
        if bot:
            bot["candles"] = await fetch_live_candles_ccxt(symbol, timeframe, limit=100)
            bot["status"] = "running"
            
            msg = f"[{datetime.now().strftime('%H:%M:%S')}] Neural Stream Connected."
            bot["logs"].append(msg)
            
            # 🟢 EMIT CONNECTED
            emit_log(user_id, msg)
            emit_status(user_id, {"status": "running", "currentBalance": bot["balance"]})
            
            asyncio.create_task(live_neural_heartbeat(user_id))
    except Exception as e:
        logger.error(f"Background Task Error: {e}")

# --- 5. ENDPOINTS ---

@app.post("/api/backtest/run")
async def run_backtest(data: BotStartRequest):
    history = await fetch_live_candles_ccxt(data.config['symbol'], data.config['timeframe'], limit=500)
    if not history: return {"error": "History fetch failed."}
    return BacktestEngine.run(pd.DataFrame(history), data.config)

@app.post("/api/bot/start")
async def start_bot(data: BotStartRequest, background_tasks: BackgroundTasks):
    user_id = data.userId.strip()
    
    # 🟢 EMIT HANDSHAKE
    msg = f"[{datetime.now().strftime('%H:%M:%S')}] Handshake complete. Fetching data..."
    emit_log(user_id, msg)
    
    ACTIVE_BOTS[user_id] = {
        "status": "initializing",
        "config": data.config,
        "balance": float(data.config.get("capitalAllocation", 1000)),
        "positions": [],
        "candles": [],
        "logs": [msg],
        "equityCurve": [{"time": datetime.now().isoformat(), "balance": float(data.config.get("capitalAllocation", 1000))}]
    }
    background_tasks.add_task(initialize_bot_background, user_id, data.config)
    return {"status": "initializing"}

@app.get("/api/bot/status")
async def get_bot_status(userId: str):
    userId = userId.strip()
    if userId not in ACTIVE_BOTS: return {"status": "stopped", "logs": []}
    bot = ACTIVE_BOTS[userId]
    return {
        "status": bot["status"], "currentBalance": bot["balance"],
        "activePositions": bot["positions"], "candles": bot["candles"],
        "logs": bot["logs"][:100], "equityCurve": bot["equityCurve"]
    }

@app.post("/api/bot/stop")
async def stop_bot(data: BotActionRequest):
    uid = data.userId.strip()
    if uid in ACTIVE_BOTS: 
        ACTIVE_BOTS[uid]["status"] = "stopped"
        # 🟢 EMIT STOP
        emit_status(uid, {"status": "stopped", "currentBalance": ACTIVE_BOTS[uid]["balance"]})
        emit_log(uid, f"[{datetime.now().strftime('%H:%M:%S')}] Bot Stopped.")
    return {"status": "stopped"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
(venv) root@intelligent-mendel:~/Project/ML# 
