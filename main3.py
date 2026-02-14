import asyncio
import logging
import os
import joblib
import json
import sqlite3
import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt 
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Union
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# 🟢 SOCKET HELPERS
from app.services.socket_emitter import emit_log, emit_status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEO-Engine")

ACTIVE_BOTS = {} 

# ==========================================
# 🗄️ 0. DATABASE HANDLER (Persistence Layer)
# ==========================================
class DatabaseHandler:
    DB_FILE = "bot_state.db"

    @classmethod
    def init_db(cls):
        """Initialize the SQLite database for persistence."""
        conn = sqlite3.connect(cls.DB_FILE)
        c = conn.cursor()
        # Create table to store full bot state
        c.execute('''CREATE TABLE IF NOT EXISTS bot_sessions
                     (user_id TEXT PRIMARY KEY, config TEXT, balance REAL, 
                      positions TEXT, trade_history TEXT, equity_curve TEXT, logs TEXT,
                      status TEXT, last_update TIMESTAMP)''')
        conn.commit()
        conn.close()

    @classmethod
    def save_state(cls, user_id, bot_data):
        """Save the current bot state to DB."""
        conn = sqlite3.connect(cls.DB_FILE)
        c = conn.cursor()
        # Serialize complex lists/dicts to JSON
        c.execute('''INSERT OR REPLACE INTO bot_sessions 
                     (user_id, config, balance, positions, trade_history, equity_curve, logs, status, last_update)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (user_id, 
                   json.dumps(bot_data['config']), 
                   bot_data['balance'], 
                   json.dumps(bot_data['positions']), 
                   json.dumps(bot_data['trade_history']), 
                   json.dumps(bot_data.get('equityCurve', [])),
                   json.dumps(bot_data.get('logs', [])), 
                   bot_data['status'],
                   datetime.now().isoformat()))
        conn.commit()
        conn.close()

    @classmethod
    def load_state(cls, user_id):
        """Load bot state from DB if exists."""
        conn = sqlite3.connect(cls.DB_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM bot_sessions WHERE user_id = ?", (user_id,))
        row = c.fetchone()
        conn.close()
        
        if row:
            try:
                return {
                    "config": json.loads(row['config']),
                    "balance": row['balance'],
                    "positions": json.loads(row['positions']),
                    "trade_history": json.loads(row['trade_history']),
                    "equityCurve": json.loads(row['equity_curve']),
                    "logs": json.loads(row['logs']),
                    "status": "stopped" # Always load as stopped initially
                }
            except Exception as e:
                logger.error(f"DB Load Error: {e}")
                return None
        return None

# Initialize DB on start
DatabaseHandler.init_db()

# --- REQUEST MODELS ---
class BotStartRequest(BaseModel):
    userId: str
    config: Dict[str, Any]

class BotStopRequest(BaseModel):
    userId: str

class BotClosePositionRequest(BaseModel):
    userId: str
    symbol: str

class BacktestRequest(BaseModel):
    userId: str
    symbol: str
    timeframe: str
    start_date: str 
    end_date: str   
    strategies: List[Dict[str, Any]]
    initial_capital: float = 10000.0

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🟢 ON STARTUP: (Optional) Load state logic if needed
    yield
    # 🟢 ON SHUTDOWN: Save all active bots to DB
    for user_id, bot in ACTIVE_BOTS.items():
        bot["status"] = "stopped"
        DatabaseHandler.save_state(user_id, bot)

app = FastAPI(title="NEO-V25.12 Sovereign Engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ==========================================
# 🧠 1. NEURAL PREDICTOR
# ==========================================
class NeuralPredictor:
    @staticmethod
    def get_prediction(model_id: str, df: pd.DataFrame) -> float:
        try:
            recent = df.tail(10)
            momentum = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
            model_path = f"./models/{model_id}_model.pkl"
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                return float(model.predict_proba([[momentum]])[0][1])
            base = 1.0 / (1.0 + np.exp(-momentum * 100))
            return float(min(1.0, max(0.0, base)))
        except Exception: return 0.5

# ==========================================
# 🧠 2. SHARED STRATEGY BRAIN (Full HD Logic)
# ==========================================
class StrategyBrain:
    @staticmethod
    def calculate_signals(df: pd.DataFrame, config: Dict[str, Any], l_thresh: float, s_thresh: float):
        active_thoughts, votes = [], 0
        strategies = config.get('strategies', [])
        current_price = df['close'].iloc[-1]

        # 🟢 Indicators
        ema20 = ta.ema(df['close'], 20).iloc[-1]
        ema50 = ta.ema(df['close'], 50).iloc[-1]
        ema200 = ta.ema(df['close'], 200).iloc[-1]
        bb = ta.bbands(df['close'], length=20, std=2.0)
        lower, mid, upper = bb.iloc[-1, 0], bb.iloc[-1, 1], bb.iloc[-1, 2]
        pr = int((current_price - lower) / (upper - lower) * 100)

        # 🟢 1. FULL 10 STRATEGY EVALUATION
        for strat in strategies:
            code = strat.get('code')
            params = strat.get('params', {})
            try:
                if code == "rsi_threshold":
                    rsi = ta.rsi(df['close'], 14).iloc[-1]
                    if rsi < 30: votes += 1; active_thoughts.append(f"RSI {int(rsi)} Low")
                    elif rsi > 70: votes -= 1; active_thoughts.append(f"RSI {int(rsi)} High")
                elif code == "stoch":
                    k = ta.stoch(df['high'], df['low'], df['close']).iloc[-1, 0]
                    if k < 20: votes += 1; active_thoughts.append(f"Stoch {int(k)} Low")
                    elif k > 80: votes -= 1; active_thoughts.append(f"Stoch {int(k)} High")
                elif code == "bb_fade":
                    if current_price < lower: votes += 1; active_thoughts.append(f"Price < BB Floor ${int(lower)}")
                    elif current_price > upper: votes -= 1; active_thoughts.append(f"Price > BB Ceiling ${int(upper)}")
                elif code == "sma_crossover":
                    if ta.sma(df['close'], 50).iloc[-1] > ta.sma(df['close'], 200).iloc[-1]: votes += 1; active_thoughts.append("SMA Golden Cross")
                elif code == "macd_crossover":
                    m = ta.macd(df['close']).iloc[-1]
                    if m.iloc[0] > m.iloc[2]: votes += 1; active_thoughts.append("MACD Cross Up")
                elif code == "supertrend":
                    st = ta.supertrend(df['high'], df['low'], df['close']).iloc[-1]
                    if st.iloc[1] == 1: votes += 1; active_thoughts.append("SuperTrend Bullish")
                elif code == "ema_cloud":
                    if current_price > ema50: votes += 1; active_thoughts.append("Above EMA Cloud")
                elif code == "atr_breakout":
                    atr = ta.atr(df['high'], df['low'], df['close']).iloc[-1]
                    if current_price > (ema20 + atr * 1.5): votes += 1; active_thoughts.append("ATR Breakout")
                elif code == "pa_breakout":
                    if current_price >= df['high'].tail(20).max(): votes += 1; active_thoughts.append("20-Bar High Break")
                elif code == "vol_profile":
                    if df['volume'].iloc[-1] > ta.sma(df['volume'], 20).iloc[-1] * 1.5: votes += (1 if current_price > ema20 else -1); active_thoughts.append("Volume Surge")
            except Exception: continue

        # 🚀 2. DYNAMIC DUAL-GATE LOGIC (UI Based)
        is_short = current_price < ema200
        # Dynamic limits from UI config
        ui_limit = float(config.get('mlThresholdShort', 0.90)) if is_short else float(config.get('mlThresholdLong', 0.80))
        
        conf = NeuralPredictor.get_prediction(config.get('mlModel', 'stacking'), df)
        gate_passed = conf >= ui_limit
        logic_desc = f"📊 LOGIC: {'SHORT' if is_short else 'LONG'} GATE {'PASSED' if gate_passed else 'VETOED'} ({int(conf*100)}% vs {int(ui_limit*100)}% UI Limit) {'🟢' if gate_passed else '🔴'}"

        # 🎯 3. DYNAMIC HD INTENT (Dollar Gaps)
        signal_names = " + ".join(active_thoughts) if active_thoughts else "Scanning Setup"
        gap = int(abs(current_price - ema50))
        intent_desc = f"🎯 INTENT: STALKING {'SHORT' if is_short else 'LONG'} ({signal_names} | Gap: ${gap}) {'🔴' if is_short else '🟢'}"

        # 📡 4. MARKET CONTEXT
        numeric_details = {
            "market": {
                "logic": logic_desc, "intent": intent_desc,
                "trend": f"📡 TREND: {'🚀 STRONG UPTREND' if current_price > ema50 and not is_short else '🩸 DOWNTREND'} (${int(current_price)} vs 50EMA ${int(ema50)}) {'🟢' if not is_short else '🔴'}",
                "bias": f"⚖️ BIAS: {'MOMENTUM UP' if ema20 > ema50 else 'MOMENTUM DOWN'} (Spread: ${int(abs(ema20-ema50))}) {'🟢' if ema20 > ema50 else '🔴'}",
                "mindset": f"🤖 MINDSET: {'⚠️ EXPENSIVE' if pr > 85 else '🎯 CHEAP' if pr < 15 else '⚖️ BALANCED'} ({pr}% of Range) {'🔴' if pr > 85 else '🟢' if pr < 15 else '⚪'}"
            }
        }

        final_sig = 1 if votes > 0 and not is_short and gate_passed else (-1 if votes < 0 and is_short and gate_passed else 0)
        return final_sig, active_thoughts, numeric_details, conf

# ==========================================
# 🚀 3. THE HEARTBEAT (With DB Recording)
# ==========================================
async def live_neural_heartbeat(user_id: str):
    last_log = 0
    
    # 🟢 Initialize Arrays if Fresh Start
    if user_id in ACTIVE_BOTS:
        if "equityCurve" not in ACTIVE_BOTS[user_id]: 
            ACTIVE_BOTS[user_id]["equityCurve"] = [{"time": datetime.now().isoformat(), "balance": ACTIVE_BOTS[user_id]["balance"], "confidence": 50}]
        if "logs" not in ACTIVE_BOTS[user_id]: 
            ACTIVE_BOTS[user_id]["logs"] = []

    while user_id in ACTIVE_BOTS and ACTIVE_BOTS[user_id]["status"] == "running":
        bot = ACTIVE_BOTS[user_id]
        config = bot.get('config', {})
        try:
            ohlcv_raw = await fetch_live_candles_ccxt(config['symbol'], config.get('timeframe', '1h'), 250)
            if ohlcv_raw:
                df = pd.DataFrame(ohlcv_raw)
                latest_price = df['close'].iloc[-1]
                sig, thoughts, nums, score = StrategyBrain.calculate_signals(df, config, 0.5, 0.5)
                
                # Markers for Chart
                markers = [{"time": int(c['time']), "position": "belowBar", "color": "#10b981", "shape": "circle", "text": thoughts[0] if thoughts else ""} for c in ohlcv_raw[-1:] if thoughts]

                # Unrealized PnL Calculation
                upnl = sum([(latest_price - p['entry']) * p['size'] if p['type'] == 'buy' else (p['entry'] - latest_price) * p['size'] for p in bot['positions']])
                current_equity = bot['balance'] + upnl

                # 🟢 RECORD HISTORY (Every 1 minute)
                if datetime.now().timestamp() - last_log >= 60:
                    # Save Equity & Confidence
                    bot["equityCurve"].append({
                        "time": datetime.now().isoformat(), 
                        "balance": round(current_equity, 2), 
                        "confidence": int(score * 100)
                    })
                    # Limit curve size to keep payload light
                    if len(bot["equityCurve"]) > 100: bot["equityCurve"].pop(0)

                    # Save Logs
                    new_logs = []
                    for key in ['trend', 'bias', 'mindset', 'logic', 'intent']:
                        msg = nums['market'][key]
                        emit_log(user_id, msg)
                        new_logs.append({"time": datetime.now().isoformat(), "message": msg})
                    
                    bot["logs"] = (new_logs + bot["logs"])[:50] 
                    
                    # 💾 PERSIST TO DATABASE
                    DatabaseHandler.save_state(user_id, bot)
                    
                    last_log = datetime.now().timestamp()

                # Emit Full Status
                emit_status(user_id, {
                    "status": "running", "currentBalance": round(current_equity, 2),
                    "unrealizedPnl": round(upnl, 2),
                    "activePositions": bot['positions'], 
                    "tradeMarkers": bot['trade_history'] + markers, 
                    "equityCurve": bot["equityCurve"], # Send history to frontend
                    "candles": [{"time": int(c['time']), "open": c['open'], "high": c['high'], "low": c['low'], "close": c['close']} for c in ohlcv_raw[-50:]]
                })

            await asyncio.sleep(15)
        except Exception as e: 
            logger.error(f"Sync Error: {e}"); await asyncio.sleep(10)

# ==========================================
# 📡 4. ALL ENDPOINTS
# ==========================================
async def fetch_live_candles_ccxt(symbol: str, timeframe: str, limit: int):
    async with ccxt.coinbase() as ex:
        try:
            ohlcv = await ex.fetch_ohlcv(symbol.replace('-', '/'), timeframe, limit=limit)
            return [{"time": c[0]/1000, "open": c[1], "high": c[2], "low": c[3], "close": c[4], "vol": c[5] if len(c) > 5 else 0} for c in ohlcv]
        except: return []

@app.post("/api/bot/start")
async def start_bot(data: BotStartRequest, background_tasks: BackgroundTasks):
    user_id = data.userId.strip()
    
    # 🟢 CHECK DB FOR EXISTING SESSION
    if user_id in ACTIVE_BOTS and ACTIVE_BOTS[user_id]["status"] == "running":
        return {"status": "running", "message": "Bot already active"}
    
    saved_state = DatabaseHandler.load_state(user_id)
    if saved_state:
        # Resume Session
        ACTIVE_BOTS[user_id] = saved_state
        ACTIVE_BOTS[user_id]["status"] = "running"
        ACTIVE_BOTS[user_id]["config"] = data.config # Update config
        emit_log(user_id, "♻️ SESSION RESTORED: History Loaded from Database.")
    else:
        # Start New
        capital = float(data.config.get("capitalAllocation") or 200)
        ACTIVE_BOTS[user_id] = {
            "status": "running", "config": data.config, "balance": capital, 
            "positions": [], "trade_history": [], "equityCurve": [], "logs": []
        }
        emit_log(user_id, f"🚀 Engine Started. Portfolio: ${capital}")

    background_tasks.add_task(live_neural_heartbeat, user_id)
    return {"status": "running"}

@app.post("/api/bot/stop")
async def stop_bot(data: BotStopRequest):
    if data.userId in ACTIVE_BOTS:
        ACTIVE_BOTS[data.userId]["status"] = "stopped"
        # Save state on stop
        DatabaseHandler.save_state(data.userId, ACTIVE_BOTS[data.userId])
        emit_log(data.userId, "🛑 Emergency Halt Signal Received.")
        return {"status": "stopped"}
    raise HTTPException(status_code=404)

@app.post("/api/bot/close_position")
async def close_position(data: BotClosePositionRequest):
    if data.userId in ACTIVE_BOTS and ACTIVE_BOTS[data.userId]['positions']:
        pos = ACTIVE_BOTS[data.userId]['positions'].pop(0)
        emit_log(data.userId, f"⚠️ Manual Exit: {pos['type'].upper()} closed.")
        DatabaseHandler.save_state(data.userId, ACTIVE_BOTS[data.userId])
        return {"status": "closed"}
    raise HTTPException(status_code=400, detail="No active positions")

@app.post("/api/bot/backtest")
async def run_backtest(data: BacktestRequest):
    try:
        async with ccxt.coinbase() as exchange:
            since = exchange.parse8601(data.start_date)
            ohlcv = await exchange.fetch_ohlcv(data.symbol.replace('-', '/'), data.timeframe, since=since, limit=1000)
            df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
        balance, position, trades, curve = data.initial_capital, None, [], []
        for i in range(20, len(df)):
            window = df.iloc[:i+1]
            # Use same strategy logic for backtest
            sig, _, _, _ = StrategyBrain.calculate_signals(window, {"strategies": data.strategies}, 0.5, 0.5)
            price, ts = df.iloc[i]['close'], datetime.fromtimestamp(df.iloc[i]['time']/1000, tz=timezone.utc).isoformat()
            if sig == 1 and position is None:
                position = {"type": "long", "entry": price, "size": balance / price}
                trades.append({"type": "buy", "price": price, "time": ts})
            elif sig == -1 and position is None:
                position = {"type": "short", "entry": price, "size": balance / price}
                trades.append({"type": "short", "price": price, "time": ts})
            elif (sig == -1 and position and position['type'] == 'long') or (sig == 1 and position and position['type'] == 'short'):
                pnl = (price - position['entry']) * position['size'] if position['type'] == 'long' else (position['entry'] - price) * position['size']
                balance += pnl; position = None
                trades.append({"type": "exit", "price": price, "time": ts, "pnl": pnl})
            curve.append({"time": ts, "equity": balance})
        return {"status": "success", "metrics": {"final_balance": round(balance, 2), "trade_count": len(trades)}, "trades": trades, "equity_curve": curve}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/bot/status")
async def get_status(userId: str):
    # 🟢 ENHANCED STATUS: Check memory first, then DB
    bot = ACTIVE_BOTS.get(userId.strip())
    if not bot:
        bot = DatabaseHandler.load_state(userId.strip())
    
    if bot:
        return {
            "status": bot["status"], 
            "balance": bot["balance"],
            "equityCurve": bot.get("equityCurve", []), # Allows chart hydration
            "logs": bot.get("logs", []),
            "positions": bot.get("positions", [])
        }
    return {"status": "inactive", "balance": 0}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
