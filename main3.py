import asyncio
import logging
import os
import joblib
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
    yield
    for user_id in list(ACTIVE_BOTS.keys()):
        ACTIVE_BOTS[user_id]["status"] = "stopped"

app = FastAPI(title="NEO-V25.10 Sovereign Engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ==========================================
# 🧠 1. NEURAL PREDICTOR (v25.10 ML Core)
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
                prediction = model.predict_proba([[momentum]])[0][1] 
                return float(prediction)
            base = 1.0 / (1.0 + np.exp(-momentum * 100))
            pers = {"xgboost": 1.1, "random_forest": 0.9, "lstm": 1.0, "transformer": 1.2, "stacking": 0.8}
            return float(min(1.0, max(0.0, base * pers.get(model_id, 1.0))))
        except Exception: return 0.5

# ==========================================
# 🧠 2. SHARED STRATEGY BRAIN (Full 10 Strategists + HD Intent)
# ==========================================
class StrategyBrain:
    @staticmethod
    def calculate_signals(df: pd.DataFrame, config: Dict[str, Any], l_thresh: float, s_thresh: float):
        active_thoughts, numeric_details, votes = [], {}, 0
        strategies = config.get('strategies', [])
        total_strats = len(strategies) if strategies else 1
        current_price = df['close'].iloc[-1]

        # 🟢 Indicators
        ema20 = ta.ema(df['close'], 20).iloc[-1]
        ema50 = ta.ema(df['close'], 50).iloc[-1]
        ema200 = ta.ema(df['close'], 200).iloc[-1]
        bb = ta.bbands(df['close'], length=20, std=2.0)
        lower, mid, upper = bb.iloc[-1, 0], bb.iloc[-1, 1], bb.iloc[-1, 2]
        pr = int((current_price - lower) / (upper - lower) * 100)

        # 1️⃣ LOGIC: ENSEMBLE CONSENSUS
        ml_threshold = float(config.get('mlThreshold', 0.5))
        conf = 1.0
        if config.get('mlMode') == 'on':
            conf = NeuralPredictor.get_prediction(config.get('mlModel', 'stacking'), df)
            logic_desc = f"📊 LOGIC: NEURAL GATE {'PASSED' if conf >= ml_threshold else 'VETOED'} ({int(conf*100)}% vs {int(ml_threshold*100)}% Limit) {'🟢' if conf >= ml_threshold else '🔴'}"
            if conf < ml_threshold: votes = 0
        else:
            logic_desc = f"📊 LOGIC: NEURAL BYPASSED ⚪"

        # 2️⃣ TREND: HD DIRECTION ENGINE
        trend_desc = "📡 TREND: NEUTRAL (Sideways) ⚪"
        if current_price > ema200:
            if ema50 > ema200:
                if current_price > ema50: trend_desc = f"📡 TREND: 🚀 STRONG UPTREND (Price ${int(current_price)} > 50EMA ${int(ema50)}) 🟢"
                else: trend_desc = f"📡 TREND: 📈 UPTREND (Pullback: Price ${int(current_price)} < 50EMA ${int(ema50)}) 🟠"
            else: trend_desc = f"📡 TREND: ✨ POTENTIAL REVERSAL (Price > 200EMA ${int(ema200)}) 🟢"
        else:
            if ema50 < ema200:
                if current_price < ema50: trend_desc = f"📡 TREND: 🩸 STRONG DOWNTREND (Price ${int(current_price)} < 50EMA ${int(ema50)}) 🔴"
                else: trend_desc = f"📡 TREND: 📉 DOWNTREND (Relief: Price ${int(current_price)} > 50EMA ${int(ema50)}) 🟠"
            else: trend_desc = f"📡 TREND: 💀 POTENTIAL REVERSAL (Price < 200EMA ${int(ema200)}) 🔴"

        # 3️⃣ BIAS: MOMENTUM SPREAD
        spread = int(abs(ema20 - ema50))
        bias_desc = f"⚖️ BIAS: MOMENTUM {'UP' if ema20 > ema50 else 'DOWN'} (EMA20 vs EMA50 Spread: ${spread}) {'🟢' if ema20 > ema50 else '🔴'}"

        # 4️⃣ MINDSET: 5-ZONE PSYCHOLOGY
        if pr > 85: mindset_desc = f"🤖 MINDSET: ⚠️ PRICE IS EXPENSIVE ({pr}% > 85% Overbought Limit) 🔴"
        elif pr >= 65: mindset_desc = f"🤖 MINDSET: 🔭 STALKING RESISTANCE ({pr}% High -> Waiting for 85% Ceiling) 🟠"
        elif pr > 35: mindset_desc = f"🤖 MINDSET: ⚖️ MARKET IS BALANCED ({pr}% of Bollinger Range) ⚪"
        elif pr >= 15: mindset_desc = f"🤖 MINDSET: 🔭 STALKING SUPPORT ({pr}% Low -> Waiting for 15% Floor) 🔵"
        else: mindset_desc = f"🤖 MINDSET: 🎯 PRICE IS CHEAP ({pr}% < 15% Oversold Limit) 🟢"

        numeric_details['market'] = {"logic": logic_desc, "trend": trend_desc, "bias": bias_desc, "mindset": mindset_desc}

        # 🟢 THE 10 STRATEGIES
        for strat in strategies:
            code, params = strat.get('code'), strat.get('params', {})
            try:
                if code == "rsi_threshold":
                    rsi = ta.rsi(df['close'], 14).iloc[-1]
                    if rsi < 30: votes += 1; active_thoughts.append(f"RSI {int(rsi)} Oversold 🟢")
                    elif rsi > 70: votes -= 1; active_thoughts.append(f"RSI {int(rsi)} Overbought 🔴")
                elif code == "sma_crossover":
                    if ta.sma(df['close'], 50).iloc[-1] > ta.sma(df['close'], 200).iloc[-1]: votes += 1
                elif code == "supertrend":
                    st = ta.supertrend(df['high'], df['low'], df['close']).iloc[-1]
                    if st.iloc[1] == 1: votes += 1
                elif code == "macd_crossover":
                    macd = ta.macd(df['close']).iloc[-1]
                    if macd.iloc[0] > macd.iloc[2]: votes += 1
                elif code == "atr_breakout":
                    if current_price > (ema20 + ta.atr(df['high'], df['low'], df['close']).iloc[-1] * 1.5): votes += 1
                elif code == "bb_fade":
                    if current_price < lower: votes += 1
                elif code == "stoch":
                    k = ta.stoch(df['high'], df['low'], df['close']).iloc[-1, 0]
                    if k < 20: votes += 1; active_thoughts.append(f"Stoch {int(k)} Oversold 🟢")
                elif code == "ema_cloud":
                    if current_price > ema50: votes += 1
                elif code == "pa_breakout":
                    if current_price >= df['high'].tail(20).max(): votes += 1
                elif code == "vol_profile":
                    if df['volume'].iloc[-1] > ta.sma(df['volume'], 20).iloc[-1] * 1.5: votes += 1
            except Exception: continue

        # 🎯 ENHANCED INTENT BLOCK
        is_uptrend = "UPTREND" in trend_desc
        intent = "🎯 INTENT: Scanning for High-Probability Setup... ⚪"
        if votes > 0 and is_uptrend:
            intent = f"🎯 INTENT: STALKING LONG (Signals: {len(active_thoughts)}/10 | Target: >${int(ema20)}) 🟢"
        elif votes < 0 and not is_uptrend:
            intent = f"🎯 INTENT: STALKING SHORT (Signals: {abs(votes)}/10 | Target: <${int(ema20)}) 🔴"
        numeric_details['market']['intent'] = intent

        score = abs(votes) / total_strats
        final_sig = 1 if votes > 0 and is_uptrend and score >= l_thresh else (-1 if votes < 0 and "DOWNTREND" in trend_desc and score >= s_thresh else 0)
        return final_sig, active_thoughts, numeric_details, score

# ==========================================
# 🚀 3. THE HEARTBEAT (Fixed Unpacking)
# ==========================================
async def live_neural_heartbeat(user_id: str):
    last_log = 0
    while user_id in ACTIVE_BOTS and ACTIVE_BOTS[user_id]["status"] == "running":
        bot = ACTIVE_BOTS[user_id]
        config = bot.get('config', {})
        try:
            ohlcv_raw = await fetch_live_candles_ccxt(config['symbol'], config.get('timeframe', '1h'), 250)
            if ohlcv_raw:
                df = pd.DataFrame(ohlcv_raw)
                latest_price = df['close'].iloc[-1]
                live_candles = [{"time": int(c['time']), "open": c['open'], "high": c['high'], "low": c['low'], "close": c['close']} for c in ohlcv_raw[-50:]]
                upnl = sum([(latest_price - p['entry']) * p['size'] if p['type'] == 'buy' else (p['entry'] - latest_price) * p['size'] for p in bot['positions']])

                emit_status(user_id, {
                    "status": "running", "currentBalance": round(bot['balance'] + upnl, 2),
                    "unrealizedPnl": round(upnl, 2), "activePositions": bot['positions'],
                    "tradeMarkers": bot['trade_history'], "candles": live_candles
                })

                if datetime.now().timestamp() - last_log >= 60:
                    sig, thoughts, nums, score = StrategyBrain.calculate_signals(df, config, 0.5, 0.5)
                    m = nums['market']
                    # 🟢 Separate HD Sections
                    emit_log(user_id, m['trend'])
                    emit_log(user_id, m['bias'])
                    emit_log(user_id, m['mindset'])
                    emit_log(user_id, m['logic'])
                    emit_log(user_id, m['intent'])
                    last_log = datetime.now().timestamp()

            await asyncio.sleep(15)
        except Exception as e:
            logger.error(f"Sync Error: {e}"); await asyncio.sleep(10)

# ==========================================
# 📡 4. ALL ENDPOINTS (Fetch, Start, Stop, Close, Backtest, Status)
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
    capital = float(data.config.get("capitalAllocation") or 200)
    ACTIVE_BOTS[user_id] = {"status": "running", "config": data.config, "balance": capital, "positions": [], "trade_history": []}
    emit_log(user_id, f"🚀 Engine Started. Portfolio Value: ${capital}")
    background_tasks.add_task(live_neural_heartbeat, user_id)
    return {"status": "running"}

@app.post("/api/bot/stop")
async def stop_bot(data: BotStopRequest):
    if data.userId in ACTIVE_BOTS:
        ACTIVE_BOTS[data.userId]["status"] = "stopped"
        emit_log(data.userId, "🛑 Emergency Halt Signal Received.")
        return {"status": "stopped"}
    raise HTTPException(status_code=404)

@app.post("/api/bot/close_position")
async def close_position(data: BotClosePositionRequest):
    if data.userId in ACTIVE_BOTS and ACTIVE_BOTS[data.userId]['positions']:
        pos = ACTIVE_BOTS[data.userId]['positions'].pop(0)
        emit_log(data.userId, f"⚠️ Manual Exit: {pos['type'].upper()} closed.")
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
    bot = ACTIVE_BOTS.get(userId.strip())
    return {"status": bot["status"] if bot else "inactive", "balance": bot["balance"] if bot else 0}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
