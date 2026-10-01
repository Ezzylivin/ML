import logging
import json
import os
import traceback
import copy
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import asyncio
import joblib
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from datetime import datetime
from pydantic import BaseModel

# --- Project Imports ---
from .config import bots_lock, active_live_bots, bots_collection, LOG_DIR, RESULTS_DIR, MODEL_DIR, OPTIMIZER_DIR
from .utils import recursive_clean, normalize_positions
from .live_bot import LiveExecutiveBot 
from .manager import PrecisionPyramidManager
from .strategies import compute_signal, aggregate_signals
from .backtest import execute_backtest  # Our upgraded engine

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SystemAPI")

app = FastAPI(title="Sovereign Executive v93.1")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS ---
class LoginRequest(BaseModel):
    email: str
    password: str

# --- SYSTEM CONTROLLER (The Brain) ---
class SystemController:
    def __init__(self):
        self._results_path = OPTIMIZER_DIR

    def _clean_for_json(self, data):
        return recursive_clean(data)

    async def run_backtest_logic(self, **kwargs):
        """
        UPGRADED: Unified entry point for all backtests. 
        Explicitly handles the new Trend Regime and ATR-Risk Handshake.
        """
        symbol = kwargs.get('symbol', 'BTC-USD')
        logger.info(f"📉 API Request: Starting Regime-Aware Backtest for {symbol}")
        
        # 1. Normalize Capital & Risk (Handshake from UI)
        initial_balance = kwargs.pop('initialBalance', kwargs.get('initial_balance', 1000.0))
        kwargs['initial_balance'] = float(initial_balance)
        
        # 🟢 UPGRADE: Capture snake_case risk from Upgraded UI
        if 'risk_percentage' in kwargs:
            kwargs['risk_percentage'] = float(kwargs['risk_percentage'])
        elif 'riskPercentage' in kwargs:
            kwargs['risk_percentage'] = float(kwargs.pop('riskPercentage'))

        # 🟢 UPGRADE: Explicit Regime Logic Normalization
        # Ensures ADX and SMA parameters reach the backend manager
        params = kwargs.get('params', {})
        regime_config = {
            "minAdxLevel": params.get('minAdxLevel', 20),
            "trendFilterPeriod": params.get('trendFilterPeriod', 200),
            "tslAtrMult": params.get('tslAtrMult', 3.0),
            "minAtrPct": params.get('minAtrPct', 0.5)
        }
        kwargs['params'] = regime_config

        try:
            # 2. Run the verified regime-aware engine
            if asyncio.iscoroutinefunction(execute_backtest):
                raw_result = await execute_backtest(**kwargs)
            else:
                raw_result = await asyncio.to_thread(execute_backtest, **kwargs)
            
            # 3. Ensure the response structure is consistent
            return self._clean_for_json(raw_result)
        except Exception as e:
            logger.error(f"❌ Backtest Engine Error: {traceback.format_exc()}")
            return {"metrics": {"roi": 0}, "candleData": [], "trades": [], "equityCurve": []}

    async def get_optimizer_winners(self):
        winners = []
        if not os.path.exists(self._results_path): return []
        try:
            files = [f for f in os.listdir(self._results_path) if f.endswith(".json")]
            for filename in files:
                with open(os.path.join(self._results_path, filename), 'r') as f:
                    data = json.load(f)
                    metrics = data.get('metrics', {})
                    roi = metrics.get('roi', metrics.get('totalReturn', 0))
                    winners.append({
                        "id": filename.replace('.json', ''),
                        "symbol": data.get('symbol', 'BTC-USD'),
                        "roi": float(roi),
                        "config": data.get("config", {}) or data
                    })
            winners.sort(key=lambda x: x['roi'], reverse=True)
            return winners[:20]
        except Exception as e:
            logger.error(f"Winner fetch failed: {e}")
            return []

system_controller = SystemController()

# --- 🤖 BOT MANAGEMENT ENDPOINTS ---

@app.on_event("startup")
async def startup_event():
    logger.info("System Startup - Resurrecting Active Bots...")
    with bots_lock:
        try:
            running_bots = bots_collection.find({"status": "running"})
            for doc in running_bots:
                bot_id = doc['botId']
                if bot_id not in active_live_bots:
                    bot = LiveExecutiveBot(doc['config'])
                    active_live_bots[bot_id] = bot
                    import threading
                    t = threading.Thread(target=bot.sync_and_trade)
                    t.daemon = True
                    t.start()
                    logger.info(f"✅ Resurrected Bot: {bot_id}")
        except Exception as e:
            logger.error(f"Failed to query database for resurrection: {e}")

@app.get("/")
def health_check(): 
    return {"status": "online", "version": "93.1", "server_time": datetime.now().isoformat()}

@app.post("/api/users/login")
async def login(req: LoginRequest):
    return {"token": "v93-sovereign-executive-jwt", "user": {"id": "0xUser", "email": req.email}}

@app.post('/api/bot/start')
async def start_bot(request: Request, bg: BackgroundTasks):
    try:
        body = await request.json()
        symbol = body.get('symbol', 'BTC-USD').replace('/', '-')
        bot_id = f"{body.get('userId', 'anon')}_{symbol}"
        
        with bots_lock:
            if bot_id in active_live_bots:
                return {"status": "already_running", "botId": bot_id}
            
            bot = LiveExecutiveBot(body)
            active_live_bots[bot_id] = bot
            bg.add_task(bot.sync_and_trade)
            
        return {"status": "started", "botId": bot_id}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

@app.post('/api/bot/stop')
async def stop_bot(request: Request):
    try:
        body = await request.json()
        user_id = body.get('userId')
        with bots_lock:
            for bid, bot in list(active_live_bots.items()): 
                if user_id in bid:
                    bot.stop()
                    del active_live_bots[bid]
        return {"status": "stopped"}
    except: return {"status": "error"}

@app.get('/api/bot/status')
async def get_bot_status(userId: str):
    for bid, bot in list(active_live_bots.items()):
        if userId in bid:
            return JSONResponse(content=jsonable_encoder(recursive_clean({
                "status": "running", 
                "symbol": bot.symbol, 
                "currentBalance": bot.manager.current_equity,
                "trades": bot.manager.trades,
                "equityCurve": bot.manager.get_equity_curve()
            })))
    return JSONResponse({"status": "stopped"})

@app.get("/api/bot/winners")
async def get_winners():
    winners = await system_controller.get_optimizer_winners()
    return winners

# --- 🚀 UNIFIED REGIME-AWARE BACKTEST ENDPOINT ---

@app.post('/api/backtest/run')
@app.post('/api/backtest/combo')
@app.post('/api/ml/run-backtest-on') 
async def handle_backtest(request: Request):
    """
    Unified entry point for all simulation requests.
    Enforces a strict 'CombinedResult' wrapper required by React Recharts.
    """
    try:
        config = await request.json()

        # 1. 🟢 NORMALIZE STRATEGIES LIST
        # Ensure 'strategies' is always a list for the engine, even for single runs.
        if 'strategies' not in config or not config['strategies']:
            config['strategies'] = [{
                "strategyId": config.get('strategyId'),
                "code": config.get('code'),
                "params": config.get('params', {})
            }]

        # 2. 🟢 EXECUTE LOGIC
        # Call the audited backtest logic controller
        result = await system_controller.run_backtest_logic(**config)

        # 3. 🟢 VALIDATE & WRAP RESPONSE
        # The React frontend explicitly searches for 'combinedResult' to draw charts.
        if isinstance(result, dict):
            # Check for technical errors from the engine first
            if "error" in result:
                return JSONResponse(status_code=400, content=result)

            # Mirror mandatory keys into the structure React expects
            final_payload = {
                "combinedResult": result,
                "metrics": result.get("metrics", {}),
                "trades": result.get("trades", []),
                "equityCurve": result.get("equityCurve", []),
                "candleData": result.get("candleData", [])
            }
            
            # Use jsonable_encoder to safely handle NumPy/Pandas objects
            return JSONResponse(content=jsonable_encoder(final_payload))
        
        # Fallback if result is malformed
        return JSONResponse(
            status_code=500, 
            content={"message": "Engine returned malformed result object."}
        )

    except Exception as e:
        # Capture the full trace in your logs for debugging
        logger.error(f"❌ BACKTEST HANDSHAKE FAILED: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500, 
            content={"message": f"Python Server Error: {str(e)}"}
        )

@app.get("/api/ml/available-models")
def list_models():
    if not os.path.exists(MODEL_DIR): return []
    return [{"id": f.split('.')[0], "name": f.split('.')[0]} for f in os.listdir(MODEL_DIR) if f.endswith('.joblib')]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
