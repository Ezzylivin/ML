import uvicorn
import logging
import copy
import traceback
import os
import json  # <--- 🚨 CRITICAL FIX: Was missing, causing the silent failure
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from datetime import datetime

from core.config import bots_lock, active_live_bots, bots_collection, LOG_DIR, RESULTS_DIR, MODEL_DIR
from core.utils import recursive_clean, normalize_positions, generate_chart_markers
from core.bot import LiveExecutiveBot

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("System")

app = FastAPI(title="Sovereign Executive v90.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup_event():
    logger.info("System Startup - Resurrecting Bots...")
    with bots_lock:
        running_bots = bots_collection.find({"status": "running"})
        for doc in running_bots:
            try:
                if 'config' in doc:
                    bot_id = doc['botId']
                    if bot_id not in active_live_bots:
                        bots_collection.update_one({"botId": bot_id}, {"$unset": {"stoppedAt": ""}})
                        bot = LiveExecutiveBot(doc['config'])
                        active_live_bots[bot_id] = bot
                        import threading
                        t = threading.Thread(target=bot.sync_and_trade)
                        t.daemon = True
                        t.start()
            except: pass

@app.get("/")
def health_check(): return {"status": "online", "version": "90.1"}

@app.post('/api/bot/start')
async def start_bot(request: Request, bg: BackgroundTasks):
    try:
        body = await request.json()
        bot_id = body.get('userId') + "_" + body.get('symbol', 'BTC-USD').replace('/','-')
        body['botId'] = bot_id
        
        with bots_lock:
            if bot_id in active_live_bots: 
                return JSONResponse(content=jsonable_encoder(recursive_clean({"status": "running", "botId": bot_id})))
            
            bot = LiveExecutiveBot(body)
            active_live_bots[bot_id] = bot
            bg.add_task(bot.sync_and_trade)
            
        return {"status": "started", "botId": bot_id}
    except Exception as e: return JSONResponse({"error": str(e)}, 500)

@app.post('/api/bot/stop')
async def stop_bot(request: Request):
    try:
        body = await request.json()
        target_id = body.get('userId')
        count = 0
        with bots_lock:
            to_remove = []
            for bid, bot in list(active_live_bots.items()): 
                if target_id in bid:
                    bot.stop()
                    to_remove.append(bid)
                    count += 1
            for bid in to_remove: del active_live_bots[bid]
        return {"status": "stopped", "count": count}
    except: return {"status": "error"}

@app.post('/api/bot/reset')
async def reset_bot(request: Request):
    try:
        body = await request.json()
        target_id = body.get('userId')
        with bots_lock:
            to_remove = [bid for bid in list(active_live_bots.items()) if target_id in bid[0]] 
            for bid in to_remove: 
                active_live_bots[bid[0]].stop()
                del active_live_bots[bid[0]]
        bots_collection.delete_many({"botId": {"$regex": f"^{target_id}"}})
        return {"status": "reset"}
    except: return {"error": "reset failed"}

@app.get('/api/bot/status')
async def get_bot_status(botId: str = None, userId: str = None):
    try:
        target_bot = None
        if botId: 
            target_bot = active_live_bots.get(botId)
        elif userId:
            for bid, bot in list(active_live_bots.items()):
                if userId in bid: target_bot = bot; break
        
        if target_bot:
            current_equity = target_bot.manager.current_equity
            initial_capital = target_bot.manager.initial_capital
            positions = normalize_positions(target_bot.manager.positions) 
            trades = copy.deepcopy(target_bot.manager.trades)
            candles = copy.deepcopy(target_bot.latest_candles)
            total_profit = current_equity - initial_capital
            active_position = positions[0] if positions else None

            response = {
                "status": "running",
                "symbol": target_bot.symbol,
                "timeframe": target_bot.tf,
                "currentBalance": current_equity,
                "activePositions": positions, 
                "activePosition": active_position, 
                "tradeHistory": trades,
                "candles": candles,
                "performanceMetrics": {
                    "totalTrades": len(trades),
                    "netProfit": total_profit
                },
                "chartMarkers": generate_chart_markers(trades, positions),
            }
            return JSONResponse(content=jsonable_encoder(recursive_clean(response)))

        else:
            lookup_id = botId if botId else (userId if userId else None)
            if not lookup_id: return JSONResponse({"error": "ID required"}, 400)
            
            doc = bots_collection.find_one({"botId": {"$regex": f"^{lookup_id}"}})
            if doc:
                active_list = doc.get('activePositions', [])
                active_pos = doc.get('activePosition') or (active_list[0] if active_list else None)

                response = {
                    "status": doc.get('status', 'stopped'), 
                    "currentBalance": doc.get('currentBalance', 0),
                    "activePositions": active_list,
                    "activePosition": active_pos, 
                    "tradeHistory": doc.get('tradeHistory', []),
                    "candles": [],
                    "performanceMetrics": {"netProfit": 0},
                    "chartMarkers": [],
                }
                return JSONResponse(content=jsonable_encoder(recursive_clean(response)))
            
            return JSONResponse({"status": "not_found"}, 404)

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=200)

@app.get('/api/bot/logs')
async def get_bot_logs(botId: str = None, userId: str = None):
    try:
        lookup_id = botId if botId else (userId if userId else None)
        if not lookup_id: return []
        
        doc = bots_collection.find_one({"botId": {"$regex": f"^{lookup_id}"}})
        if doc and 'logs' in doc:
            sorted_logs = sorted(doc['logs'], key=lambda x: x['timestamp'], reverse=True)
            return sorted_logs[:200]
        return []
    except: return []

# 🚀 FIXED ENDPOINT
@app.get("/api/bot/winners")
def get_winners():
    winners = []
    # 1. Check if Directory Exists
    if not os.path.exists(RESULTS_DIR): 
        logger.warning(f"Winners directory not found at: {RESULTS_DIR}")
        return []
        
    for f in os.listdir(RESULTS_DIR):
        if f.endswith(".json"):
            try:
                with open(os.path.join(RESULTS_DIR, f), 'r') as file:
                    data = json.load(file)
                    
                    # 2. Normalize Data Structure
                    config_data = {"strategies": data} if isinstance(data, list) else data
                    
                    # 3. Ensure Metrics exist for sorting
                    if 'metrics' not in config_data:
                        config_data['metrics'] = {"totalReturn": 0}

                    id_val = f.replace('.json', '')
                    winners.append({"id": id_val, "name": id_val, "config": config_data})
            except Exception as e:
                logger.error(f"Error loading winner file {f}: {e}")
                pass
                
    # 4. Sort by Profit (Descending)
    winners.sort(key=lambda x: x['config']['metrics'].get('totalReturn', 0), reverse=True)
    return winners

@app.post('/api/backtest/combo')
@app.post('/api/backtest/run')
async def handle_backtest(request: Request):
    return JSONResponse(content={"combinedResult": {"metrics": {}, "equityCurve": []}})

@app.get("/api/ml/available-models")
def list_models():
    if not os.path.exists(MODEL_DIR): return []
    return [{"id": f.split('.')[0], "name": f.split('.')[0]} for f in os.listdir(MODEL_DIR) if f.endswith('.joblib')]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
