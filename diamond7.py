# File: /root/Project/ML/diamond.py
# 🚀 UPGRADE: v83.1 - Multi-Bot Identity Fix
# 🛠 Fixes: botId Scope, Multi-Bot Status, Safe Stop/Logs Routing

# --- [imports unchanged] ---
import os, json, logging, traceback, math, threading, warnings, joblib, time, sqlite3
import pandas as pd
import numpy as np
import ccxt
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas_ta as ta
from datetime import datetime, timezone, timedelta

# --- [config unchanged] ---

# --- 6. ENDPOINTS ---

@app.post('/api/bot/start')
async def start_live_bot(request: Request, bg: BackgroundTasks):
    try:
        body = await request.json()
        user_id = body.get('userId')
        bot_id = body.get('botId')

        if not user_id or not bot_id:
            return JSONResponse({"error": "Missing userId or botId"}, 400)

        with bots_lock:
            if bot_id in active_live_bots:
                return JSONResponse({"error": "Bot already active"}, 400)

            try:
                bot = LiveExecutiveBot(body)
                active_live_bots[bot_id] = bot
            except Exception as init_err:
                logger.error(f"❌ Bot Init Failed: {init_err}")
                return JSONResponse({"error": str(init_err)}, 500)

        bg.add_task(bot.sync_and_trade)
        return {"status": "started", "botId": bot_id}

    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse({"error": str(e)}, 500)


@app.post('/api/bot/stop')
async def stop_live_bot(request: Request):
    body = await request.json()
    bot_id = body.get('botId')

    if not bot_id:
        return JSONResponse({"error": "Missing botId"}, 400)

    with bots_lock:
        if bot_id in active_live_bots:
            active_live_bots[bot_id].stop()
            del active_live_bots[bot_id]
            return {"status": "stopped", "botId": bot_id}

    return JSONResponse({"error": "Bot not found"}, 404)


@app.get('/api/bot/status')
async def get_bot_status(botId: str = None):
    if not botId:
        return JSONResponse({"error": "botId required"}, 400)

    bot = active_live_bots.get(botId)

    if not bot:
        rows = execute_db_with_retry(
            "SELECT cash FROM bots WHERE bot_id=?", (botId,)
        )
        if rows:
            return {
                "status": "stopped",
                "currentBalance": rows[0][0],
                "active": False
            }
        return JSONResponse({"status": "not_found"}, 404)

    # --- Live Metrics ---
    trades = bot.manager.trades
    total_trades = len(trades)
    wins = len([t for t in trades if t['profit'] > 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    max_dd = max(bot.manager.dd_series) * 100 if bot.manager.dd_series else 0

    return {
        "status": "running",
        "botId": botId,
        "symbol": bot.symbol,
        "timeframe": bot.tf,
        "currentBalance": bot.manager.current_equity,
        "positions": bot.manager.positions,
        "trades": trades,
        "performanceMetrics": {
            "totalTrades": total_trades,
            "winRate": round(win_rate, 2),
            "netProfit": bot.manager.current_equity - bot.manager.initial_capital,
            "maxDrawdown": round(max_dd, 2),
            "sharpeRatio": 0
        }
    }


@app.get('/api/bot/logs')
async def get_bot_logs(botId: str = None):
    if not botId:
        return JSONResponse({"error": "botId required"}, 400)

    log_path = os.path.join(LOG_DIR, f"{botId}.log")
    if not os.path.exists(log_path):
        return []

    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()[-100:]

        parsed = []
        for line in lines:
            parts = line.strip().split(' | ')
            if len(parts) >= 3:
                parsed.append({
                    "timestamp": parts[0],
                    "type": parts[1].lower(),
                    "message": parts[2]
                })
        return parsed[::-1]

    except Exception:
        return []

# --- [BACKTEST SECTION UNCHANGED] ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
