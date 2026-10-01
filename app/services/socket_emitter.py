# File: app/utils/socket_emitter.py
# 🚀 NEW FILE: Handles communication with Node.js

import requests
import os
import asyncio

# 🟢 CONFIG: Point this to your Node.js Backend
# If running locally, use http://localhost:10000 (or whatever port server.js uses)
# If deployed, use your Render URL
import logging
logger = logging.getLogger("SocketEmitter")


def broadcast(user_id, event_type, data):
    """
    Sends a signal to the Node.js backend, which relays it via WebSocket.
    """
    if not user_id:
        return

    # Read config at CALL TIME (not import time) so it reflects the loaded
    # environment regardless of import order / when .env is loaded.
    node_backend_url = os.getenv("NODE_BACKEND_URL", "https://neov6backend.onrender.com")
    internal_api_key = os.getenv("INTERNAL_API_KEY", "")

    url = f"{node_backend_url}/api/internal/broadcast"
    payload = {
        "userId": user_id,
        "type": event_type, # 'bot_log' or 'bot_status_update'
        "data": data
    }

    headers = {"x-internal-key": internal_api_key} if internal_api_key else {}

    try:
        # FIX #21: broadcast() runs in a worker thread (see emit_log/emit_status),
        # so a generous timeout no longer stalls the trading loop. The old 0.5s
        # timeout dropped updates under any backend latency (root of the
        # "WAITING FOR SIGNAL" saga).
        resp = requests.post(url, json=payload, headers=headers, timeout=4)
        if resp.status_code != 200:
            logger.warning(
                f"⚠️ broadcast to backend rejected: HTTP {resp.status_code} "
                f"(check INTERNAL_API_KEY matches between ML and backend)"
            )
    except Exception as e:
        # FIX #21: non-fatal so trading isn't interrupted by UI lag — but log it.
        # This was a silent `pass` that hid every dropped update.
        logger.warning(f"⚠️ broadcast to backend failed: {type(e).__name__}: {e}")

# 🟢 HELPER 1: Send a Log Message (The "Thinking" Stream)
# async so callers can `await emit_log(...)`; the blocking HTTP POST runs in a
# worker thread so it never stalls the async trading loop.
async def emit_log(user_id, message):
    await asyncio.to_thread(broadcast, user_id, "bot_log", message)

# 🟢 HELPER 2: Send a Status/Balance Update
async def emit_status(user_id, status_data):
    # status_data should be a dict like: {'status': 'running', 'currentBalance': 500.0}
    await asyncio.to_thread(broadcast, user_id, "bot_status_update", status_data)

# 🟢 HELPER 3: Send a TRADE ALERT (entry/exit) -> Node backend emails the user.
# Includes the base user id so Node resolves the account even for composite fleet
# keys ("<uid>::<symbol>::<side>"). Fire-and-forget; never stalls the trading loop.
async def emit_trade_alert(user_id, data):
    base = str(user_id).split("::")[0]
    payload = {**(data or {}), "baseUserId": base}
    await asyncio.to_thread(broadcast, user_id, "trade_alert", payload)
