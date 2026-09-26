# File: app/utils/socket_emitter.py
# 🚀 NEW FILE: Handles communication with Node.js

import requests
import os
import asyncio
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
        # Timeout is fast so the trading bot doesn't hang waiting for the UI
        resp = requests.post(url, json=payload, headers=headers, timeout=0.5)
        if resp.status_code != 200:
            # Surface delivery failures (e.g. 401 = INTERNAL_API_KEY mismatch)
            # so this is diagnosable instead of silently swallowed.
            logger.warning(
                f"⚠️ broadcast to backend rejected: HTTP {resp.status_code} "
                f"(check INTERNAL_API_KEY matches between ML and backend)"
            )
    except Exception:
        # Network errors are non-fatal so trading isn't interrupted by UI lag
        pass

# 🟢 HELPER 1: Send a Log Message (The "Thinking" Stream)
# async so callers can `await emit_log(...)`; the blocking HTTP POST runs in a
# worker thread so it never stalls the async trading loop.
async def emit_log(user_id, message):
    await asyncio.to_thread(broadcast, user_id, "bot_log", message)

# 🟢 HELPER 2: Send a Status/Balance Update
async def emit_status(user_id, status_data):
    # status_data should be a dict like: {'status': 'running', 'currentBalance': 500.0}
    await asyncio.to_thread(broadcast, user_id, "bot_status_update", status_data)
