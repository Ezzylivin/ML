# File: app/utils/socket_emitter.py
# 🚀 NEW FILE: Handles communication with Node.js

import requests
import os

# 🟢 CONFIG: Point this to your Node.js Backend
# If running locally, use http://localhost:10000 (or whatever port server.js uses)
# If deployed, use your Render URL
NODE_BACKEND_URL = os.getenv("NODE_BACKEND_URL", "https://neov6backend.onrender.com") 

def broadcast(user_id, event_type, data):
    """
    Sends a signal to the Node.js backend, which relays it via WebSocket.
    """
    if not user_id:
        return

    url = f"{NODE_BACKEND_URL}/api/internal/broadcast"
    payload = {
        "userId": user_id,
        "type": event_type, # 'bot_log' or 'bot_status_update'
        "data": data
    }

    try:
        # Timeout is fast so the trading bot doesn't hang waiting for the UI
        requests.post(url, json=payload, timeout=0.5)
    except Exception as e:
        # Silent fail is preferred here so trading isn't interrupted by UI lag
        pass

# 🟢 HELPER 1: Send a Log Message (The "Thinking" Stream)
def emit_log(user_id, message):
    broadcast(user_id, "bot_log", message)

# 🟢 HELPER 2: Send a Status/Balance Update
def emit_status(user_id, status_data):
    # status_data should be a dict like: {'status': 'running', 'currentBalance': 500.0}
    broadcast(user_id, "bot_status_update", status_data)
(venv) root@intelligent-mendel:~/Project/ML# 
