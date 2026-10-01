import requests
import json
import socket

# --- CONFIGURATION ---
IONOS_IP = "74.208.28.77" # Your ML Server IP
ML_PORT = 8000
RENDER_GATEWAY = "https://neov6backend.onrender.com/api/backtest/run"
VERCEL_FRONTEND = "https://neov6.vercel.app"

def test_ml_server_internals():
    print(f"🔍 [1/3] INTERNAL: Testing FastAPI on {IONOS_IP}:{ML_PORT}...")
    local_target = f"http://127.0.0.1:{ML_PORT}/api/backtest/run"
    try:
        # We send a dummy payload to check for a 404 (Route Missing) vs 422 (Schema mismatch)
        res = requests.post(local_target, json={}, timeout=5)
        if res.status_code == 404:
            print(f"❌ FAIL: Route Not Found. Check if @app.post('/api/backtest/run') is in main3.py.")
        elif res.status_code == 422:
            print(f"✅ SUCCESS: Route is active (422 confirms the API is listening for data).")
        else:
            print(f"📡 STATUS: Server responded with {res.status_code}.")
    except Exception as e:
        print(f"❌ FAIL: ML Server is not responding on port {ML_PORT}. Is Gunicorn running?")

def test_render_to_ml_bridge():
    print(f"\n🔍 [2/3] GATEWAY: Testing Render Proxy -> ML Server...")
    try:
        # This simulates Vercel calling Render, which then calls your ML server
        res = requests.post(RENDER_GATEWAY, json={}, timeout=10)
        print(f"📡 Render Response: {res.status_code}")
        
        if res.status_code == 500:
            print("❌ FAIL: Render Gateway returned 500. It cannot reach your IONOS IP.")
            print(f"👉 Check: Is port {ML_PORT} open in IONOS firewall (ufw allow {ML_PORT}/tcp)?")
        elif "Python Server API failed" in res.text:
            print("❌ FAIL: The request reached IONOS, but Python returned an error (likely 404).")
        elif res.status_code == 404:
             print("❌ FAIL: Render itself couldn't find the /api/backtest/run route. Check your Express router.")
    except Exception as e:
        print(f"❌ FAIL: Connection to Render timed out.")

def test_ingress_visibility():
    print(f"\n🔍 [3/3] INGRESS: Checking if port {ML_PORT} is open to the internet...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex((IONOS_IP, ML_PORT))
    if result == 0:
        print(f"✅ SUCCESS: Port {ML_PORT} is OPEN and accepting external connections.")
    else:
        print(f"❌ FAIL: Port {ML_PORT} is CLOSED. External traffic (Render) will be blocked.")
    sock.close()

if __name__ == "__main__":
    print(f"--- NEO-V6 CLOUD CONNECTIVITY DEBUGGER ---")
    test_ml_server_internals()
    test_ingress_visibility()
    test_render_to_ml_bridge()
