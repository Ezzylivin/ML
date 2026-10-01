import logging
import os
import traceback
import re
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- Project Imports ---
from app.config2 import MODEL_DIR
from app.utils2 import recursive_clean
from app.backtest2 import execute_backtest  
# Use this in both files instead of importing from each other
from app.controllers import system_controller

logger = logging.getLogger("SystemAPI")
app = FastAPI(title="NEO-V7 Sovereign Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def save_for_audit(result_data):
    try:
        # We use recursive_clean to ensure NumPy types are converted to standard Python
        clean_data = recursive_clean(result_data)
        with open('latest_results.json', 'w') as f:
            json.dump(clean_data, f, indent=4)
        logger.info("💾 Results cached to latest_results.json for Certification Audit.")
    except Exception as e:
        logger.error(f"❌ Audit Save Failed: {e}")

# 🟢 HARDENED STATUS ENDPOINT
@app.get("/backtest/status")
async def get_backtest_status():
    heartbeat_path = "engine_heartbeat.txt"
    if not os.path.exists(heartbeat_path):
        return {"progress": 0.0, "status": "Initializing Engine..."}
    
    try:
        with open(heartbeat_path, "r") as f:
            content = f.read().strip()
        
        # 🟢 REGEX: Extract '12.5' from "OPTIMIZING | 12.5% | trial #10"
        match = re.search(r"(\d+\.?\d*)%", content)
        progress_val = float(match.group(1)) if match else 0.0
        
        # Parse the status message (text after the last |)
        status_msg = content.split('|')[-1].strip() if '|' in content else "Running..."
        
        if "COMPLETE" in content:
            return {"progress": 100.0, "status": "Simulation Certified."}
            
        return {"progress": progress_val, "status": status_msg}
    except Exception:
        return {"progress": 0.0, "status": "Synchronizing Link..."}


@app.post('/backtest/run')
@app.post('/backtest/run/')      # 🛡️ Catches Render/Nginx slash variants
@app.post('/backtest/combo')
@app.post('/ml/run-backtest-on')
async def run_backtest(request: Request):
    try:
        data = await request.json()
        result = await system_controller.run_backtest_logic(**data)
        save_for_audit(result)
        return JSONResponse(content=recursive_clean(result))
    except Exception as e:
        return JSONResponse(content={"status": "error", "error": f"Payload Fault: {str(e)}"}, status_code=200)

@app.get("/ml/available-models")
def list_models():
    if not os.path.exists(MODEL_DIR): return {"status": "success", "models": []}
    models = [{"id": f.rsplit('.', 1)[0], "name": f.rsplit('.', 1)[0]} 
              for f in os.listdir(MODEL_DIR) if f.endswith(('.keras', '.joblib'))]
    return {"status": "success", "models": models}
