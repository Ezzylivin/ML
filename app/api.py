import logging
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import os
# Import the controller singleton
from .controller import system_controller

# Initialize FastAPI
app = FastAPI(title="Sovereign Executive v89.42 - API Interface")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Setup API Logger
logger = logging.getLogger("API")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- Pydantic Models ---

class StartBotRequest(BaseModel):
    userId: str
    symbol: str = Field(default="BTC-USD", description="Trading symbol")
    strategies: list = Field(default=[], description="List of strategies")
    initialBalance: float = Field(default=1000.0)
    class Config:
        extra = "allow"

class StopBotRequest(BaseModel):
    userId: str

class BacktestRequest(BaseModel):
    symbol: str = Field(default="BTC-USD", description="Trading pair")
    timeframe: str = Field(default="1h", description="Candle timeframe")
    initialBalance: float = Field(default=1000.0, description="Start balance")
    initial_capital: float = Field(default=1000.0, description="Backend compat")
    limit: int = Field(default=1000, description="Number of candles to fetch")
    risk_percentage: float = Field(default=1.0, description="Risk per trade %")
    risk_mode: str = Field(default="static", description="Risk mode: static or dynamic")
    
    # 🟢 CRITICAL FIX: Allow strategies, comboConfig, mlMode, etc.
    class Config:
        extra = "allow" 

# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "online", "role": "api_gateway"}

@app.post('/api/bot/start')
async def start_bot(request: StartBotRequest, bg: BackgroundTasks):
    try:
        # Convert to dict to pass all extra fields
        data = request.dict()
        bot_id, mode = system_controller.start_bot_logic(
            data.get('userId'), 
            data.get('symbol', 'BTC-USD'), 
            data.get('initialBalance', 1000), 
            data.get('strategies', []), 
            bg
        )
        return {"status": "started", "botId": bot_id, "mode": mode}
    except Exception as e:
        logger.error(f"Start API Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post('/api/bot/stop')
async def stop_bot(request: StopBotRequest):
    try:
        stopped_bots = system_controller.stop_bot_logic(request.userId)
        return {"status": "stopped", "stopped_bots": stopped_bots}
    except Exception as e:
        logger.error(f"Stop API Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get('/api/bot/status')
async def get_bot_status(botId: str = None, userId: str = None):
    try:
        status = system_controller.get_bot_status_logic(botId, userId)
        if status:
            return status
        return JSONResponse(content={"status": "not_active"}, status_code=404)
    except Exception as e:
        logger.error(f"Status API Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 🚀 UPGRADED BACKTEST ENDPOINT
@app.post('/api/backtest/combo')
@app.post('/api/ml/run-backtest-on')
@app.post('/api/ml/run-combo-backtest')
async def run_backtest(req: BacktestRequest):
    try:
        # 1. Convert Pydantic object to dictionary (includes 'strategies', 'comboConfig', etc.)
        data = req.dict()
        print(f"🔹 API RECEIVED DATA: Keys: {list(data.keys())}")

        # 2. Fix Variable Naming (Frontend 'initialBalance' vs Backend 'initial_capital')
        if 'initialBalance' in data and 'initial_capital' not in data:
            data['initial_capital'] = data['initialBalance']
        
        # 3. Pass THE WHOLE DATA OBJECT to the controller
        # This ensures 'strategies' and 'mlMode' are actually received by the logic.
        result = await system_controller.run_backtest_logic(**data)
        
        return result
        
    except FileNotFoundError as fe:
        return JSONResponse(content={"error": str(fe)}, status_code=404)
    except Exception as e:
        logger.error(f"Backtest API Error: {str(e)}")
        # Print stack trace to logs for easier debugging
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/bot/winners")
async def get_winners():
    return await system_controller.get_optimizer_winners()

@app.get("/api/ml/available-models")
def list_models():
    models = system_controller.get_models_logic()
    return {"status": "success", "count": len(models), "models": models}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
