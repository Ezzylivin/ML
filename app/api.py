import logging
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import the controller singleton
from controller import system_controller

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

class StopBotRequest(BaseModel):
    userId: str

class BacktestRequest(BaseModel):
    symbol: str = Field(default="BTC-USD", description="Trading pair")
    timeframe: str = Field(default="1h", description="Candle timeframe")
    initial_capital: float = Field(default=1000.0, description="Start balance")
    # New fields added for the updated service:
    limit: int = Field(default=1000, description="Number of candles to fetch")
    risk_percentage: float = Field(default=1.0, description="Risk per trade %")
    risk_mode: str = Field(default="static", description="Risk mode: static or dynamic")
# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "online", "role": "api_gateway"}

@app.post('/api/bot/start')
async def start_bot(request: StartBotRequest, bg: BackgroundTasks):
    try:
        bot_id, mode = system_controller.start_bot_logic(
            request.userId, 
            request.symbol, 
            request.initialBalance, 
            request.strategies, 
            bg
        )
        return {"status": "started", "botId": bot_id, "mode": mode}
    except ValueError as ve:
        return JSONResponse(content={"status": "error", "message": str(ve)}, status_code=400)
    except Exception as e:
        logger.error(f"Start API Error: {str(e)}")
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)

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

@app.post("/api/backtest/run")
def run_backtest(req: BacktestRequest):
    try:
       result = await system_controller.run_backtest_logic(
            symbol=req.symbol,
            timeframe=req.timeframe,
            limit=req.limit,
            initial_balance=req.initial_capital,
            risk_percentage=req.risk_percentage,
            risk_mode=req.risk_mode
        )
        return result
    except FileNotFoundError as fe:
        return JSONResponse(content={"error": str(fe)}, status_code=404)
    except Exception as e:
        logger.error(f"Backtest API Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/bot/winners")
def get_winners():
    try:
        return system_controller.get_winners_logic()
    except Exception as e:
        logger.error(f"Winners API Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
